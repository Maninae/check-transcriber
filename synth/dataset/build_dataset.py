"""Stage 3 CLI: build a split synthetic dataset of phone-photo scenes.

Usage:
    python -m synth.dataset.build_dataset --output DIR --scenes 200 --seed 7
    python -m synth.dataset.build_dataset --output DIR --scenes 200 --procedural-backgrounds 12
    python -m synth.dataset.build_dataset --output DIR --print-sheets 4

Output layout (Ultralytics-compatible):
    DIR/manifest.json, DIR/data.yaml
    DIR/{train,val,eval}/images/*.jpg          scene photos
    DIR/{train,val,eval}/annotations/*.json    full scene labels (see compose/scene_label.py)
    DIR/{train,val,eval}/labels/*.txt          YOLO segmentation polygons
    DIR/{train,val,eval}/labels_obb/*.txt      YOLO oriented boxes
    DIR/{train,val,eval}/annotations_coco.json COCO polygons + corner keypoints
"""

import argparse
import datetime
import json
import logging
import multiprocessing
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

from synth import GENERATOR_VERSION
from synth.backgrounds.loader import list_background_sources
from synth.backgrounds.procedural import write_procedural_backgrounds
from synth.compose.compose_scene import SceneConfig
from synth.dataset.annotation_exports import write_coco_for_split, write_yolo_data_yaml
from synth.dataset.manifest import DatasetManifest, write_manifest
from synth.dataset.scene_worker import SceneTask, generate_scene, worker_initializer
from synth.dataset.splits import (
    BACKGROUND_SALT,
    DEFAULT_SPLIT_FRACTIONS,
    SPLIT_NAMES,
    TEMPLATE_SALT,
    assign_ids_to_splits,
    scene_counts_per_split,
)
from synth.print.print_sheets import write_print_sheets
from synth.paths import BACKGROUND_DIR, DATA_ROOT, SYNTH_OUTPUT_DIR
from synth.render.check_templates import DEFAULT_TEMPLATE_COUNT, build_template_catalog

logger = logging.getLogger(__name__)

PROCEDURAL_BACKGROUND_DIR = DATA_ROOT / "procedural-backgrounds"
SPLIT_SUBDIRECTORIES = ("images", "annotations", "labels", "labels_obb")


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=SYNTH_OUTPUT_DIR / "dataset", help="output directory (must be empty or new)")
    parser.add_argument("--scenes", type=int, default=100, help="total scenes across all splits")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--templates", type=int, default=DEFAULT_TEMPLATE_COUNT, help="size of the template catalog")
    parser.add_argument("--backgrounds", type=Path, default=BACKGROUND_DIR, help="root scanned recursively for JPEG/PNG")
    parser.add_argument("--procedural-backgrounds", type=int, default=0,
                        help="use N procedural fabrics instead of --backgrounds (for tests or before real backgrounds exist)")
    parser.add_argument("--harmonize", action=argparse.BooleanOptionalAction, default=False,
                        help="run PCT-Net harmonization on each pasted check (needs torch)")
    parser.add_argument("--harmonize-blend", type=float, default=0.5)
    parser.add_argument("--print-sheets", type=int, default=0, metavar="PAGES",
                        help="instead of scenes, write PAGES print-ready Letter PDF pages of fake checks + labels CSV")
    return parser.parse_args(argv)


def prepare_output_directory(output_directory: Path) -> None:
    """Create the split tree; refuse to write into a non-empty directory."""
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_directory}")
    for split_name in SPLIT_NAMES:
        for subdirectory in SPLIT_SUBDIRECTORIES:
            (output_directory / split_name / subdirectory).mkdir(parents=True, exist_ok=True)


def build_dataset(arguments: argparse.Namespace) -> DatasetManifest:
    """Generate every scene in parallel and write exports and the manifest."""
    output_directory = arguments.output.resolve()
    background_root = arguments.backgrounds
    if arguments.procedural_backgrounds:
        background_root = PROCEDURAL_BACKGROUND_DIR / f"seed_{arguments.seed}"
        write_procedural_backgrounds(background_root, arguments.procedural_backgrounds, arguments.seed)
    background_sources = list_background_sources(background_root)
    if len(background_sources) < len(SPLIT_NAMES):
        raise SystemExit(f"need at least 3 backgrounds under {background_root}, found {len(background_sources)}; "
                         "add photos, wait for the FLUX batch, or pass --procedural-backgrounds N")
    prepare_output_directory(output_directory)

    template_ids = [template.template_id for template in build_template_catalog(arguments.templates)]
    templates_by_split = assign_ids_to_splits(template_ids, DEFAULT_SPLIT_FRACTIONS, arguments.seed, TEMPLATE_SALT)
    background_paths = {source.background_id: str(source.file_path) for source in background_sources}
    backgrounds_by_split = assign_ids_to_splits(sorted(background_paths), DEFAULT_SPLIT_FRACTIONS, arguments.seed, BACKGROUND_SALT)
    scene_counts = scene_counts_per_split(arguments.scenes, DEFAULT_SPLIT_FRACTIONS)
    scene_config = SceneConfig(harmonize=arguments.harmonize, harmonize_blend=arguments.harmonize_blend)

    tasks = [
        SceneTask(arguments.seed, split_name, scene_index, tuple(templates_by_split[split_name]),
                  tuple((background_id, background_paths[background_id]) for background_id in backgrounds_by_split[split_name]),
                  str(output_directory / split_name), arguments.templates, scene_config)
        for split_name in SPLIT_NAMES for scene_index in range(scene_counts[split_name])
    ]
    check_counts = {split_name: 0 for split_name in SPLIT_NAMES}
    report_every = max(1, len(tasks) // 20)
    start_time = time.time()
    context = multiprocessing.get_context("spawn")
    with context.Pool(arguments.workers, initializer=worker_initializer, initargs=(arguments.harmonize,)) as pool:
        for completed, summary in enumerate(pool.imap_unordered(generate_scene, tasks, chunksize=2), start=1):
            check_counts[summary["split"]] += summary["check_count"]
            if completed % report_every == 0 or completed == len(tasks):
                elapsed = time.time() - start_time
                rate = completed / elapsed
                print(f"[{completed}/{len(tasks)}] {rate:.2f} scenes/s, eta {(len(tasks) - completed) / rate:.0f}s", flush=True)

    for split_name in SPLIT_NAMES:
        write_coco_for_split(output_directory / split_name)
    write_yolo_data_yaml(output_directory)
    elapsed = time.time() - start_time
    manifest = DatasetManifest(
        generator_version=GENERATOR_VERSION, seed=arguments.seed,
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        command=" ".join(["python -m synth.dataset.build_dataset", *sys.argv[1:]]),
        scene_counts=scene_counts, check_counts=check_counts,
        template_ids_by_split=templates_by_split, background_ids_by_split=backgrounds_by_split,
        background_root=str(background_root), harmonized=arguments.harmonize, harmonize_blend=arguments.harmonize_blend,
        scene_config=json.loads(json.dumps(asdict(scene_config))),  # tuples -> lists, as they read back
        split_fractions=DEFAULT_SPLIT_FRACTIONS,
        notes=[f"built {len(tasks)} scenes in {elapsed:.1f}s with {arguments.workers} workers ({len(tasks) / elapsed:.2f} scenes/s)",
               "all names, addresses, banks and routing numbers are fake; routing numbers fail the ABA checksum on purpose"],
    )
    write_manifest(manifest, output_directory)
    print(f"done: {len(tasks)} scenes, {sum(check_counts.values())} checks, {len(tasks) / elapsed:.2f} scenes/s -> {output_directory}")
    return manifest


def main(argv: list[str] | None = None) -> None:
    """Entry point for `python -m synth.dataset.build_dataset`."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    arguments = parse_arguments(sys.argv[1:] if argv is None else argv)
    if arguments.print_sheets:
        write_print_sheets(arguments.output, arguments.print_sheets, arguments.seed, arguments.templates)
        return
    build_dataset(arguments)


if __name__ == "__main__":
    main()
