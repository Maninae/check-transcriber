"""Stage 3 CLI: build a split synthetic dataset of phone-photo scenes, its exports and OCR manifest.

Usage:
    python -m dataset_builder.build_dataset --output DIR --scenes 200 --seed 7 --workers 2
    python -m dataset_builder.build_dataset --output DIR --scenes 200 --procedural-backgrounds 12
    python -m dataset_builder.build_dataset --output DIR --scenes 5000 --plan-only     # pools + counts, renders nothing
    python -m dataset_builder.build_dataset --output DIR --scenes 5000 --resume        # continue an interrupted build
    python -m dataset_builder.build_dataset --output DIR --print-sheets 4              # printable gold-set checks

Output layout (Ultralytics-compatible):
    DIR/manifest.json, DIR/build_plan.json, DIR/failures.jsonl (only if something failed)
    DIR/data.yaml, DIR/data_obb.yaml, DIR/yolo_obb/<split>/{images,labels}  (symlinks)
    DIR/{train,val,eval}/images/*.jpg           scene photos
    DIR/{train,val,eval}/annotations/*.json     full scene labels (see scene_composer/scene_label.py)
    DIR/{train,val,eval}/labels/*.txt           YOLO segmentation (paper outline)
    DIR/{train,val,eval}/labels_obb/*.txt       YOLO oriented boxes (4 corners)
    DIR/{train,val,eval}/annotations_coco.json  COCO: checks (outline + corner keypoints) and fields (with text)
    DIR/ocr/...                                 per-field OCR manifest and crops (see ocr_manifest.py)
"""

import argparse
import datetime
import logging
import multiprocessing
import os
import sys
import time
from pathlib import Path

from dataset_builder.build_plan import (
    BuildPlan,
    format_build_plan,
    make_build_plan,
    require_matching_build_plan,
    scene_config_from_plan,
    write_build_plan,
)
from dataset_builder.build_progress import StageProgress
from dataset_builder.exports.annotation_exports import write_yolo_data_yaml
from dataset_builder.exports.coco_export import write_coco_for_split
from dataset_builder.manifest import DatasetManifest, current_git_commit, write_manifest
from dataset_builder.ocr.ocr_manifest import prepare_ocr_directories, write_ocr_rows_for_scene, write_split_ocr_manifest
from dataset_builder.scene_task_runner import run_scene_task
from dataset_builder.scene_worker import SceneTask, worker_initializer
from scene_composer import GENERATOR_VERSION
from scene_composer.compose_scene import SceneConfig
from scene_composer.scene_ingredient_pools import ingredient_pools_for_split
from synthetic_checks.check_templates import DEFAULT_TEMPLATE_COUNT
from synthetic_checks.print_sheets import write_print_sheets
from synthetic_checks.splits import SPLIT_NAMES
from synthetic_data_paths import BACKGROUND_DIR, DATASET_OUTPUT_DIR

logger = logging.getLogger(__name__)

SPLIT_SUBDIRECTORIES = ("images", "annotations", "labels", "labels_obb")
DEFAULT_OCR_SPLITS = "val,eval"
DEFAULT_WORKER_COUNT = max(1, min(4, (os.cpu_count() or 2) - 1))


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=DATASET_OUTPUT_DIR / "dataset", help="output directory (new or empty, unless --resume)")
    parser.add_argument("--scenes", type=int, default=100, help="total scenes across all splits")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKER_COUNT)
    parser.add_argument("--templates", type=int, default=DEFAULT_TEMPLATE_COUNT, help="size of the template catalog")
    parser.add_argument("--backgrounds", type=Path, default=BACKGROUND_DIR, help="background root; only its accepted subfolders (flux/, photos/, web/) are scanned")
    parser.add_argument("--procedural-backgrounds", type=int, default=0,
                        help="use N procedural fabrics instead of --backgrounds (for tests or before real backgrounds exist)")
    parser.add_argument("--ocr-splits", default=DEFAULT_OCR_SPLITS,
                        help="comma-separated splits that get the per-field OCR manifest and crops ('none' to skip)")
    parser.add_argument("--plan-only", action="store_true", help="print split pools and counts, render nothing")
    parser.add_argument("--resume", action="store_true", help="continue a build in a non-empty --output with the same plan")
    parser.add_argument("--harmonize", action=argparse.BooleanOptionalAction, default=False,
                        help="run PCT-Net harmonization on each pasted check (needs torch)")
    parser.add_argument("--harmonize-blend", type=float, default=0.5)
    parser.add_argument("--print-sheets", type=int, default=0, metavar="PAGES",
                        help="instead of scenes, write PAGES print-ready Letter PDF pages of fake checks + labels")
    parser.add_argument("--print-pool", choices=("all", "eval"), default="all",
                        help="print sheets from every template and font, or only this seed's eval-split pools")
    return parser.parse_args(argv)


def parse_ocr_splits(text: str) -> list[str]:
    """`val,eval` -> ['val', 'eval']; 'none' -> []."""
    names = [] if text.strip() in ("", "none") else [name.strip() for name in text.split(",")]
    unknown = set(names) - set(SPLIT_NAMES)
    if unknown:
        raise SystemExit(f"--ocr-splits: unknown split(s) {sorted(unknown)}; choose from {SPLIT_NAMES}")
    return [name for name in SPLIT_NAMES if name in names]


def prepare_output_directory(output_directory: Path, plan: BuildPlan, resume: bool) -> None:
    """Create the split tree; refuse a non-empty directory unless resuming the same plan."""
    if output_directory.exists() and any(output_directory.iterdir()):
        if not resume:
            raise FileExistsError(f"output directory is not empty: {output_directory} (pass --resume to continue it)")
        require_matching_build_plan(plan, output_directory)
    for split_name in SPLIT_NAMES:
        for subdirectory in SPLIT_SUBDIRECTORIES:
            (output_directory / split_name / subdirectory).mkdir(parents=True, exist_ok=True)
    write_build_plan(plan, output_directory)


def scene_tasks_for_plan(plan: BuildPlan, output_directory: Path) -> list[SceneTask]:
    """One task per scene; each carries only its own split's pools."""
    scene_config = scene_config_from_plan(plan)
    tasks = []
    for split_name in SPLIT_NAMES:
        pools = ingredient_pools_for_split(plan.split_pools[split_name], Path(plan.background_root), plan.background_paths)
        for scene_index in range(plan.scene_counts[split_name]):
            tasks.append(SceneTask(plan.seed, split_name, scene_index, pools, str(output_directory / split_name),
                                   plan.template_count, scene_config))
    return tasks


def run_scene_stage(pool, tasks: list[SceneTask], output_directory: Path) -> tuple[dict[str, int], StageProgress]:
    """Generate (or skip) every scene; returns check counts per split and the stage's progress record."""
    progress = StageProgress("scenes", len(tasks), output_directory)
    check_counts = {split_name: 0 for split_name in SPLIT_NAMES}
    for summary in pool.imap_unordered(run_scene_task, tasks, chunksize=1):
        check_counts[summary["split"]] += summary["check_count"]
        progress.record(summary["scene_id"], summary["status"],
                        {"error": summary.get("error"), "traceback": summary.get("traceback")})
    return check_counts, progress


def run_ocr_stage(pool, output_directory: Path, ocr_splits: list[str]) -> tuple[dict[str, dict[str, int]], StageProgress]:
    """Crop every field of every finished scene in the OCR splits and write one JSONL per split."""
    prepare_ocr_directories(output_directory, ocr_splits)
    jobs = [(str(output_directory), split_name, path.stem) for split_name in ocr_splits
            for path in sorted((output_directory / split_name / "annotations").glob("*.json"))]
    progress = StageProgress("ocr", len(jobs), output_directory)
    rows_by_split: dict[str, list[dict]] = {split_name: [] for split_name in ocr_splits}
    for job, result in zip(jobs, pool.imap(write_ocr_rows_for_scene, jobs, chunksize=1)):
        rows_by_split[job[1]].extend(result["rows"])
        progress.record(result["scene_id"], result["status"], {"error": result.get("error")})
    row_counts = {}
    for split_name, rows in rows_by_split.items():
        write_split_ocr_manifest(output_directory, split_name, rows)
        row_counts[split_name] = {"rows": len(rows), "usable": sum(row["usable"] for row in rows),
                                  "handwritten_usable": sum(row["usable"] and row["handwritten"] for row in rows)}
    return row_counts, progress


def build_dataset(arguments: argparse.Namespace) -> DatasetManifest | None:
    """Plan, generate every scene in parallel, crop OCR fields, and write exports and the manifest."""
    output_directory = arguments.output.resolve()
    plan = make_build_plan(arguments.seed, arguments.scenes, arguments.templates, arguments.backgrounds,
                           arguments.procedural_backgrounds,
                           SceneConfig(harmonize=arguments.harmonize, harmonize_blend=arguments.harmonize_blend),
                           parse_ocr_splits(arguments.ocr_splits))
    if arguments.plan_only:
        print(format_build_plan(plan))
        return None
    prepare_output_directory(output_directory, plan, arguments.resume)
    start_time = time.time()
    context = multiprocessing.get_context("spawn")
    with context.Pool(arguments.workers, initializer=worker_initializer, initargs=(arguments.harmonize,)) as pool:
        check_counts, scene_progress = run_scene_stage(pool, scene_tasks_for_plan(plan, output_directory), output_directory)
        ocr_row_counts, ocr_progress = run_ocr_stage(pool, output_directory, plan.ocr_splits)
    export_start = time.time()
    for split_name in SPLIT_NAMES:
        write_coco_for_split(output_directory / split_name)
    write_yolo_data_yaml(output_directory)
    timings = {"scenes": scene_progress.elapsed_seconds(), "ocr": ocr_progress.elapsed_seconds(),
               "exports": round(time.time() - export_start, 1), "total": round(time.time() - start_time, 1)}
    built = scene_progress.counts["built"]
    manifest = DatasetManifest(
        generator_version=GENERATOR_VERSION, git_commit=current_git_commit(), seed=plan.seed,
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        command=" ".join(["python -m dataset_builder.build_dataset", *sys.argv[1:]]), workers=arguments.workers,
        template_count=plan.template_count, scene_counts=plan.scene_counts, check_counts=check_counts,
        failed_scene_ids=sorted(scene_progress.failed_ids), split_pools=plan.split_pools,
        split_fractions=plan.split_fractions, background_root=plan.background_root,
        harmonized=arguments.harmonize, harmonize_blend=arguments.harmonize_blend, scene_config=plan.scene_config,
        ocr_splits=plan.ocr_splits, ocr_row_counts=ocr_row_counts, timings_seconds=timings,
        notes=[f"this run built {built} scenes, skipped {scene_progress.counts['skipped']} already built, "
               f"failed {scene_progress.counts['failed']}; {arguments.workers} workers",
               "all names, addresses, banks and routing numbers are fake; routing numbers fail the ABA checksum on purpose"],
    )
    write_manifest(manifest, output_directory)
    print(f"done: {sum(plan.scene_counts.values())} scenes ({built} built now), {sum(check_counts.values())} checks, "
          f"{len(manifest.failed_scene_ids)} failed, {timings['total']}s -> {output_directory}")
    return manifest


def main(argv: list[str] | None = None) -> None:
    """Entry point for `python -m dataset_builder.build_dataset`."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    arguments = parse_arguments(sys.argv[1:] if argv is None else argv)
    if arguments.print_sheets:
        write_print_sheets(arguments.output, arguments.print_sheets, arguments.seed, arguments.templates,
                           held_out_split="eval" if arguments.print_pool == "eval" else None)
        return
    build_dataset(arguments)


if __name__ == "__main__":
    main()
