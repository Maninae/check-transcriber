"""CLI: compose one scene on demand and write its photo and label.

Usage:
    python -m scene_composer.generate_one --seed 7 --out /tmp/x.jpg --labels /tmp/x.json
    python -m scene_composer.generate_one --seed 7 --split eval --scene-index 3 --out /tmp/x.jpg --labels /tmp/x.json
    python -m scene_composer.generate_one --seed 7 --backgrounds /path/to/backgrounds --out /tmp/x.jpg --labels /tmp/x.json

The JPEG is saved exactly as dataset scenes are. The label JSON is the dataset annotation for the same
(seed, split, scene index), plus a `provenance` block (generator version, seed, split, scene index).
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from scene_composer.on_demand import compose_scene_on_demand
from synthetic_checks.check_templates import DEFAULT_TEMPLATE_COUNT
from synthetic_checks.splits import SPLIT_NAMES
from synthetic_data_paths import BACKGROUND_DIR

logger = logging.getLogger(__name__)


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, required=True, help="split seed and scene seed (as build_dataset --seed)")
    parser.add_argument("--split", choices=SPLIT_NAMES, default="train", help="draw only from this split's pools")
    parser.add_argument("--scene-index", type=int, default=0, help="which scene of the split")
    parser.add_argument("--out", type=Path, required=True, help="output JPEG path")
    parser.add_argument("--labels", type=Path, required=True, help="output label JSON path")
    parser.add_argument("--backgrounds", type=Path, default=BACKGROUND_DIR, help="background root (accepted subfolders only)")
    parser.add_argument("--templates", type=int, default=DEFAULT_TEMPLATE_COUNT, help="size of the template catalog")
    parser.add_argument("--harmonize", action="store_true", help="run PCT-Net harmonization (needs torch)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for `python -m scene_composer.generate_one`."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    arguments = parse_arguments(sys.argv[1:] if argv is None else argv)
    composed = compose_scene_on_demand(arguments.seed, arguments.split, arguments.scene_index,
                                       backgrounds=arguments.backgrounds, template_count=arguments.templates,
                                       harmonize=arguments.harmonize)
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.labels.parent.mkdir(parents=True, exist_ok=True)
    composed.write_jpeg(arguments.out)
    arguments.labels.write_text(json.dumps({**composed.label_record(), "provenance": composed.provenance()}, indent=1) + "\n")
    logger.info("%s: %d checks on %s -> %s, %s", composed.label.scene_id, len(composed.label.checks),
                composed.label.background_id, arguments.out, arguments.labels)


if __name__ == "__main__":
    main()
