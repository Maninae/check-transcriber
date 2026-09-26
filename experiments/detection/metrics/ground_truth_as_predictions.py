"""Write a split's GT corners as a predictions file: the metrics sanity check.

Scoring this file must give IoU vs corner quad 1.0, corner error ~0 and orientation
100%; IoU vs outline lands slightly under 1 because deformed outlines are not quads.

    python -m experiments.detection.metrics.ground_truth_as_predictions \\
        --split val --output /Volumes/vega/.../gt_as_predictions__val.json
"""

import argparse
import logging
from pathlib import Path

from experiments.detection.config.paths import SPLIT_NAMES
from experiments.detection.dataset.scene_annotations import SceneAnnotation, load_split_scene_annotations
from experiments.detection.predictions.detected_check import DetectedCheck, save_predictions_file

logger = logging.getLogger(__name__)

GROUND_TRUTH_DETECTOR_NAME = "ground_truth_corners"


def ground_truth_predictions_by_scene_id(scene_annotations: list[SceneAnnotation]) -> dict[str, list[DetectedCheck]]:
    """Every GT check as a score-1 prediction with its own TL/TR/BR/BL order (orientation known)."""
    return {
        scene.scene_id: [
            DetectedCheck(corners=check.corners.copy(), score=1.0, orientation_known=True) for check in scene.checks
        ]
        for scene in scene_annotations
    }


def main() -> None:
    """Load a split's GT and save it as a predictions file."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Write GT corners as a predictions file.")
    parser.add_argument("--split", choices=SPLIT_NAMES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    arguments = parser.parse_args()
    scene_annotations = load_split_scene_annotations(arguments.split, limit=arguments.limit)
    save_predictions_file(
        arguments.output,
        GROUND_TRUTH_DETECTOR_NAME,
        arguments.split,
        ground_truth_predictions_by_scene_id(scene_annotations),
        detector_config={"source": "scene annotation corners"},
    )
    logger.info("wrote %d scenes to %s", len(scene_annotations), arguments.output)


if __name__ == "__main__":
    main()
