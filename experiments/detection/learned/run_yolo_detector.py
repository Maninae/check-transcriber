"""Run a trained YOLO checkpoint over a split and write a predictions file.

    python -m experiments.detection.learned.run_yolo_detector \
        --weights <best.pt> --split val --output <predictions.json> [--refine]

`--refine` passes every detection through the corner-refinement stage
(`experiments.detection.refinement`) before saving. Latency per image (model plus
refinement, excluding JPEG decode) is recorded in the file header.
"""

import argparse
import logging
import time
from pathlib import Path

import cv2

from experiments.detection.dataset.scene_annotations import load_split_scene_annotations
from experiments.detection.learned.yolo_inference import DEFAULT_CONFIDENCE_THRESHOLD, YoloCheckDetector
from experiments.detection.predictions.detected_check import save_predictions_file
from experiments.detection.refinement.quadrilateral_refinement import refine_detected_checks

logger = logging.getLogger(__name__)


def main() -> None:
    """Detect checks in every scene of a split and save the predictions."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--refine", action="store_true")
    arguments = parser.parse_args()

    detector = YoloCheckDetector(
        arguments.weights, arguments.image_size, arguments.device, arguments.confidence
    )
    scenes = load_split_scene_annotations(arguments.split, limit=arguments.limit)
    predictions_by_scene_id = {}
    total_detection_seconds = 0.0
    for scene_number, scene in enumerate(scenes):
        image_bgr = cv2.imread(str(scene.image_path), cv2.IMREAD_COLOR)
        start_time = time.perf_counter()
        detections = detector.detect_checks(image_bgr)
        if arguments.refine:
            detections = refine_detected_checks(image_bgr, detections)
        total_detection_seconds += time.perf_counter() - start_time
        predictions_by_scene_id[scene.scene_id] = detections
        if scene_number % 100 == 0:
            logger.info("%d/%d scenes", scene_number, len(scenes))
    save_predictions_file(
        arguments.output,
        detector_name=f"yolo-{detector.task_name}{'+refine' if arguments.refine else ''}",
        split_name=arguments.split,
        predictions_by_scene_id=predictions_by_scene_id,
        detector_config={
            "weights": arguments.weights,
            "image_size": arguments.image_size,
            "confidence_threshold": arguments.confidence,
            "device": arguments.device,
            "seconds_per_image": total_detection_seconds / max(len(scenes), 1),
        },
    )
    logger.info("wrote %s", arguments.output)


if __name__ == "__main__":
    main()
