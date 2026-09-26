"""Run a trained CenterNet checkpoint over a split and write a predictions file.

    python -m experiments.detection.learned.run_centernet_detector \
        --weights <best.pt> --split val --output <predictions.json> [--limit N]

Mirrors `run_yolo_detector.py`: full-resolution scene JPEGs in, `DetectedCheck`s with
ordered TL,TR,BR,BL corners out (`orientation_known=True`), and latency per image
(downscale + model + decode, excluding JPEG decode) in the header as `seconds_per_image`.
"""

import argparse
import logging
import time
from pathlib import Path

import cv2

from experiments.detection.dataset.scene_annotations import load_split_scene_annotations
from experiments.detection.learned.centernet.centernet_decoding import DEFAULT_SCORE_THRESHOLD
from experiments.detection.learned.centernet.centernet_inference import CenterNetCheckPredictor
from experiments.detection.predictions.detected_check import save_predictions_file

logger = logging.getLogger(__name__)


def main() -> None:
    """Detect checks in every scene of a split and save the predictions."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--confidence", type=float, default=DEFAULT_SCORE_THRESHOLD)
    parser.add_argument("--limit", type=int, default=None)
    arguments = parser.parse_args()

    predictor = CenterNetCheckPredictor(arguments.weights, arguments.device, arguments.confidence)
    scenes = load_split_scene_annotations(arguments.split, limit=arguments.limit)
    predictions_by_scene_id = {}
    total_detection_seconds = 0.0
    for scene_number, scene in enumerate(scenes):
        image_bgr = cv2.imread(str(scene.image_path), cv2.IMREAD_COLOR)
        start_time = time.perf_counter()
        predictions_by_scene_id[scene.scene_id] = predictor.detect_checks(image_bgr)
        total_detection_seconds += time.perf_counter() - start_time
        if scene_number % 100 == 0:
            logger.info("%d/%d scenes", scene_number, len(scenes))
    save_predictions_file(
        arguments.output,
        detector_name="centernet-mobilenetv3",
        split_name=arguments.split,
        predictions_by_scene_id=predictions_by_scene_id,
        detector_config={
            "weights": str(arguments.weights),
            "image_size": predictor.input_size_pixels,
            "backbone": predictor.training_config.backbone_name,
            "confidence_threshold": arguments.confidence,
            "device": arguments.device,
            "seconds_per_image": total_detection_seconds / max(len(scenes), 1),
        },
    )
    logger.info("wrote %s", arguments.output)


if __name__ == "__main__":
    main()
