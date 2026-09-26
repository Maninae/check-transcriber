"""Run the classical detector over a split and save one predictions JSON.

    python -m experiments.detection.classical.run_classical_detector \\
        --split val --limit 100 --workers 4 --output <dir>/predictions.json [--score]

- `seconds_per_image` in the file's config header is the mean wall time of
  `detect_checks_classical` per image inside a worker (image decode excluded), so it
  does not depend on the worker count.
- `--config-json` overrides config fields from a JSON dict (the tuning sweep's output).
- `--score` also scores in-process with `experiments.detection.metrics` and prints the
  headline line (only for val/eval).
"""

import argparse
import dataclasses
import json
import logging
import time
from multiprocessing import Pool
from pathlib import Path

import cv2

from experiments.detection.classical.classical_detector_config import ClassicalDetectorConfig
from experiments.detection.classical.detect_checks_classical import detect_checks_classical
from experiments.detection.dataset.scene_annotations import SceneAnnotation, load_split_scene_annotations
from experiments.detection.metrics.metrics_report_writing import headline_summary_line
from experiments.detection.metrics.score_predictions import score_predictions_against_split
from experiments.detection.predictions.detected_check import DetectedCheck, save_predictions_file

logger = logging.getLogger(__name__)

DETECTOR_NAME = "classical_opencv"
MAXIMUM_WORKERS = 4  # the machine is shared; more workers starve other jobs


def detect_one_scene(task: tuple[str, str, ClassicalDetectorConfig]) -> tuple[str, list[DetectedCheck], float]:
    """Worker entry: read one image, detect, return (scene_id, detections, detect seconds)."""
    scene_id, image_path, config = task
    cv2.setNumThreads(1)
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"cannot read {image_path}")
    start_time = time.perf_counter()
    detections = detect_checks_classical(image_bgr, config)
    return scene_id, detections, time.perf_counter() - start_time


def run_detector_on_scenes(
    scenes: list[SceneAnnotation], config: ClassicalDetectorConfig, worker_count: int
) -> tuple[dict[str, list[DetectedCheck]], float]:
    """Detect every scene; returns predictions by scene id and mean seconds per image."""
    tasks = [(scene.scene_id, str(scene.image_path), config) for scene in scenes]
    predictions_by_scene_id: dict[str, list[DetectedCheck]] = {}
    total_detect_seconds = 0.0
    worker_count = max(1, min(worker_count, MAXIMUM_WORKERS))
    with Pool(worker_count) as worker_pool:
        for scene_id, detections, detect_seconds in worker_pool.imap_unordered(detect_one_scene, tasks, chunksize=2):
            predictions_by_scene_id[scene_id] = detections
            total_detect_seconds += detect_seconds
    return predictions_by_scene_id, total_detect_seconds / max(len(scenes), 1)


def build_config_from_overrides(config_overrides: dict) -> ClassicalDetectorConfig:
    """Default config with fields replaced from a dict (lists become tuples)."""
    normalized_overrides = {
        key: tuple(value) if isinstance(value, list) else value for key, value in config_overrides.items()
    }
    return dataclasses.replace(ClassicalDetectorConfig(), **normalized_overrides)


def parse_command_line_arguments() -> argparse.Namespace:
    """CLI flags."""
    parser = argparse.ArgumentParser(description="Run the classical OpenCV check detector on a split.")
    parser.add_argument("--split", choices=("train", "val", "eval"), required=True)
    parser.add_argument("--limit", type=int, default=None, help="first N scenes by id")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True, help="predictions JSON path")
    parser.add_argument("--config-json", type=Path, default=None, help="JSON dict of config overrides")
    parser.add_argument("--score", action="store_true", help="also score in-process and print the headline")
    return parser.parse_args()


def main() -> None:
    """Detect, save predictions, optionally score."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    arguments = parse_command_line_arguments()
    config_overrides = json.loads(arguments.config_json.read_text()) if arguments.config_json else {}
    config = build_config_from_overrides(config_overrides)
    scenes = load_split_scene_annotations(arguments.split, limit=arguments.limit)
    logger.info("detecting %d %s scenes with %d workers", len(scenes), arguments.split, arguments.workers)
    predictions_by_scene_id, seconds_per_image = run_detector_on_scenes(scenes, config, arguments.workers)
    header_config = config.to_json_dict() | {"seconds_per_image": round(seconds_per_image, 4)}
    save_predictions_file(arguments.output, DETECTOR_NAME, arguments.split, predictions_by_scene_id, header_config)
    logger.info("wrote %s (%.3f s/image)", arguments.output, seconds_per_image)
    if arguments.score and arguments.split in ("val", "eval"):
        metrics = score_predictions_against_split(
            predictions_by_scene_id, scenes, detector_config=header_config, detector_name=DETECTOR_NAME, include_records=False
        )
        print(headline_summary_line(metrics))


if __name__ == "__main__":
    main()
