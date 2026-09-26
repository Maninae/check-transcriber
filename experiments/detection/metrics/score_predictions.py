"""Score a detector's predictions against a split: the in-process API and the CLI.

In-process (tuning loops):
    metrics = score_predictions_against_split(predictions_by_scene_id, scene_annotations)
    print(headline_summary_line(metrics))

CLI (writes metrics.json and metrics.md):
    python -m experiments.detection.metrics.score_predictions \\
        --predictions run/predictions.json --split val --output-dir run/metrics

- The score threshold drops low-score predictions before every metric (matching,
  counts, precision), so one number defines what the detector "returned".
- A scene with no entry in the predictions file counts as zero predictions (and is
  tallied in `scenes.scenes_missing_from_predictions`).
"""

import argparse
import dataclasses
import logging
from pathlib import Path

from experiments.detection.dataset.scene_annotations import SceneAnnotation, load_split_scene_annotations
from experiments.detection.metrics.attribute_breakdowns import all_breakdowns
from experiments.detection.metrics.metrics_report_writing import headline_summary_line, write_metrics_report
from experiments.detection.metrics.scene_scoring import score_scene
from experiments.detection.metrics.split_aggregation import (
    detection_rates_by_threshold,
    localization_summary,
    scene_summary,
)
from experiments.detection.predictions.detected_check import DetectedCheck, load_predictions_file

logger = logging.getLogger(__name__)

SCORABLE_SPLIT_NAMES = ("val", "eval")
DEFAULT_SCORE_THRESHOLD = 0.0


def score_predictions_against_split(
    predictions_by_scene_id: dict[str, list[DetectedCheck]],
    scene_annotations: list[SceneAnnotation],
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    detector_config: dict | None = None,
    detector_name: str = "unnamed_detector",
    include_records: bool = True,
) -> dict:
    """Score predictions against GT scenes; returns the full metrics dict.

    Args:
        predictions_by_scene_id: scene_id -> detections in full-resolution pixels.
        scene_annotations: the GT scenes to score (predictions for other scenes are ignored).
        score_threshold: predictions with score below this are dropped before scoring.
        detector_config: the predictions file header config; `seconds_per_image` is reported.
        detector_name: label for the report title.
        include_records: False skips the per-scene/per-check record lists (lighter for tuning loops).
    Returns:
        {"run", "detection", "localization", "scenes", "latency", "breakdowns",
         "per_scene_records", "per_check_records"}.
    """
    if not scene_annotations:
        raise ValueError("no scenes to score")
    scene_records, check_records = [], []
    for scene in scene_annotations:
        scene_record, scene_check_records = score_scene(
            scene, predictions_by_scene_id.get(scene.scene_id), score_threshold
        )
        scene_records.append(scene_record)
        check_records.extend(scene_check_records)
    scored_scene_ids = {scene.scene_id for scene in scene_annotations}
    seconds_per_image = (detector_config or {}).get("seconds_per_image")
    metrics = {
        "run": {
            "detector_name": detector_name,
            "split_name": scene_annotations[0].split_name,
            "score_threshold": score_threshold,
            "prediction_scenes_not_scored": sum(
                scene_id not in scored_scene_ids for scene_id in predictions_by_scene_id
            ),
            "detector_config": detector_config or {},
        },
        "detection": detection_rates_by_threshold(scene_records, check_records),
        "localization": localization_summary(check_records),
        "scenes": scene_summary(scene_records),
        "latency": {"seconds_per_image": None if seconds_per_image is None else float(seconds_per_image)},
        "breakdowns": all_breakdowns(scene_records, check_records),
    }
    if include_records:
        metrics["per_scene_records"] = [dataclasses.asdict(record) for record in scene_records]
        metrics["per_check_records"] = [dataclasses.asdict(record) for record in check_records]
    return metrics


def parse_command_line_arguments() -> argparse.Namespace:
    """CLI flags for scoring one predictions file."""
    parser = argparse.ArgumentParser(description="Score check-detector predictions against a split.")
    parser.add_argument("--predictions", type=Path, required=True, help="predictions JSON (detected_check format)")
    parser.add_argument("--split", choices=SCORABLE_SPLIT_NAMES, required=True)
    parser.add_argument("--limit", type=int, default=None, help="score only the first N scenes (sorted by id)")
    parser.add_argument("--score-threshold", type=float, default=DEFAULT_SCORE_THRESHOLD)
    parser.add_argument("--output-dir", type=Path, required=True, help="where metrics.json and metrics.md go")
    return parser.parse_args()


def main() -> None:
    """Load predictions and GT, score, write the report, log the headline."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    arguments = parse_command_line_arguments()
    predictions_header, predictions_by_scene_id = load_predictions_file(arguments.predictions)
    predictions_split_name = predictions_header.get("split_name")
    if predictions_split_name is not None and predictions_split_name != arguments.split:
        raise ValueError(f"predictions are for split {predictions_split_name!r}, asked to score {arguments.split!r}")
    scene_annotations = load_split_scene_annotations(arguments.split, limit=arguments.limit)
    logger.info("scoring %d scenes from %s", len(scene_annotations), arguments.predictions)
    metrics = score_predictions_against_split(
        predictions_by_scene_id,
        scene_annotations,
        score_threshold=arguments.score_threshold,
        detector_config=predictions_header.get("config"),
        detector_name=predictions_header.get("detector_name", arguments.predictions.stem),
    )
    metrics["run"]["predictions_file"] = str(arguments.predictions)
    metrics_json_path, metrics_markdown_path = write_metrics_report(metrics, arguments.output_dir)
    logger.info("wrote %s and %s", metrics_json_path, metrics_markdown_path)
    print(headline_summary_line(metrics))


if __name__ == "__main__":
    main()
