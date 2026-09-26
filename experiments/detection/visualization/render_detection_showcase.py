"""Render the showcase overlays (hard scenes) and per-detector failure galleries.

    python -m experiments.detection.visualization.render_detection_showcase --split eval \
        --detector classical=<predictions.json>:<metrics.json> \
        --detector yolo-obb+refine=<predictions.json>:<metrics.json> \
        --output-dir <dir>

Every overlay shows ground truth plus all detectors; each detector also gets one
failure gallery JPEG built from its own metrics records.
"""

import argparse
import json
import logging
from pathlib import Path

from experiments.detection.dataset.scene_annotations import load_split_scene_annotations
from experiments.detection.predictions.detected_check import load_predictions_file
from experiments.detection.visualization.failure_gallery import collect_failure_tiles, tile_gallery
from experiments.detection.visualization.hard_scene_selection import select_hard_scenes
from experiments.detection.visualization.scene_overlay import (
    DETECTOR_COLOURS_BGR,
    render_scene_overlay,
    save_overlay_jpeg,
)

logger = logging.getLogger(__name__)

GALLERY_WORST_MATCHED_COUNT = 8
GALLERY_MAX_MISSES_AND_FALSE_POSITIVES = 8


def parse_detector_argument(detector_argument: str) -> tuple[str, Path, Path]:
    """'name=predictions.json:metrics.json' -> (name, predictions path, metrics path)."""
    detector_name, paths = detector_argument.split("=", 1)
    predictions_path, metrics_path = paths.rsplit(":", 1)
    return detector_name, Path(predictions_path), Path(metrics_path)


def main() -> None:
    """Render overlays for the selected hard scenes and one gallery per detector."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", default="eval")
    parser.add_argument("--detector", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    scenes = load_split_scene_annotations(arguments.split)
    scenes_by_id = {scene.scene_id: scene for scene in scenes}
    detectors = [parse_detector_argument(argument) for argument in arguments.detector]
    predictions_by_detector = {name: load_predictions_file(path)[1] for name, path, _ in detectors}

    for overlay_number, (criterion_name, scene) in enumerate(select_hard_scenes(scenes), start=1):
        overlay = render_scene_overlay(
            scene,
            {name: predictions.get(scene.scene_id, []) for name, predictions in predictions_by_detector.items()},
            title=f"{scene.scene_id}: {criterion_name}, {scene.background_category}",
        )
        output_path = arguments.output_dir / f"overlay_{overlay_number:02d}__{criterion_name}__{scene.scene_id}.jpg"
        save_overlay_jpeg(overlay, output_path)
        logger.info("wrote %s (%.1f MB)", output_path, output_path.stat().st_size / 1e6)

    for detector_number, (detector_name, _, metrics_path) in enumerate(detectors):
        metrics = json.loads(metrics_path.read_text())
        tiles = collect_failure_tiles(
            metrics,
            scenes_by_id,
            predictions_by_detector[detector_name],
            DETECTOR_COLOURS_BGR[detector_number % len(DETECTOR_COLOURS_BGR)],
            GALLERY_WORST_MATCHED_COUNT,
            GALLERY_MAX_MISSES_AND_FALSE_POSITIVES,
        )
        output_path = arguments.output_dir / f"failures__{detector_name}.jpg"
        save_overlay_jpeg(tile_gallery(tiles), output_path)
        logger.info("wrote %s with %d tiles (%.1f MB)", output_path, len(tiles), output_path.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
