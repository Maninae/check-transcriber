"""Draw ground truth (green) and classical predictions (red) on downscaled scene images.

    python -m experiments.detection.classical.render_debug_overlays \\
        --predictions <predictions.json> --split val --scene-ids val_000006 val_000051 \\
        --output-dir <dir>

GT outlines are drawn thin green; predicted quads thick red with their score. Output
JPEGs are 1400 px on the long side, meant to be read by eye.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from experiments.detection.dataset.scene_annotations import load_scene_annotation
from experiments.detection.config.paths import split_annotations_directory
from experiments.detection.predictions.detected_check import load_predictions_file

OVERLAY_LONG_SIDE_PIXELS = 1400
GROUND_TRUTH_COLOR_BGR = (40, 200, 40)
PREDICTION_COLOR_BGR = (40, 40, 230)


def render_scene_overlay(image_path: Path, ground_truth_outlines: list[np.ndarray], predictions: list) -> np.ndarray:
    """Downscaled image with GT outlines and predicted quads drawn on it."""
    image_bgr = cv2.imread(str(image_path))
    scale = OVERLAY_LONG_SIDE_PIXELS / max(image_bgr.shape[:2])
    overlay = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    for outline in ground_truth_outlines:
        cv2.polylines(overlay, [np.rint(outline * scale).astype(np.int32)], True, GROUND_TRUTH_COLOR_BGR, 2)
    for prediction in predictions:
        corners = np.rint(prediction.corners * scale).astype(np.int32)
        cv2.polylines(overlay, [corners], True, PREDICTION_COLOR_BGR, 3)
        cv2.circle(overlay, tuple(corners[0]), 6, PREDICTION_COLOR_BGR, -1)
        cv2.putText(overlay, f"{prediction.score:.2f}", tuple(corners.mean(axis=0).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, PREDICTION_COLOR_BGR, 2)
    return overlay


def main() -> None:
    """Render one overlay JPEG per requested scene."""
    parser = argparse.ArgumentParser(description="Render GT vs prediction overlays.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--scene-ids", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    _, predictions_by_scene_id = load_predictions_file(arguments.predictions)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    for scene_id in arguments.scene_ids:
        scene = load_scene_annotation(split_annotations_directory(arguments.split) / f"{scene_id}.json", arguments.split)
        overlay = render_scene_overlay(
            scene.image_path, [check.outline for check in scene.checks], predictions_by_scene_id.get(scene_id, [])
        )
        cv2.imwrite(str(arguments.output_dir / f"{scene_id}.jpg"), overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    main()
