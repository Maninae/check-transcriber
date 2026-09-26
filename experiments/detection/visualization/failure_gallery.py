"""Tile the worst checks of one detector into a single gallery JPEG.

A failure is a missed check (no prediction at IoU >= 0.5), a false positive, or a
matched check among the largest corner errors. Each tile is a crop around the check
with ground truth (green) and the prediction (detector colour), captioned with the
scene id, the failure kind and the error, so a reader can go straight to the cause.
"""

import cv2
import numpy as np

from experiments.detection.dataset.scene_annotations import SceneAnnotation
from experiments.detection.predictions.detected_check import DetectedCheck
from experiments.detection.visualization.scene_overlay import GROUND_TRUTH_COLOUR_BGR, draw_quad_with_top_left_marker

TILE_SIZE_PIXELS = 420
TILE_CAPTION_HEIGHT_PIXELS = 34
GALLERY_COLUMNS = 4
CROP_PADDING_FRACTION = 0.25


def crop_tile(
    scene: SceneAnnotation,
    image_bgr: np.ndarray,
    focus_points: np.ndarray,
    ground_truth_quads: list[np.ndarray],
    predicted_quads: list[np.ndarray],
    detector_colour: tuple,
    caption: str,
) -> np.ndarray:
    """Square crop around `focus_points` with quads drawn, plus a caption strip."""
    x_min, y_min = focus_points.min(axis=0)
    x_max, y_max = focus_points.max(axis=0)
    half_side = max(x_max - x_min, y_max - y_min) * (0.5 + CROP_PADDING_FRACTION)
    center_x, center_y = (x_min + x_max) / 2, (y_min + y_max) / 2
    origin = np.array([center_x - half_side, center_y - half_side])
    scale = TILE_SIZE_PIXELS / (2 * half_side)
    affine = np.array([[scale, 0, -origin[0] * scale], [0, scale, -origin[1] * scale]], np.float32)
    tile = cv2.warpAffine(image_bgr, affine, (TILE_SIZE_PIXELS, TILE_SIZE_PIXELS), flags=cv2.INTER_AREA, borderValue=(40, 40, 40))
    for quad in ground_truth_quads:
        cv2.polylines(tile, [np.round((quad - origin) * scale).astype(np.int32)], True, GROUND_TRUTH_COLOUR_BGR, 3, cv2.LINE_AA)
    for quad in predicted_quads:
        draw_quad_with_top_left_marker(tile, quad - origin, scale, detector_colour, 2, 5)
    caption_strip = np.full((TILE_CAPTION_HEIGHT_PIXELS, TILE_SIZE_PIXELS, 3), 255, np.uint8)
    cv2.putText(caption_strip, caption, (8, 23), cv2.FONT_HERSHEY_DUPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
    return np.vstack([caption_strip, tile])


def tile_gallery(tiles: list[np.ndarray]) -> np.ndarray:
    """Arrange tiles in rows of GALLERY_COLUMNS, padding the last row."""
    blank_tile = np.full_like(tiles[0], 255)
    padded_tiles = tiles + [blank_tile] * (-len(tiles) % GALLERY_COLUMNS)
    rows = [np.hstack(padded_tiles[start : start + GALLERY_COLUMNS]) for start in range(0, len(padded_tiles), GALLERY_COLUMNS)]
    return np.vstack(rows)


def collect_failure_tiles(
    metrics: dict,
    scenes_by_id: dict[str, SceneAnnotation],
    predictions_by_scene_id: dict[str, list[DetectedCheck]],
    detector_colour: tuple,
    worst_matched_count: int,
    max_misses_and_false_positives: int,
) -> list[np.ndarray]:
    """Build tiles for misses, false positives and the worst matched corner errors."""
    per_check_records = metrics["per_check_records"]
    missed_records = [record for record in per_check_records if record.get("matched_prediction_index") is None]
    matched_records = sorted(
        (record for record in per_check_records if record.get("corner_error_mean_px") is not None),
        key=lambda record: -record["corner_error_mean_px"],
    )
    false_positive_scene_ids = [
        record["scene_id"] for record in metrics["per_scene_records"] if record["false_positives_by_threshold"]["0.50"] > 0
    ]
    tiles, image_cache = [], {}

    def load_image(scene_id: str) -> np.ndarray:
        if scene_id not in image_cache:
            image_cache.clear()
            image_cache[scene_id] = cv2.imread(str(scenes_by_id[scene_id].image_path), cv2.IMREAD_COLOR)
        return image_cache[scene_id]

    for record in missed_records[:max_misses_and_false_positives]:
        scene = scenes_by_id[record["scene_id"]]
        ground_truth_corners = scene.checks[record["check_index"]].corners
        tiles.append(
            crop_tile(scene, load_image(scene.scene_id), ground_truth_corners, [ground_truth_corners],
                      [d.corners for d in predictions_by_scene_id.get(scene.scene_id, [])], detector_colour,
                      f"{scene.scene_id} #{record['check_index']} MISSED ({scene.background_category})")
        )
    matched_prediction_indices_by_scene: dict[str, set[int]] = {}
    for record in per_check_records:
        if record.get("matched_prediction_index") is not None:
            matched_prediction_indices_by_scene.setdefault(record["scene_id"], set()).add(record["matched_prediction_index"])
    false_positive_count = 0
    for scene_id in false_positive_scene_ids:
        scene = scenes_by_id[scene_id]
        for prediction_index, detection in enumerate(predictions_by_scene_id.get(scene_id, [])):
            if prediction_index in matched_prediction_indices_by_scene.get(scene_id, set()) or false_positive_count >= max_misses_and_false_positives:
                continue
            false_positive_count += 1
            tiles.append(
                crop_tile(scene, load_image(scene_id), detection.corners, [c.corners for c in scene.checks],
                          [detection.corners], detector_colour,
                          f"{scene_id} FALSE POSITIVE score {detection.score:.2f}")
            )
    for record in matched_records[:worst_matched_count]:
        scene = scenes_by_id[record["scene_id"]]
        detection = predictions_by_scene_id[scene.scene_id][record["matched_prediction_index"]]
        ground_truth_corners = scene.checks[record["check_index"]].corners
        tiles.append(
            crop_tile(scene, load_image(scene.scene_id), np.vstack([ground_truth_corners, detection.corners]),
                      [ground_truth_corners], [detection.corners], detector_colour,
                      f"{scene.scene_id} #{record['check_index']} corner err {record['corner_error_mean_px']:.0f} px")
        )
    return tiles
