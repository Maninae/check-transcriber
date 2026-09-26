"""Draw ground truth and one or more detectors' quads on a scene, for Owen to eyeball.

Reading order the image is designed for: the check outlines first (ground truth is a
thick green line, each detector a thinner line in its own colour on top), then the
small legend band at the top. Each detector's top-left corner is marked with a dot, so
orientation mistakes are visible (the dot should sit on the green TL dot).
"""

import cv2
import numpy as np

from experiments.detection.dataset.scene_annotations import SceneAnnotation
from experiments.detection.predictions.detected_check import DetectedCheck

GROUND_TRUTH_COLOUR_BGR = (60, 200, 60)
DETECTOR_COLOURS_BGR = [(0, 140, 255), (230, 60, 200), (255, 170, 0), (40, 40, 230)]
OVERLAY_LONG_SIDE_PIXELS = 2000
LEGEND_BAND_HEIGHT_PIXELS = 56
LEGEND_FONT = cv2.FONT_HERSHEY_DUPLEX
OVERLAY_JPEG_QUALITY = 85


def scaled_points(points: np.ndarray, scale: float) -> np.ndarray:
    """Float image points -> int32 polyline points at the overlay scale."""
    return np.round(points * scale).astype(np.int32).reshape(-1, 1, 2)


def draw_quad_with_top_left_marker(
    canvas: np.ndarray, corners: np.ndarray, scale: float, colour: tuple, thickness: int, marker_radius: int
) -> None:
    """Closed quad plus a filled dot on its first (top-left) corner."""
    cv2.polylines(canvas, [scaled_points(corners, scale)], True, colour, thickness, cv2.LINE_AA)
    top_left = tuple(int(v) for v in np.round(corners[0] * scale))
    cv2.circle(canvas, top_left, marker_radius, colour, -1, cv2.LINE_AA)


def draw_legend_band(canvas: np.ndarray, entries: list[tuple[str, tuple]], title: str) -> np.ndarray:
    """Prepend a white band naming each colour, with the scene title in gray after them."""
    band = np.full((LEGEND_BAND_HEIGHT_PIXELS, canvas.shape[1], 3), 255, np.uint8)
    cursor_x = 16
    for label, colour in entries:
        cv2.rectangle(band, (cursor_x, 18), (cursor_x + 28, 38), colour, -1)
        cv2.putText(band, label, (cursor_x + 38, 36), LEGEND_FONT, 0.75, (30, 30, 30), 1, cv2.LINE_AA)
        cursor_x += 38 + cv2.getTextSize(label, LEGEND_FONT, 0.75, 1)[0][0] + 36
    if title:
        cv2.putText(band, title, (cursor_x + 20, 36), LEGEND_FONT, 0.75, (130, 130, 130), 1, cv2.LINE_AA)
    return np.vstack([band, canvas])


def render_scene_overlay(
    scene: SceneAnnotation,
    detections_by_detector_name: dict[str, list[DetectedCheck]],
    title: str = "",
) -> np.ndarray:
    """Return the overlay image (BGR) for one scene."""
    image_bgr = cv2.imread(str(scene.image_path), cv2.IMREAD_COLOR)
    scale = OVERLAY_LONG_SIDE_PIXELS / max(image_bgr.shape[:2])
    canvas = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    for check in scene.checks:
        cv2.polylines(canvas, [scaled_points(check.outline, scale)], True, GROUND_TRUTH_COLOUR_BGR, 7, cv2.LINE_AA)
        cv2.circle(canvas, tuple(int(v) for v in np.round(check.corners[0] * scale)), 13, GROUND_TRUTH_COLOUR_BGR, -1)
    legend_entries = [("ground truth", GROUND_TRUTH_COLOUR_BGR)]
    for detector_number, (detector_name, detections) in enumerate(detections_by_detector_name.items()):
        colour = DETECTOR_COLOURS_BGR[detector_number % len(DETECTOR_COLOURS_BGR)]
        for detection in detections:
            draw_quad_with_top_left_marker(canvas, detection.corners, scale, colour, 3, 7)
        legend_entries.append((f"{detector_name} ({len(detections)})", colour))
    return draw_legend_band(canvas, legend_entries, title)


def save_overlay_jpeg(overlay_bgr: np.ndarray, output_path) -> None:
    """JPEG write; overlays at 2000 px long side and quality 85 land well under 3 MB."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay_bgr, [cv2.IMWRITE_JPEG_QUALITY, OVERLAY_JPEG_QUALITY])
