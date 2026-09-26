"""Draw ordered check quads (and optionally a heatmap) on an image for eyeballing.

Each quad is drawn as a closed polyline with a filled dot on its TL corner and a
smaller one on TR, so the corner ORDER is visible, not just the outline. Ground truth
is drawn green, predictions magenta.
"""

import cv2
import numpy as np

GROUND_TRUTH_COLOR_BGR = (60, 200, 60)
PREDICTION_COLOR_BGR = (220, 40, 220)
TOP_LEFT_DOT_RADIUS = 7
TOP_RIGHT_DOT_RADIUS = 4
HEATMAP_BLEND_WEIGHT = 0.45


def draw_ordered_quads(image_bgr: np.ndarray, corner_sets: list[np.ndarray], color_bgr: tuple, line_thickness: int = 2, labels: list[str] | None = None) -> np.ndarray:
    """Draw quads in place with TL (big dot) and TR (small dot) markers; returns the image."""
    for quad_index, corners in enumerate(corner_sets):
        integer_corners = np.round(corners).astype(np.int32)
        cv2.polylines(image_bgr, [integer_corners.reshape(-1, 1, 2)], True, color_bgr, line_thickness, cv2.LINE_AA)
        cv2.circle(image_bgr, tuple(int(value) for value in integer_corners[0]), TOP_LEFT_DOT_RADIUS, color_bgr, -1, cv2.LINE_AA)
        cv2.circle(image_bgr, tuple(int(value) for value in integer_corners[1]), TOP_RIGHT_DOT_RADIUS, color_bgr, -1, cv2.LINE_AA)
        if labels is not None:
            cv2.putText(image_bgr, labels[quad_index], tuple(int(value) for value in integer_corners.mean(axis=0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2, cv2.LINE_AA)
    return image_bgr


def blend_heatmap(image_bgr: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """Upsample a [0, 1] heatmap to the image size and alpha-blend it as a color map."""
    heatmap_resized = cv2.resize(heatmap.astype(np.float32), (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    heatmap_colored = cv2.applyColorMap(np.clip(heatmap_resized * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return cv2.addWeighted(image_bgr, 1 - HEATMAP_BLEND_WEIGHT, heatmap_colored, HEATMAP_BLEND_WEIGHT, 0)
