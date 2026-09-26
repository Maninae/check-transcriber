"""Drawn test scenes for the classical detector: white 2.4:1 "checks" with printed lines.

Each check is a rotated rectangle of near-white paper carrying a few thin dark rules and
text-like dashes (the detector's interior gate requires print), drawn on a dark flat
or a textured background. Returned corners are the exact rectangle corners in pixels.
"""

import cv2
import numpy as np

CHECK_ASPECT_RATIO = 2.4
PAPER_GRAY_LEVEL = 235
PRINT_GRAY_LEVEL = 40


def rotated_rectangle_corners(center: tuple[float, float], width: float, angle_degrees: float) -> np.ndarray:
    """(4, 2) corners of a width x width/2.4 rectangle rotated about its center."""
    height = width / CHECK_ASPECT_RATIO
    half_extents = np.array([[-width, -height], [width, -height], [width, height], [-width, height]]) / 2.0
    angle = np.radians(angle_degrees)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return half_extents @ rotation.T + np.asarray(center, dtype=np.float64)


def make_background(height: int, width: int, textured: bool, seed: int = 0) -> np.ndarray:
    """Dark flat gray, or a colored woven-looking texture (stripes plus noise)."""
    if not textured:
        return np.full((height, width, 3), 60, dtype=np.uint8)
    random_generator = np.random.default_rng(seed)
    rows, columns = np.mgrid[0:height, 0:width]
    weave = 0.5 + 0.5 * np.sin(rows / 3.0) * np.sin(columns / 3.0)
    noise = random_generator.normal(0.0, 0.15, (height, width))
    base_color = np.array([40, 60, 150], dtype=np.float64)  # a reddish rug in BGR
    shading = np.clip(0.6 + 0.4 * weave + noise, 0.2, 1.2)
    return np.clip(base_color[None, None, :] * shading[..., None], 0, 255).astype(np.uint8)


def draw_check(image_bgr: np.ndarray, corners: np.ndarray) -> None:
    """Fill a check rectangle with paper and print a few rules and text dashes on it."""
    cv2.fillPoly(image_bgr, [np.rint(corners).astype(np.int32)], (PAPER_GRAY_LEVEL,) * 3, lineType=cv2.LINE_AA)
    top_left, top_right, _, bottom_left = corners
    along = top_right - top_left
    across = bottom_left - top_left
    for row_fraction in (0.18, 0.4, 0.62, 0.84):
        for start_fraction in np.arange(0.08, 0.86, 0.1):
            start_point = top_left + start_fraction * along + row_fraction * across
            end_point = start_point + 0.07 * along
            cv2.line(image_bgr, tuple(np.rint(start_point).astype(int)), tuple(np.rint(end_point).astype(int)),
                     (PRINT_GRAY_LEVEL,) * 3, 3, cv2.LINE_AA)


def build_scene(
    check_corner_list: list[np.ndarray], height: int = 1500, width: int = 2000, textured: bool = False
) -> np.ndarray:
    """A BGR scene with every check drawn in order (later checks on top)."""
    scene_bgr = make_background(height, width, textured)
    for corners in check_corner_list:
        draw_check(scene_bgr, corners)
    return cv2.GaussianBlur(scene_bgr, (0, 0), 0.8)
