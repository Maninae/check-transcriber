"""Synthetic check photos with exactly known corners, for the corner-refinement tests.

A check is a filled quad (slight perspective) drawn 8x supersampled and area-downsampled,
so its edge sits at the true sub-pixel position in pixel-centre coordinates (the
convention of the dataset labels and of `cv2.remap`). Each check carries the distractors
that fool naive edge search: a printed border a few px inside the edge and dark
"text" bars. Backgrounds are blurred noise textures.
"""

import cv2
import numpy as np

SUPERSAMPLING_FACTOR = 8
DRAWING_SHIFT_BITS = 8
PERSPECTIVE_CHECK_CORNERS = np.array([[112.3, 131.7], [631.6, 104.2], [648.9, 377.4], [96.8, 356.1]])
SCENE_WIDTH, SCENE_HEIGHT = 760, 480
# Printed border like the val checks: ~2 px of ink starting ~8 px inside the paper edge.
BORDER_INSET_PIXELS = 8.0
BORDER_WIDTH_PIXELS = 2.0


def make_textured_background(width: int, height: int, mean_level: float, texture_amplitude: float, seed: int) -> np.ndarray:
    """Blurred-noise BGR texture around `mean_level` (0-255)."""
    random_generator = np.random.default_rng(seed)
    noise = random_generator.normal(0, 1, (height, width, 3)).astype(np.float32)
    texture = cv2.GaussianBlur(noise, (0, 0), 2.5)
    texture /= max(float(texture.std()), 1e-6)
    return np.clip(mean_level + texture_amplitude * texture, 0, 255)


def fill_quad_supersampled(canvas: np.ndarray, corners: np.ndarray, colour_bgr: tuple[float, float, float]) -> np.ndarray:
    """Composite a quad onto a float canvas with exact sub-pixel coverage."""
    height, width = canvas.shape[:2]
    scale = SUPERSAMPLING_FACTOR
    mask = np.zeros((height * scale, width * scale), np.uint8)
    # Low-res pixel-centre coordinate x maps to high-res (x + 0.5) * scale - 0.5.
    high_res_corners = (corners + 0.5) * scale - 0.5
    fixed_point = np.round(high_res_corners * (1 << DRAWING_SHIFT_BITS)).astype(np.int32)
    cv2.fillPoly(mask, [fixed_point.reshape(-1, 1, 2)], 255, cv2.LINE_8, DRAWING_SHIFT_BITS)
    coverage = cv2.resize(mask.astype(np.float32) / 255, (width, height), interpolation=cv2.INTER_AREA)[..., None]
    return canvas * (1 - coverage) + np.array(colour_bgr, np.float32) * coverage


def inset_quad_by_pixels(corners: np.ndarray, inset_pixels: float) -> np.ndarray:
    """Quad whose every side is moved `inset_pixels` inward (for printed borders)."""
    centroid = corners.mean(axis=0)
    moved_lines = []
    for side_index in range(4):
        start, end = corners[side_index], corners[(side_index + 1) % 4]
        direction = (end - start) / np.linalg.norm(end - start)
        inward_normal = np.array([-direction[1], direction[0]])
        if np.dot(inward_normal, centroid - start) < 0:
            inward_normal = -inward_normal
        moved_lines.append((start + inset_pixels * inward_normal, direction))
    inset_corners = []
    for corner_index in range(4):
        (first_point, first_direction), (second_point, second_direction) = moved_lines[corner_index - 1], moved_lines[corner_index]
        parameters = np.linalg.solve(np.array([first_direction, -second_direction]).T, second_point - first_point)
        inset_corners.append(first_point + parameters[0] * first_direction)
    return np.array(inset_corners)


def draw_check_with_distractors(canvas: np.ndarray, corners: np.ndarray, paper_bgr, ink_bgr) -> np.ndarray:
    """Paper quad plus an inset printed border and a few dark text bars."""
    canvas = fill_quad_supersampled(canvas, corners, paper_bgr)
    border_outer, border_inner = inset_quad_by_pixels(corners, BORDER_INSET_PIXELS), inset_quad_by_pixels(corners, BORDER_INSET_PIXELS + BORDER_WIDTH_PIXELS)
    canvas = fill_quad_supersampled(canvas, border_outer, ink_bgr)
    canvas = fill_quad_supersampled(canvas, border_inner, paper_bgr)
    top_left, top_right, bottom_right, bottom_left = corners
    for row_fraction in (0.3, 0.5, 0.7):
        left = top_left + (bottom_left - top_left) * row_fraction
        right = top_right + (bottom_right - top_right) * row_fraction
        start, end = left + (right - left) * 0.15, left + (right - left) * 0.7
        normal = np.array([0.0, 4.0])
        canvas = fill_quad_supersampled(canvas, np.array([start, end, end + normal, start + normal]), ink_bgr)
    return canvas


def render_scene(paper_bgr, background_level: float, texture_amplitude: float, seed: int = 0, corners: np.ndarray = PERSPECTIVE_CHECK_CORNERS, ink_bgr=(60, 60, 70)) -> np.ndarray:
    """One check on a textured background, as a uint8 BGR image."""
    canvas = make_textured_background(SCENE_WIDTH, SCENE_HEIGHT, background_level, texture_amplitude, seed)
    canvas = draw_check_with_distractors(canvas, corners, paper_bgr, ink_bgr)
    random_generator = np.random.default_rng(seed + 1)
    canvas += random_generator.normal(0, 2.0, canvas.shape).astype(np.float32)  # sensor noise
    return np.clip(np.round(canvas), 0, 255).astype(np.uint8)
