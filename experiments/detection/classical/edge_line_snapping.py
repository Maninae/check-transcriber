"""Snap each side of a quadrilateral to the strongest nearby intensity step.

Used twice: at working resolution right after fitting (mask-derived quads sit a few
pixels inside the true border, because opening, edge dilation and the texture window
all erode regions), and at full resolution for the final sub-pixel corners.

For each side, intensity profiles are sampled perpendicular to it (`cv.remap` on a grid
of sample points). The strongest step near the current position wins; a Gaussian prior
on the offset keeps it from jumping to a check's printed inner border. A parabola through
the peak gives sub-pixel offset, a Huber `fitLine` gives the side's line, and adjacent
lines intersect into corners. Sides lying on the image border are left alone (a check
running out of frame has no edge there).
"""

import cv2
import numpy as np

from experiments.detection.classical.quadrilateral_geometry import intersect_lines

SIDE_END_EXCLUSION_FRACTION = 0.1
MINIMUM_VALID_FRACTION = 0.35  # of a side's samples; otherwise the side keeps its old line
MAXIMUM_CORNER_SHIFT_RADIUS_FACTOR = 1.5  # corner may move at most this x the search radius
BORDER_MARGIN_PIXELS = 3.0


def side_is_on_image_border(side_start: np.ndarray, side_end: np.ndarray, image_width: int, image_height: int) -> bool:
    """Both endpoints near the same image border: the check leaves the frame here."""
    for coordinate_index, border_value in ((0, 0.0), (0, image_width - 1.0), (1, 0.0), (1, image_height - 1.0)):
        if (
            abs(side_start[coordinate_index] - border_value) < BORDER_MARGIN_PIXELS
            and abs(side_end[coordinate_index] - border_value) < BORDER_MARGIN_PIXELS
        ):
            return True
    return False


def fit_side_line_from_profiles(
    intensity: np.ndarray,
    side_start: np.ndarray,
    side_end: np.ndarray,
    search_radius_pixels: int,
    samples_per_side: int,
    minimum_step_strength: float,
) -> np.ndarray | None:
    """Line (vx, vy, x0, y0) through the strongest nearby step along one side; None if too weak."""
    side_vector = side_end - side_start
    side_length = float(np.linalg.norm(side_vector))
    if side_length < 4:
        return None
    unit_normal = np.array([-side_vector[1], side_vector[0]]) / side_length
    fractions = np.linspace(SIDE_END_EXCLUSION_FRACTION, 1 - SIDE_END_EXCLUSION_FRACTION, samples_per_side)
    offsets = np.arange(-search_radius_pixels - 1, search_radius_pixels + 2, dtype=np.float64)
    base_points = side_start + fractions[:, None] * side_vector
    sample_x = (base_points[:, 0:1] + offsets[None, :] * unit_normal[0]).astype(np.float32)
    sample_y = (base_points[:, 1:2] + offsets[None, :] * unit_normal[1]).astype(np.float32)
    profiles = cv2.remap(intensity, sample_x, sample_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    step_strength = np.abs(profiles[:, 2:] - profiles[:, :-2]) / 2.0  # centered difference at offsets[1:-1]
    step_offsets = offsets[1:-1]
    prior = np.exp(-0.5 * (step_offsets / (search_radius_pixels / 2.0)) ** 2)
    peak_index = np.argmax(step_strength * prior[None, :], axis=1)
    row_index = np.arange(len(peak_index))
    peak_strength = step_strength[row_index, peak_index]

    left_value = step_strength[row_index, np.clip(peak_index - 1, 0, len(step_offsets) - 1)]
    right_value = step_strength[row_index, np.clip(peak_index + 1, 0, len(step_offsets) - 1)]
    curvature = left_value - 2 * peak_strength + right_value
    safe_curvature = np.where(np.abs(curvature) > 1e-6, curvature, -1.0)
    subpixel_shift = np.where(np.abs(curvature) > 1e-6, 0.5 * (left_value - right_value) / safe_curvature, 0.0)
    edge_offsets = step_offsets[peak_index] + np.clip(subpixel_shift, -0.5, 0.5)

    valid = peak_strength > minimum_step_strength
    if valid.mean() < MINIMUM_VALID_FRACTION:
        return None
    edge_points = base_points[valid] + edge_offsets[valid, None] * unit_normal
    return cv2.fitLine(edge_points.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).reshape(4).astype(np.float64)


def snap_quadrilateral_sides_to_edges(
    intensity: np.ndarray,
    corners: np.ndarray,
    search_radius_pixels: int,
    samples_per_side: int,
    minimum_step_strength: float,
    full_image_size: tuple[int, int] | None = None,
    corner_offset: np.ndarray | None = None,
) -> np.ndarray:
    """Snapped (4, 2) corners in `intensity` coordinates; falls back side by side to the input.

    `full_image_size` (width, height) and `corner_offset` let a caller pass a crop: the
    image-border test then runs in full-image coordinates (corners + offset).
    """
    corner_offset = np.zeros(2) if corner_offset is None else corner_offset
    image_width, image_height = full_image_size or (intensity.shape[1], intensity.shape[0])
    side_lines = []
    for side_index in range(4):
        side_start, side_end = corners[side_index], corners[(side_index + 1) % 4]
        side_direction = (side_end - side_start) / max(float(np.linalg.norm(side_end - side_start)), 1e-9)
        original_line = np.concatenate([side_direction, side_start])
        if side_is_on_image_border(side_start + corner_offset, side_end + corner_offset, image_width, image_height):
            side_lines.append(original_line)
            continue
        fitted_line = fit_side_line_from_profiles(
            intensity, side_start, side_end, search_radius_pixels, samples_per_side, minimum_step_strength
        )
        side_lines.append(original_line if fitted_line is None else fitted_line)

    snapped_corners = []
    for corner_index in range(4):
        corner = intersect_lines(side_lines[corner_index - 1], side_lines[corner_index])
        if corner is None or np.linalg.norm(corner - corners[corner_index]) > MAXIMUM_CORNER_SHIFT_RADIUS_FACTOR * search_radius_pixels:
            corner = corners[corner_index]
        snapped_corners.append(corner)
    return np.array(snapped_corners)
