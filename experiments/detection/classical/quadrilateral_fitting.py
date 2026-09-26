"""Fit a quadrilateral to a candidate region's contour.

Two steps:
1. Coarse quad: `approxPolyN(hull, 4)` finds the 4-gon enclosing the convex hull with the
   least added area. It never fails to return 4 corners (unlike `approxPolyDP`), and it
   tolerates rounded corners and small notches from text touching the border.
2. Side refit: each contour point is assigned to the nearest coarse side; points near the
   middle of a side get a robust `fitLine` (Huber), and adjacent lines are intersected.
   This recovers corners that the coarse quad placed outside a lifted or clipped corner.

Rectangularity (filled region area / quad area) is kept so selection can reject blobs
that are not one rectangle (two merged checks, a check plus a strip of background).
"""

from dataclasses import dataclass

import cv2
import numpy as np

from experiments.detection.classical.quadrilateral_geometry import (
    intersect_lines,
    order_corners_clockwise,
    quadrilateral_area,
    quadrilateral_aspect_ratio,
)

SIDE_POINT_DISTANCE_FRACTION = 0.03  # contour points farther than this x side length from a side are ignored
SIDE_CORNER_EXCLUSION_FRACTION = 0.12  # skip the ends of each side, where corners round off
MINIMUM_POINTS_PER_SIDE = 8
MAXIMUM_CORNER_SHIFT_FRACTION = 0.15  # refit corner may move at most this x the shorter side
COARSE_GATE_SLACK = 0.9  # coarse-quad pre-gates are this much looser than the final gates


@dataclass
class FittedQuadrilateral:
    """A quad fitted to one region, with the shape statistics used for filtering."""

    corners: np.ndarray  # (4, 2) float, clockwise, working pixels
    rectangularity: float  # filled region area / quad area, 1.0 for a perfect rectangle
    source_name: str


def distance_from_points_to_segment(points: np.ndarray, segment_start: np.ndarray, segment_end: np.ndarray) -> np.ndarray:
    """Euclidean distance from each point to a line segment."""
    segment_vector = segment_end - segment_start
    segment_length_squared = max(float(segment_vector @ segment_vector), 1e-9)
    projections = np.clip(((points - segment_start) @ segment_vector) / segment_length_squared, 0.0, 1.0)
    closest_points = segment_start + projections[:, None] * segment_vector
    return np.linalg.norm(points - closest_points, axis=1)


def refit_sides_with_lines(contour_points: np.ndarray, coarse_corners: np.ndarray) -> np.ndarray:
    """Refine a coarse quad by fitting a robust line to each side's contour points."""
    side_distances = np.stack(
        [
            distance_from_points_to_segment(contour_points, coarse_corners[side_index], coarse_corners[(side_index + 1) % 4])
            for side_index in range(4)
        ],
        axis=1,
    )
    nearest_side = np.argmin(side_distances, axis=1)
    fitted_lines = []
    for side_index in range(4):
        side_start, side_end = coarse_corners[side_index], coarse_corners[(side_index + 1) % 4]
        side_vector = side_end - side_start
        side_length = float(np.linalg.norm(side_vector))
        along_fraction = ((contour_points - side_start) @ side_vector) / max(side_length * side_length, 1e-9)
        keep = (
            (nearest_side == side_index)
            & (side_distances[:, side_index] < SIDE_POINT_DISTANCE_FRACTION * side_length + 2.0)
            & (along_fraction > SIDE_CORNER_EXCLUSION_FRACTION)
            & (along_fraction < 1.0 - SIDE_CORNER_EXCLUSION_FRACTION)
        )
        side_points = contour_points[keep].astype(np.float32)
        if len(side_points) < MINIMUM_POINTS_PER_SIDE:
            return coarse_corners
        fitted_lines.append(cv2.fitLine(side_points, cv2.DIST_HUBER, 0, 0.01, 0.01).reshape(4))

    refit_corners = []
    for corner_index in range(4):
        corner = intersect_lines(fitted_lines[corner_index - 1], fitted_lines[corner_index])
        if corner is None:
            return coarse_corners
        refit_corners.append(corner)
    refit_corners = np.array(refit_corners)
    shorter_side = min(np.linalg.norm(np.roll(coarse_corners, -1, axis=0) - coarse_corners, axis=1))
    if np.max(np.linalg.norm(refit_corners - coarse_corners, axis=1)) > MAXIMUM_CORNER_SHIFT_FRACTION * shorter_side:
        return coarse_corners
    return refit_corners


def fit_quadrilateral_to_contour(
    contour_points: np.ndarray,
    source_name: str,
    minimum_rectangularity: float = 0.0,
    aspect_range: tuple[float, float] = (0.0, np.inf),
) -> FittedQuadrilateral | None:
    """Coarse approxPolyN quad plus a per-side line refit; None for degenerate regions.

    `minimum_rectangularity` and `aspect_range` reject on the coarse quad before the
    (costly) line refit; pass loose values, the caller re-gates the refit quad.
    """
    hull = cv2.convexHull(contour_points.reshape(-1, 1, 2).astype(np.float32))
    if len(hull) < 4:
        return None
    coarse_corners = cv2.approxPolyN(hull, 4, epsilon_percentage=-1, ensure_convex=True).reshape(-1, 2)
    if len(coarse_corners) != 4:
        return None
    coarse_corners = order_corners_clockwise(coarse_corners.astype(np.float64))
    filled_region_area = abs(cv2.contourArea(contour_points.reshape(-1, 1, 2).astype(np.float32)))
    coarse_area = quadrilateral_area(coarse_corners)
    if coarse_area <= 0 or filled_region_area / coarse_area < minimum_rectangularity * COARSE_GATE_SLACK:
        return None
    if not aspect_range[0] * COARSE_GATE_SLACK <= quadrilateral_aspect_ratio(coarse_corners) <= aspect_range[1] / COARSE_GATE_SLACK:
        return None
    refined_corners = order_corners_clockwise(refit_sides_with_lines(contour_points.astype(np.float64), coarse_corners))
    quad_area = quadrilateral_area(refined_corners)
    if quad_area <= 0:
        return None
    return FittedQuadrilateral(refined_corners, min(filled_region_area / quad_area, 1.5), source_name)
