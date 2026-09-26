"""Pure geometry helpers for check quadrilaterals (ordering, side lengths, overlap).

Every function here takes plain (4, 2) float arrays and is free of image state, so
the fitting, verification and selection stages can share them. All operations map to
OpenCV.js calls (`contourArea`, `intersectConvexConvex`, `minAreaRect`) or trivial math.
"""

import cv2
import numpy as np


def order_corners_clockwise(corners: np.ndarray) -> np.ndarray:
    """Return the four corners clockwise in image coordinates (y down), top-most-left first.

    Sorting by angle around the centroid gives a consistent cyclic order; the starting
    corner is the one with the smallest x + y so that outputs are deterministic.
    """
    corners = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    centroid = corners.mean(axis=0)
    angles = np.arctan2(corners[:, 1] - centroid[1], corners[:, 0] - centroid[0])
    clockwise_corners = corners[np.argsort(angles)]  # increasing angle is clockwise when y points down
    start_index = int(np.argmin(clockwise_corners.sum(axis=1)))
    return np.roll(clockwise_corners, -start_index, axis=0)


def quadrilateral_side_lengths(corners: np.ndarray) -> np.ndarray:
    """Lengths of the four sides, side i running from corner i to corner i+1."""
    return np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)


def quadrilateral_aspect_ratio(corners: np.ndarray) -> float:
    """Long over short dimension, averaging each pair of opposite sides (>= 1)."""
    side_lengths = quadrilateral_side_lengths(corners)
    first_pair = (side_lengths[0] + side_lengths[2]) / 2.0
    second_pair = (side_lengths[1] + side_lengths[3]) / 2.0
    return float(max(first_pair, second_pair) / max(min(first_pair, second_pair), 1e-6))


def quadrilateral_area(corners: np.ndarray) -> float:
    """Unsigned polygon area."""
    return float(abs(cv2.contourArea(np.asarray(corners, dtype=np.float32))))


def quadrilateral_interior_angles_degrees(corners: np.ndarray) -> np.ndarray:
    """Interior angle at each corner, in degrees."""
    previous_vectors = np.roll(corners, 1, axis=0) - corners
    next_vectors = np.roll(corners, -1, axis=0) - corners
    cosines = (previous_vectors * next_vectors).sum(axis=1) / (
        np.linalg.norm(previous_vectors, axis=1) * np.linalg.norm(next_vectors, axis=1) + 1e-9
    )
    return np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))


def is_convex_quadrilateral(corners: np.ndarray) -> bool:
    """True when the four corners form a convex, non-self-intersecting polygon."""
    return bool(cv2.isContourConvex(np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)))


def convex_polygon_intersection_area(first_corners: np.ndarray, second_corners: np.ndarray) -> float:
    """Area shared by two convex polygons (cv.intersectConvexConvex)."""
    intersection_area, _ = cv2.intersectConvexConvex(
        np.asarray(first_corners, dtype=np.float32), np.asarray(second_corners, dtype=np.float32)
    )
    return float(max(intersection_area, 0.0))


def convex_quadrilateral_iou(first_corners: np.ndarray, second_corners: np.ndarray) -> float:
    """Intersection over union of two convex quadrilaterals."""
    intersection_area = convex_polygon_intersection_area(first_corners, second_corners)
    union_area = quadrilateral_area(first_corners) + quadrilateral_area(second_corners) - intersection_area
    return intersection_area / union_area if union_area > 0 else 0.0


def intersect_lines(first_line: np.ndarray, second_line: np.ndarray) -> np.ndarray | None:
    """Intersection of two lines given as (vx, vy, x0, y0); None when near-parallel."""
    first_direction, first_point = first_line[:2], first_line[2:]
    second_direction, second_point = second_line[:2], second_line[2:]
    determinant = first_direction[0] * (-second_direction[1]) - first_direction[1] * (-second_direction[0])
    if abs(determinant) < 1e-9:
        return None
    offset = second_point - first_point
    first_parameter = (offset[0] * (-second_direction[1]) - offset[1] * (-second_direction[0])) / determinant
    return first_point + first_parameter * first_direction


def min_area_rect_corners(points: np.ndarray) -> np.ndarray:
    """Corners of the minimum-area rotated rectangle around a point set, clockwise."""
    rotated_rect = cv2.minAreaRect(np.asarray(points, dtype=np.float32).reshape(-1, 1, 2))
    return order_corners_clockwise(cv2.boxPoints(rotated_rect).astype(np.float64))
