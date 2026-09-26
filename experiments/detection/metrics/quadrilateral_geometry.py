"""Polygon IoU and corner-error geometry for check quadrilaterals.

- Every polygon is clipped to the image rectangle [0, W] x [0, H] before IoU: a check
  partly out of frame can only be detected where it is visible, and the GT outline
  may leave the frame.
- Invalid (self-intersecting, "bow-tie") predicted quads are repaired with
  `shapely.make_valid` rather than raising; a degenerate or non-finite quad has zero
  area and scores IoU 0.
- Corner error aligns the predicted corner order to the GT order by the best of the
  four cyclic shifts. Shift k pairs predicted corner (i + k) % 4 with GT corner i, so
  shift 0 means the predicted order starts at the check's own top-left and shift 2
  means the check is read upside down (landscape axis still right).
"""

import numpy as np
import shapely
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.geometry.base import BaseGeometry

CORNER_COUNT = 4
UPSIDE_DOWN_CYCLIC_SHIFT = 2


def polygonal_part(geometry: BaseGeometry) -> BaseGeometry:
    """Keep only the areal pieces of a geometry (make_valid can emit stray lines/points)."""
    if isinstance(geometry, (Polygon, MultiPolygon)):
        return geometry
    polygon_pieces = [
        part for part in shapely.get_parts(geometry) if isinstance(part, (Polygon, MultiPolygon))
    ]
    if not polygon_pieces:
        return Polygon()
    return shapely.union_all(polygon_pieces)


def clipped_polygon_in_frame(points: np.ndarray, image_width: int, image_height: int) -> BaseGeometry:
    """Polygon through `points` (N, 2), repaired if invalid, clipped to the image rectangle.

    Returns an empty polygon for fewer than 3 points or any non-finite coordinate.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 3 or not np.all(np.isfinite(points)):
        return Polygon()
    polygon = Polygon(points)
    if not polygon.is_valid:
        polygon = polygonal_part(shapely.make_valid(polygon))
    if polygon.is_empty:
        return polygon
    return polygonal_part(polygon.intersection(box(0.0, 0.0, float(image_width), float(image_height))))


def polygon_iou(first_polygon: BaseGeometry, second_polygon: BaseGeometry) -> float:
    """Intersection over union of two (already clipped) polygons; 0 when either is empty."""
    if first_polygon.is_empty or second_polygon.is_empty:
        return 0.0
    if first_polygon.area <= 0.0 or second_polygon.area <= 0.0:
        return 0.0
    intersection_area = first_polygon.intersection(second_polygon).area
    union_area = first_polygon.area + second_polygon.area - intersection_area
    if union_area <= 0.0:
        return 0.0
    return float(intersection_area / union_area)


def signed_area_image_coordinates(corners: np.ndarray) -> float:
    """Shoelace area with the image y-axis pointing down: positive means clockwise on screen."""
    x_coordinates, y_coordinates = corners[:, 0], corners[:, 1]
    return float(
        0.5 * np.sum(x_coordinates * np.roll(y_coordinates, -1) - np.roll(x_coordinates, -1) * y_coordinates)
    )


def check_short_side_length(ground_truth_corners: np.ndarray) -> float:
    """Shorter of the two mean opposite-edge lengths of a TL/TR/BR/BL quad (pixels)."""
    edge_lengths = np.linalg.norm(np.roll(ground_truth_corners, -1, axis=0) - ground_truth_corners, axis=1)
    top_bottom_mean = 0.5 * (edge_lengths[0] + edge_lengths[2])
    left_right_mean = 0.5 * (edge_lengths[1] + edge_lengths[3])
    return float(min(top_bottom_mean, left_right_mean))


def corner_inside_frame_mask(corners: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    """Boolean (4,) mask of corners lying inside the closed image rectangle."""
    return (
        (corners[:, 0] >= 0.0)
        & (corners[:, 0] <= image_width)
        & (corners[:, 1] >= 0.0)
        & (corners[:, 1] <= image_height)
    )


def best_cyclic_corner_alignment(
    predicted_corners: np.ndarray,
    ground_truth_corners: np.ndarray,
    ground_truth_corner_mask: np.ndarray,
) -> tuple[int, float, float] | None:
    """Best cyclic shift of the predicted order and its corner errors.

    Args:
        predicted_corners: (4, 2) predicted corners, clockwise.
        ground_truth_corners: (4, 2) GT corners, TL/TR/BR/BL of the check.
        ground_truth_corner_mask: (4,) bool, which GT corners count (in-frame ones).
    Returns:
        (best_shift, mean_error_px, max_error_px) over the masked corners, or None when
        no GT corner is usable. Ties go to the smallest shift.
    """
    if not np.any(ground_truth_corner_mask) or not np.all(np.isfinite(predicted_corners)):
        return None
    best_alignment = None
    for cyclic_shift in range(CORNER_COUNT):
        shifted_predicted_corners = np.roll(predicted_corners, -cyclic_shift, axis=0)
        corner_distances = np.linalg.norm(shifted_predicted_corners - ground_truth_corners, axis=1)
        used_distances = corner_distances[ground_truth_corner_mask]
        mean_error = float(used_distances.mean())
        if best_alignment is None or mean_error < best_alignment[1]:
            best_alignment = (cyclic_shift, mean_error, float(used_distances.max()))
    return best_alignment
