"""IoU matrix and greedy one-to-one matching between predicted quads and GT checks.

Greedy by descending IoU (COCO style): take the highest-IoU (prediction, GT) pair,
lock both, repeat, stopping below the threshold. Because every pair at or above a
higher threshold is decided before any lower pair is looked at, the matching at 0.9
is exactly the subset of the 0.5 matching with IoU >= 0.9, and so on down to the
0.1 diagnostic matching. We still match per threshold, which keeps each call
self-contained.
"""

import numpy as np
from shapely.geometry.base import BaseGeometry

from experiments.detection.metrics.quadrilateral_geometry import polygon_iou


def pairwise_iou_matrix(
    predicted_polygons: list[BaseGeometry], ground_truth_polygons: list[BaseGeometry]
) -> np.ndarray:
    """(num_predictions, num_ground_truth) IoU matrix of already-clipped polygons."""
    iou_matrix = np.zeros((len(predicted_polygons), len(ground_truth_polygons)), dtype=np.float64)
    for prediction_index, predicted_polygon in enumerate(predicted_polygons):
        if predicted_polygon.is_empty:
            continue
        for ground_truth_index, ground_truth_polygon in enumerate(ground_truth_polygons):
            # Cheap bounding-box reject before the exact polygon intersection.
            if not predicted_polygon.envelope.intersects(ground_truth_polygon.envelope):
                continue
            iou_matrix[prediction_index, ground_truth_index] = polygon_iou(
                predicted_polygon, ground_truth_polygon
            )
    return iou_matrix


def greedy_match_by_iou(iou_matrix: np.ndarray, iou_threshold: float) -> dict[int, int]:
    """One-to-one matching, highest IoU first; returns {ground_truth_index: prediction_index}.

    Pairs below `iou_threshold` are never matched. Ties break by (prediction, GT) index
    order so the result is deterministic.
    """
    if iou_matrix.size == 0:
        return {}
    candidate_prediction_indices, candidate_ground_truth_indices = np.nonzero(iou_matrix >= iou_threshold)
    candidate_ious = iou_matrix[candidate_prediction_indices, candidate_ground_truth_indices]
    # Stable sort on -IoU keeps the row-major (prediction, GT) order among equal IoUs.
    descending_order = np.argsort(-candidate_ious, kind="stable")
    matched_prediction_by_ground_truth: dict[int, int] = {}
    used_prediction_indices: set[int] = set()
    for candidate_position in descending_order:
        prediction_index = int(candidate_prediction_indices[candidate_position])
        ground_truth_index = int(candidate_ground_truth_indices[candidate_position])
        if prediction_index in used_prediction_indices or ground_truth_index in matched_prediction_by_ground_truth:
            continue
        matched_prediction_by_ground_truth[ground_truth_index] = prediction_index
        used_prediction_indices.add(prediction_index)
    return matched_prediction_by_ground_truth
