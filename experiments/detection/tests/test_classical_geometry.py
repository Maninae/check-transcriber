"""Unit tests for the classical detector's pure geometry and line helpers."""

import numpy as np

from experiments.detection.classical.geometry.edge_line_snapping import snap_quadrilateral_sides_to_edges
from experiments.detection.classical.candidates.line_segment_extraction import merge_collinear_segments
from experiments.detection.classical.geometry.quadrilateral_geometry import (
    convex_quadrilateral_iou,
    intersect_lines,
    order_corners_clockwise,
    quadrilateral_aspect_ratio,
)


def test_order_corners_clockwise_is_cyclic_and_deterministic():
    square = np.array([[0, 0], [10, 0], [10, 5], [0, 5]], dtype=float)
    shuffled = square[[2, 0, 3, 1]]
    ordered = order_corners_clockwise(shuffled)
    np.testing.assert_allclose(ordered, square)


def test_aspect_ratio_and_iou():
    rectangle = np.array([[0, 0], [24, 0], [24, 10], [0, 10]], dtype=float)
    assert abs(quadrilateral_aspect_ratio(rectangle) - 2.4) < 1e-9
    assert abs(convex_quadrilateral_iou(rectangle, rectangle) - 1.0) < 1e-6
    shifted = rectangle + [12, 0]
    assert abs(convex_quadrilateral_iou(rectangle, shifted) - 1 / 3) < 1e-6


def test_intersect_lines():
    horizontal = np.array([1.0, 0.0, 0.0, 5.0])
    vertical = np.array([0.0, 1.0, 3.0, 0.0])
    np.testing.assert_allclose(intersect_lines(horizontal, vertical), [3.0, 5.0])
    assert intersect_lines(horizontal, horizontal) is None


def test_merge_collinear_segments_fuses_gapped_pieces():
    pieces = np.array([[0, 0, 100, 0], [110, 0.5, 200, 0.5], [0, 50, 100, 50]], dtype=float)
    merged = merge_collinear_segments(pieces, maximum_gap_pixels=20)
    assert len(merged) == 2
    lengths = sorted(np.hypot(merged[:, 2] - merged[:, 0], merged[:, 3] - merged[:, 1]))
    assert lengths[-1] > 199


def test_snap_moves_an_inset_quad_onto_the_step_edge():
    intensity = np.zeros((200, 300), dtype=np.float32)
    intensity[50:150, 60:240] = 200.0  # bright rectangle, edges at x=59.5/239.5, y=49.5/149.5
    inset_quad = np.array([[64, 54], [235, 54], [235, 145], [64, 145]], dtype=float)
    snapped = snap_quadrilateral_sides_to_edges(intensity, inset_quad, 8, 30, 5.0)
    expected = np.array([[59.5, 49.5], [239.5, 49.5], [239.5, 149.5], [59.5, 149.5]])
    np.testing.assert_allclose(snapped, expected, atol=0.6)
