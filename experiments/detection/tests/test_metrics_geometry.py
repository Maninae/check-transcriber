"""Geometry and matching primitives: IoU with clipping/repair, cyclic alignment, greedy matching."""

import numpy as np
import pytest

from experiments.detection.metrics.detection_matching import greedy_match_by_iou
from experiments.detection.metrics.quadrilateral_geometry import (
    best_cyclic_corner_alignment,
    check_short_side_length,
    clipped_polygon_in_frame,
    polygon_iou,
    signed_area_image_coordinates,
)
from experiments.detection.tests.metrics_test_scenes import TEST_IMAGE_HEIGHT, TEST_IMAGE_WIDTH, rectangle_corners


def clip(points):
    """Clip to the test image."""
    return clipped_polygon_in_frame(np.asarray(points, dtype=np.float64), TEST_IMAGE_WIDTH, TEST_IMAGE_HEIGHT)


def test_identical_quads_have_iou_one_and_disjoint_zero():
    """IoU sanity at both ends."""
    corners = rectangle_corners(100, 100)
    assert polygon_iou(clip(corners), clip(corners)) == pytest.approx(1.0)
    assert polygon_iou(clip(corners), clip(rectangle_corners(600, 500))) == 0.0


def test_half_overlap_iou_is_one_third():
    """Shifting a rectangle by half its width overlaps 1/2 of each, IoU = 1/3."""
    assert polygon_iou(clip(rectangle_corners(100, 100)), clip(rectangle_corners(250, 100))) == pytest.approx(1 / 3)


def test_polygons_are_clipped_to_the_image():
    """A quad hanging off the left edge scores IoU 1 against its in-frame part."""
    hanging_quad = rectangle_corners(-150, 100)
    in_frame_part = np.array([[0, 100], [150, 100], [150, 220], [0, 220]], dtype=np.float64)
    assert clip(hanging_quad).area == pytest.approx(150 * 120)
    assert polygon_iou(clip(hanging_quad), clip(in_frame_part)) == pytest.approx(1.0)


def test_self_intersecting_quad_is_repaired_not_raised():
    """A bow-tie (two corners swapped) is repaired into its two triangles: half the area, IoU 0.5."""
    corners = rectangle_corners(100, 100)
    bow_tie = corners[[0, 1, 3, 2]]
    iou = polygon_iou(clip(bow_tie), clip(corners))
    assert iou == pytest.approx(0.5)


@pytest.mark.parametrize(
    "degenerate_corners",
    [
        [[200, 200]] * 4,
        [[100, 100], [200, 100], [300, 100], [400, 100]],
        [[100, 100], [np.nan, 100], [300, 200], [100, 200]],
        [[-500, -500], [-400, -500], [-400, -400], [-500, -400]],
    ],
    ids=["single_point", "collinear", "nan", "fully_out_of_frame"],
)
def test_degenerate_quads_score_iou_zero(degenerate_corners):
    """Zero-area, non-finite or off-image quads give IoU 0 instead of crashing."""
    assert polygon_iou(clip(degenerate_corners), clip(rectangle_corners(100, 100))) == 0.0


def test_cyclic_alignment_finds_shift_and_error():
    """Rolling the predicted order is undone by the best shift; errors are exact."""
    ground_truth = rectangle_corners(100, 100)
    all_corners = np.ones(4, dtype=bool)
    for roll_amount in range(4):
        rolled_prediction = np.roll(ground_truth + [1.0, 0.0], roll_amount, axis=0)
        best_shift, mean_error, max_error = best_cyclic_corner_alignment(rolled_prediction, ground_truth, all_corners)
        assert np.allclose(np.roll(rolled_prediction, -best_shift, axis=0), ground_truth + [1.0, 0.0])
        assert (best_shift == 0) == (roll_amount == 0)
        assert mean_error == pytest.approx(1.0) and max_error == pytest.approx(1.0)


def test_cyclic_alignment_ignores_masked_corners_and_none_when_empty():
    """Masked-out GT corners do not contribute; an all-false mask gives None."""
    ground_truth = rectangle_corners(100, 100)
    prediction = ground_truth.copy()
    prediction[0] += [50.0, 0.0]
    mask = np.array([False, True, True, True])
    assert best_cyclic_corner_alignment(prediction, ground_truth, mask)[1] == pytest.approx(0.0)
    assert best_cyclic_corner_alignment(prediction, ground_truth, np.zeros(4, dtype=bool)) is None


def test_short_side_and_winding():
    """Short side of a 300x120 rectangle is 120; TL/TR/BR/BL is clockwise on screen."""
    corners = rectangle_corners(100, 100)
    assert check_short_side_length(corners) == pytest.approx(120.0)
    assert signed_area_image_coordinates(corners) > 0
    assert signed_area_image_coordinates(corners[::-1]) < 0


def test_greedy_matching_is_one_to_one_and_highest_first():
    """The 0.9 pair wins; the prediction it used cannot match again; below threshold is dropped."""
    iou_matrix = np.array([[0.9, 0.6], [0.8, 0.3], [0.05, 0.2]])
    assert greedy_match_by_iou(iou_matrix, 0.5) == {0: 0}
    assert greedy_match_by_iou(iou_matrix, 0.1) == {0: 0, 1: 1}
    assert greedy_match_by_iou(np.zeros((0, 3)), 0.5) == {}
