"""Per-scene scoring on hand-built scenes: the cases the detector will actually produce."""

import numpy as np
import pytest

from experiments.detection.metrics.scene_scoring import check_count_bucket_label, score_scene
from experiments.detection.predictions.detected_check import DetectedCheck
from experiments.detection.tests.metrics_test_scenes import (
    TEST_CHECK_HEIGHT,
    make_check,
    make_scene,
    rectangle_corners,
    two_check_scene,
)


def ground_truth_predictions(scene, orientation_known=True):
    """Every GT check echoed back as a prediction."""
    return [DetectedCheck(corners=check.corners.copy(), orientation_known=orientation_known) for check in scene.checks]


def test_perfect_prediction():
    """GT echoed back: IoU 1, corner error 0, orientation right, scene perfect."""
    scene = two_check_scene()
    scene_record, check_records = score_scene(scene, ground_truth_predictions(scene), score_threshold=0.0)
    assert scene_record.scene_perfect and scene_record.count_exact
    assert scene_record.true_positives_by_threshold == {"0.50": 2, "0.75": 2, "0.90": 2}
    assert scene_record.false_positives_by_threshold == {"0.50": 0, "0.75": 0, "0.90": 0}
    for record in check_records:
        assert record.iou_with_outline == pytest.approx(1.0)
        assert record.iou_with_corner_quad == pytest.approx(1.0)
        assert record.corner_error_mean_px == pytest.approx(0.0)
        assert record.best_cyclic_shift == 0 and record.orientation_correct and record.axis_correct
        assert record.corners_used_for_error == 4 and record.prediction_clockwise


def test_one_pixel_shift():
    """A 1 px shift: corner error exactly 1 px, percent of the 120 px short side, IoU just under 1."""
    scene = two_check_scene()
    predictions = [DetectedCheck(corners=check.corners + [1.0, 0.0], orientation_known=True) for check in scene.checks]
    _, check_records = score_scene(scene, predictions, score_threshold=0.0)
    record = check_records[0]
    assert record.corner_error_mean_px == pytest.approx(1.0)
    assert record.corner_error_max_px == pytest.approx(1.0)
    assert record.corner_error_mean_percent_short_side == pytest.approx(100.0 / TEST_CHECK_HEIGHT)
    assert record.iou_with_outline == pytest.approx(299.0 / 301.0)
    assert record.orientation_correct


def test_rotated_corner_order_is_orientation_and_axis_wrong_but_corners_right():
    """Starting at the check's TR instead of TL: localization perfect, orientation and axis wrong."""
    scene = two_check_scene()
    predictions = [DetectedCheck(corners=np.roll(check.corners, -1, axis=0), orientation_known=True) for check in scene.checks]
    scene_record, check_records = score_scene(scene, predictions, score_threshold=0.0)
    assert scene_record.scene_perfect
    record = check_records[0]
    assert record.corner_error_mean_px == pytest.approx(0.0)
    assert record.best_cyclic_shift != 0
    assert record.orientation_correct is False and record.axis_correct is False


def test_upside_down_order_keeps_axis_right():
    """Starting at the check's BR: orientation wrong, landscape axis right."""
    scene = two_check_scene()
    predictions = [DetectedCheck(corners=np.roll(check.corners, 2, axis=0), orientation_known=True) for check in scene.checks]
    _, check_records = score_scene(scene, predictions, score_threshold=0.0)
    assert check_records[0].best_cyclic_shift == 2
    assert check_records[0].orientation_correct is False and check_records[0].axis_correct is True


def test_orientation_unknown_scores_axis_only():
    """orientation_known=False leaves orientation_correct unset but still scores the axis."""
    scene = two_check_scene()
    _, check_records = score_scene(scene, ground_truth_predictions(scene, orientation_known=False), score_threshold=0.0)
    assert check_records[0].orientation_correct is None and check_records[0].axis_correct is True


def test_missed_check():
    """One of two checks missed: one TP, no FP, not perfect, count off by one."""
    scene = two_check_scene()
    scene_record, check_records = score_scene(scene, ground_truth_predictions(scene)[:1], score_threshold=0.0)
    assert scene_record.true_positives_by_threshold["0.50"] == 1
    assert scene_record.false_positives_by_threshold["0.50"] == 0
    assert not scene_record.scene_perfect and not scene_record.count_exact
    assert check_records[1].matched_iou_threshold_keys == [] and check_records[1].iou_with_outline is None


def test_duplicate_detection_is_one_false_positive():
    """The same check predicted twice: the better copy matches, the other is a FP."""
    scene = two_check_scene()
    predictions = ground_truth_predictions(scene) + [DetectedCheck(corners=scene.checks[0].corners + [5.0, 0.0])]
    scene_record, check_records = score_scene(scene, predictions, score_threshold=0.0)
    assert scene_record.true_positives_by_threshold["0.50"] == 2
    assert scene_record.false_positives_by_threshold["0.50"] == 1
    assert not scene_record.scene_perfect
    assert check_records[0].matched_prediction_index == 0


def test_partially_out_of_frame_check():
    """GT hangs off the left edge; a prediction snapped to the frame edge still scores IoU 1,
    and corner error uses only the two in-frame GT corners."""
    hanging_check = make_check(0, rectangle_corners(-100, 300))
    scene = make_scene("scene_hanging", [hanging_check])
    assert not hanging_check.fully_in_frame
    edge_snapped_prediction = np.array([[0, 300], [200, 300], [200, 420], [0, 420]], dtype=np.float64)
    _, check_records = score_scene(scene, [DetectedCheck(corners=edge_snapped_prediction, orientation_known=True)], 0.0)
    record = check_records[0]
    assert record.iou_with_outline == pytest.approx(1.0)
    assert record.iou_with_corner_quad == pytest.approx(1.0)
    assert record.corners_used_for_error == 2
    assert record.corner_error_mean_px == pytest.approx(0.0)
    assert record.orientation_correct


def test_degenerate_prediction_is_an_unmatched_false_positive():
    """A zero-area prediction never matches and counts against precision."""
    scene = make_scene("scene_single", [make_check(0, rectangle_corners(100, 100))])
    degenerate = DetectedCheck(corners=np.array([[150.0, 150.0]] * 4))
    scene_record, check_records = score_scene(scene, [degenerate], score_threshold=0.0)
    assert scene_record.false_positives_by_threshold["0.50"] == 1
    assert check_records[0].matched_prediction_index is None


def test_score_threshold_drops_predictions_everywhere():
    """A low-score duplicate below the threshold is neither a FP nor counted."""
    scene = two_check_scene()
    predictions = ground_truth_predictions(scene) + [DetectedCheck(corners=scene.checks[0].corners + [5.0, 0.0], score=0.2)]
    scene_record, _ = score_scene(scene, predictions, score_threshold=0.5)
    assert scene_record.predicted_count == 2 and scene_record.scene_perfect


def test_missing_scene_counts_as_no_predictions():
    """predictions=None: everything missed and the scene is flagged as absent."""
    scene_record, _ = score_scene(two_check_scene(), None, score_threshold=0.0)
    assert not scene_record.had_predictions_entry and scene_record.predicted_count == 0


def test_stricter_matchings_are_subsets():
    """A 10 px shift on a 120 px tall check drops below 0.9 IoU but stays above 0.75."""
    scene = two_check_scene()
    predictions = [DetectedCheck(corners=check.corners + [0.0, 10.0]) for check in scene.checks]
    scene_record, check_records = score_scene(scene, predictions, score_threshold=0.0)
    assert scene_record.true_positives_by_threshold == {"0.50": 2, "0.75": 2, "0.90": 0}
    assert check_records[0].matched_iou_threshold_keys == ["0.50", "0.75"]


@pytest.mark.parametrize("count, label", [(1, "1-3"), (3, "1-3"), (4, "4-6"), (9, "7-9"), (12, "10-12"), (13, "13+"), (0, "0")])
def test_check_count_buckets(count, label):
    """Bucket edges are inclusive."""
    assert check_count_bucket_label(count) == label
