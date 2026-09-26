"""Split-level aggregation, breakdowns and the written report, end to end in-process."""

import json

import numpy as np
import pytest

from experiments.detection.metrics.metrics_report_writing import headline_summary_line, write_metrics_report
from experiments.detection.metrics.score_predictions import score_predictions_against_split
from experiments.detection.predictions.detected_check import DetectedCheck
from experiments.detection.tests.metrics_test_scenes import make_check, make_scene, rectangle_corners, two_check_scene


@pytest.fixture
def scored_metrics():
    """Scene A: perfect, one upside-down, one duplicate FP. Scene B: its only (folded) check missed."""
    scene_a = two_check_scene("scene_a")
    scene_b = make_scene("scene_b", [make_check(0, rectangle_corners(300, 300), ("fold", "waves"))], layout_mode="loose_fan")
    predictions_by_scene_id = {
        "scene_a": [
            DetectedCheck(corners=scene_a.checks[0].corners.copy(), orientation_known=True),
            DetectedCheck(corners=np.roll(scene_a.checks[1].corners, 2, axis=0), orientation_known=True),
            DetectedCheck(corners=scene_a.checks[0].corners + [4.0, 0.0], orientation_known=True),
        ],
        "extra_scene_not_in_split": [],
    }
    return score_predictions_against_split(
        predictions_by_scene_id, [scene_a, scene_b], detector_config={"seconds_per_image": 0.05}, detector_name="unit"
    )


def test_headline_aggregates(scored_metrics):
    """2 TP of 3 predictions and 3 GT; orientation 1/2, axis 2/2; one scene missing."""
    rates = scored_metrics["detection"]["0.50"]
    assert rates["precision"] == pytest.approx(2 / 3) and rates["recall"] == pytest.approx(2 / 3)
    assert rates["f1"] == pytest.approx(2 / 3)
    localization = scored_metrics["localization"]
    assert localization["matched_checks"] == 2
    assert localization["orientation_accuracy"] == pytest.approx(0.5)
    assert localization["axis_accuracy"] == pytest.approx(1.0)
    assert localization["corner_error_mean_px"]["median"] == pytest.approx(0.0)
    assert localization["fraction_corner_error_below_px"]["2"] == pytest.approx(1.0)
    assert scored_metrics["scenes"]["scenes_missing_from_predictions"] == 1
    assert scored_metrics["scenes"]["scene_perfect_rate"] == pytest.approx(0.0)
    assert scored_metrics["run"]["prediction_scenes_not_scored"] == 1
    assert scored_metrics["latency"]["seconds_per_image"] == pytest.approx(0.05)


def test_breakdowns(scored_metrics):
    """Scene attributes carry precision; overlapping deformation groups; natural bucket order."""
    layout_rows = scored_metrics["breakdowns"]["layout_mode"]
    assert layout_rows["grid"]["precision@0.50"] == pytest.approx(2 / 3)
    assert layout_rows["loose_fan"]["recall@0.50"] == 0.0 and layout_rows["loose_fan"]["precision@0.50"] is None
    deformation_rows = scored_metrics["breakdowns"]["deformation"]
    assert deformation_rows["has fold"]["checks"] == 1 and deformation_rows["none"]["checks"] == 2
    assert "precision@0.50" not in deformation_rows["none"]
    assert list(scored_metrics["breakdowns"]["check_count_bucket"]) == ["1-3"]


def test_report_files_and_headline_line(scored_metrics, tmp_path):
    """metrics.json round-trips with record lists; metrics.md has the headline and breakdown tables."""
    metrics_json_path, metrics_markdown_path = write_metrics_report(scored_metrics, tmp_path)
    reloaded = json.loads(metrics_json_path.read_text())
    assert len(reloaded["per_check_records"]) == 3 and len(reloaded["per_scene_records"]) == 2
    markdown = metrics_markdown_path.read_text()
    assert "## Detection" in markdown and "### deformation" in markdown and "50.0 ms/image" in markdown
    assert headline_summary_line(scored_metrics).startswith("F1@0.50 66.7%")


def test_empty_split_fails_loud():
    """Scoring zero scenes is a caller bug, not an all-None report."""
    with pytest.raises(ValueError):
        score_predictions_against_split({}, [])
