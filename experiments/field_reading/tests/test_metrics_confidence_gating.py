"""Gating curve on hand-built toys whose answers can be worked out on paper."""

import numpy as np
import pytest

from experiments.field_reading.metrics.confidence_gating import build_gating_operating_points, summarize_gating


def test_operating_points_on_ten_row_toy() -> None:
    """10 rows, confidences 1.0 .. 0.1, wrong only at 0.3 and 0.1; every row filled."""
    confidences = np.linspace(1.0, 0.1, 10)
    correct_flags = np.array([True] * 7 + [False, True, False])
    summary = summarize_gating(confidences, correct_flags, np.ones(10, bool))
    assert summary["fill_all"] == {"coverage": 1.0, "accuracy": 0.8}
    assert summary["accuracy_at_coverage"] == {"0.80": 7 / 8, "0.90": 8 / 9, "0.95": 0.8, "1.00": 0.8}
    # 95% and 98% and 99%: only the first 7 rows (all correct) qualify -> coverage 0.7 at threshold 0.4.
    for target in ("0.95", "0.98", "0.99"):
        assert summary["max_coverage_at_accuracy"][target]["coverage"] == pytest.approx(0.7)
        assert summary["max_coverage_at_accuracy"][target]["threshold"] == pytest.approx(0.4)


def test_tied_confidences_fill_together() -> None:
    """A threshold cannot split a tie: 4 rows at 0.9 (3 correct) form one block."""
    points = build_gating_operating_points(np.array([0.9, 0.9, 0.9, 0.9, 0.5]),
                                           np.array([True, True, False, True, True]), np.ones(5, bool))
    assert points.filled_count.tolist() == [4, 5]
    assert points.accuracy.tolist() == pytest.approx([0.75, 0.8])


def test_blank_rows_are_never_filled_so_full_coverage_is_unreachable() -> None:
    """2 of 4 rows blank: coverage tops out at 0.5, the 80%+ points are None, and blanks are not correct."""
    summary = summarize_gating(np.array([0.9, 0.8, 0.7, 0.6]), np.array([True, True, False, False]),
                               np.array([True, True, False, False]))
    assert summary["fill_all"] == {"coverage": 0.5, "accuracy": 1.0}
    assert all(value is None for value in summary["accuracy_at_coverage"].values())
    assert summary["max_coverage_at_accuracy"]["0.99"]["coverage"] == 0.5


def test_method_without_confidence_gets_only_the_fill_all_point() -> None:
    """All-null confidence: fill_all and the 100% point, no curve or targets."""
    summary = summarize_gating(np.full(4, np.nan), np.array([True, False, True, True]), np.ones(4, bool))
    assert summary["has_confidence"] is False
    assert summary["accuracy_at_coverage"] == {"1.00": 0.75}
    assert "max_coverage_at_accuracy" not in summary and "curve" not in summary


def test_unreachable_accuracy_target_reports_zero_coverage() -> None:
    """If even the most confident block is below target, coverage is 0 with no threshold."""
    summary = summarize_gating(np.array([0.9, 0.1]), np.array([False, True]), np.ones(2, bool))
    assert summary["max_coverage_at_accuracy"]["0.95"] == {"coverage": 0.0, "threshold": None, "accuracy": None}
