"""Coverage-vs-accuracy curves for confidence gating (the app fills a field only above a threshold).

For one method on one field's rows:
- coverage(t) = fraction of ALL scored rows filled at threshold t (non-blank and confidence >= t)
- accuracy(t) = fraction correct among the filled rows
Thresholds step over distinct confidence values, so tied confidences are filled or blanked
together (no threshold can split a tie). Blank predictions are never filled at any threshold, so
100% coverage is unreachable for a method that blanks rows; unreachable points are None.
A method with no confidence at all only gets its fill-everything point.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REPORTED_COVERAGE_LEVELS = [0.80, 0.90, 0.95, 1.00]
REPORTED_ACCURACY_TARGETS = [0.95, 0.98, 0.99]
CURVE_POINT_COUNT = 21


def build_gating_operating_points(confidences: np.ndarray, correct_flags: np.ndarray, covered_flags: np.ndarray) -> pd.DataFrame:
    """One row per distinct threshold (descending): threshold, filled_count, coverage, accuracy.

    Args:
        confidences: (N,) float, NaN allowed (NaN rows are filled only by the lowest threshold).
        correct_flags: (N,) bool, the row is read correctly.
        covered_flags: (N,) bool, the prediction is non-blank.
    """
    total_row_count = len(confidences)
    fillable_mask = covered_flags.astype(bool)
    fillable_confidences = np.where(np.isnan(confidences[fillable_mask]), -np.inf, confidences[fillable_mask])
    fillable_correct = correct_flags[fillable_mask].astype(bool)
    if total_row_count == 0 or len(fillable_confidences) == 0:
        return pd.DataFrame(columns=["threshold", "filled_count", "coverage", "accuracy"])
    order = np.argsort(-fillable_confidences, kind="stable")
    sorted_confidences = fillable_confidences[order]
    cumulative_correct = np.cumsum(fillable_correct[order])
    # The last index of each run of tied confidences is where a threshold can stop.
    block_end_indices = np.flatnonzero(np.append(sorted_confidences[1:] != sorted_confidences[:-1], True))
    filled_counts = block_end_indices + 1
    return pd.DataFrame({
        "threshold": sorted_confidences[block_end_indices],
        "filled_count": filled_counts,
        "coverage": filled_counts / total_row_count,
        "accuracy": cumulative_correct[block_end_indices] / filled_counts,
    })


def as_json_threshold(threshold_value: float) -> float | None:
    """-inf (the NaN-confidence block) becomes None in reports."""
    return None if not np.isfinite(threshold_value) else float(threshold_value)


def summarize_gating(confidences: np.ndarray, correct_flags: np.ndarray, covered_flags: np.ndarray) -> dict[str, object]:
    """Gating summary for one method x field: fill-all point, accuracy at coverage levels, max coverage at accuracy targets."""
    total_row_count = len(confidences)
    covered_count = int(covered_flags.sum())
    fill_all_point = {
        "coverage": covered_count / total_row_count if total_row_count else None,
        "accuracy": float(correct_flags[covered_flags.astype(bool)].mean()) if covered_count else None,
    }
    has_confidence = bool(np.isfinite(confidences[covered_flags.astype(bool)]).any()) if covered_count else False
    summary: dict[str, object] = {"n": total_row_count, "has_confidence": has_confidence, "fill_all": fill_all_point}
    full_coverage_accuracy = fill_all_point["accuracy"] if covered_count == total_row_count else None
    if not has_confidence:
        summary["accuracy_at_coverage"] = {"1.00": full_coverage_accuracy}
        return summary
    if np.isnan(confidences[covered_flags.astype(bool)]).any():
        logger.warning("some filled rows have null confidence; they rank below every scored row")
    operating_points = build_gating_operating_points(confidences, correct_flags, covered_flags)
    accuracy_at_coverage: dict[str, float | None] = {}
    for coverage_level in REPORTED_COVERAGE_LEVELS:
        reachable_points = operating_points[operating_points.coverage >= coverage_level - 1e-12]
        accuracy_at_coverage[f"{coverage_level:.2f}"] = None if reachable_points.empty else float(reachable_points.accuracy.iloc[0])
    max_coverage_at_accuracy: dict[str, dict[str, float | None]] = {}
    for accuracy_target in REPORTED_ACCURACY_TARGETS:
        passing_points = operating_points[operating_points.accuracy >= accuracy_target - 1e-12]
        if passing_points.empty:
            max_coverage_at_accuracy[f"{accuracy_target:.2f}"] = {"coverage": 0.0, "threshold": None, "accuracy": None}
            continue
        best_point = passing_points.loc[passing_points.coverage.idxmax()]
        max_coverage_at_accuracy[f"{accuracy_target:.2f}"] = {
            "coverage": float(best_point.coverage),
            "threshold": as_json_threshold(best_point.threshold),
            "accuracy": float(best_point.accuracy),
        }
    curve_indices = np.unique(np.linspace(0, len(operating_points) - 1, CURVE_POINT_COUNT).round().astype(int))
    summary["accuracy_at_coverage"] = accuracy_at_coverage
    summary["max_coverage_at_accuracy"] = max_coverage_at_accuracy
    summary["curve"] = [
        {"threshold": as_json_threshold(point.threshold), "coverage": float(point.coverage), "accuracy": float(point.accuracy)}
        for point in operating_points.iloc[curve_indices].itertuples()
    ]
    return summary
