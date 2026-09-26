"""Headline aggregates over a split, computed from the per-scene and per-check records.

- Detection counts are pooled over the split (micro-averaged): precision = TP / kept
  predictions, recall = TP / GT checks.
- Localization stats (IoU, corner error, orientation) use only checks matched at the
  primary 0.5 threshold, so a detector is not rewarded for near-misses.
- Every ratio with an empty denominator is None, never a silent 0.
"""

import numpy as np

from experiments.detection.metrics.metric_records import (
    CORNER_ERROR_PIXEL_CUTOFFS,
    MATCHING_IOU_THRESHOLDS,
    PRIMARY_MATCHING_IOU_THRESHOLD,
    CheckScoreRecord,
    SceneScoreRecord,
    threshold_key,
)

PERCENTILES_REPORTED: tuple[int, ...] = (90, 95)


def safe_ratio(numerator: float, denominator: float) -> float | None:
    """numerator / denominator, or None when the denominator is zero."""
    return None if denominator == 0 else float(numerator) / float(denominator)


def f1_score(precision: float | None, recall: float | None) -> float | None:
    """Harmonic mean of precision and recall (0 when both are 0, None when either is undefined)."""
    if precision is None or recall is None:
        return None
    return 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)


def summarize_values(values: list[float]) -> dict:
    """count / mean / median / p90 / p95 of a list; statistics are None when it is empty."""
    summary = {"count": len(values), "mean": None, "median": None}
    summary.update({f"p{percentile}": None for percentile in PERCENTILES_REPORTED})
    if not values:
        return summary
    value_array = np.asarray(values, dtype=np.float64)
    summary["mean"] = float(value_array.mean())
    summary["median"] = float(np.median(value_array))
    for percentile in PERCENTILES_REPORTED:
        summary[f"p{percentile}"] = float(np.percentile(value_array, percentile))
    return summary


def is_primary_match(check_record: CheckScoreRecord) -> bool:
    """True when the check was matched at the primary (0.5) IoU threshold."""
    return threshold_key(PRIMARY_MATCHING_IOU_THRESHOLD) in check_record.matched_iou_threshold_keys


def detection_rates_by_threshold(
    scene_records: list[SceneScoreRecord], check_records: list[CheckScoreRecord]
) -> dict[str, dict]:
    """threshold_key -> {true_positives, predictions, ground_truth, precision, recall, f1}."""
    total_predictions = sum(scene_record.predicted_count for scene_record in scene_records)
    total_ground_truth = len(check_records)
    rates_by_threshold = {}
    for iou_threshold in MATCHING_IOU_THRESHOLDS:
        key = threshold_key(iou_threshold)
        true_positives = sum(scene_record.true_positives_by_threshold[key] for scene_record in scene_records)
        precision = safe_ratio(true_positives, total_predictions)
        recall = safe_ratio(true_positives, total_ground_truth)
        rates_by_threshold[key] = {
            "true_positives": true_positives,
            "predictions": total_predictions,
            "ground_truth": total_ground_truth,
            "precision": precision,
            "recall": recall,
            "f1": f1_score(precision, recall),
        }
    return rates_by_threshold


def localization_summary(check_records: list[CheckScoreRecord]) -> dict:
    """IoU, corner-error, and orientation statistics over checks matched at 0.5."""
    matched_records = [record for record in check_records if is_primary_match(record)]
    corner_records = [record for record in matched_records if record.corner_error_mean_px is not None]
    corner_errors_px = [record.corner_error_mean_px for record in corner_records]
    orientation_known_records = [record for record in corner_records if record.orientation_correct is not None]
    return {
        "matched_checks": len(matched_records),
        "iou_with_outline": summarize_values([record.iou_with_outline for record in matched_records]),
        "iou_with_corner_quad": summarize_values([record.iou_with_corner_quad for record in matched_records]),
        "corner_error_mean_px": summarize_values(corner_errors_px),
        "corner_error_max_px": summarize_values([record.corner_error_max_px for record in corner_records]),
        "corner_error_percent_short_side": summarize_values(
            [record.corner_error_mean_percent_short_side for record in corner_records]
        ),
        "fraction_corner_error_below_px": {
            f"{cutoff:g}": safe_ratio(sum(error < cutoff for error in corner_errors_px), len(corner_errors_px))
            for cutoff in CORNER_ERROR_PIXEL_CUTOFFS
        },
        "checks_with_partial_corners": sum(record.corners_used_for_error < 4 for record in corner_records),
        "checks_without_usable_corners": len(matched_records) - len(corner_records),
        "orientation_known_checks": len(orientation_known_records),
        "orientation_accuracy": safe_ratio(
            sum(record.orientation_correct for record in orientation_known_records), len(orientation_known_records)
        ),
        "axis_accuracy": safe_ratio(sum(record.axis_correct for record in corner_records), len(corner_records)),
        "counter_clockwise_predictions": sum(record.prediction_clockwise is False for record in matched_records),
    }


def scene_summary(scene_records: list[SceneScoreRecord]) -> dict:
    """Count accuracy, scene-perfect rate and coverage of the predictions file."""
    return {
        "scenes": len(scene_records),
        "count_accuracy": safe_ratio(sum(record.count_exact for record in scene_records), len(scene_records)),
        "mean_absolute_count_error": safe_ratio(
            sum(abs(record.predicted_count - record.ground_truth_count) for record in scene_records),
            len(scene_records),
        ),
        "scene_perfect_rate": safe_ratio(sum(record.scene_perfect for record in scene_records), len(scene_records)),
        "scenes_missing_from_predictions": sum(not record.had_predictions_entry for record in scene_records),
    }
