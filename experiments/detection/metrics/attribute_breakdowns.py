"""Recall / precision / IoU / corner error broken down by scene and check attributes.

- Scene attributes (background, layout, shadow, check-count bucket) group both the
  scenes and their checks, so precision is attributable: a false positive belongs to
  a scene.
- Check attributes (deformation, frame, overlap, orientation, size) group checks only,
  so their rows carry no precision.
- Deformation groups overlap on purpose: a check with fold + curl counts in both
  "has fold" and "has curl". "waves only" and "none" are exclusive.
"""

import re

import numpy as np

from experiments.detection.metrics.metric_records import (
    MATCHING_IOU_THRESHOLDS,
    PRIMARY_MATCHING_IOU_THRESHOLD,
    CheckScoreRecord,
    SceneScoreRecord,
    threshold_key,
)
from experiments.detection.metrics.split_aggregation import is_primary_match, safe_ratio

STRICT_RECALL_IOU_THRESHOLD = MATCHING_IOU_THRESHOLDS[-1]

SCENE_ATTRIBUTE_NAMES: tuple[str, ...] = (
    "background_category",
    "background_source",
    "layout_mode",
    "check_count_bucket",
    "cast_shadow_kind",
)


def natural_sort_key(group_value: str) -> tuple:
    """Sort "4-6" before "10-12" and "90" before "180": leading integer first, then text."""
    leading_integer_match = re.match(r"\d+", group_value)
    if leading_integer_match is None:
        return (1, 0, group_value)
    return (0, int(leading_integer_match.group()), group_value)


def deformation_groups(check_record: CheckScoreRecord) -> list[str]:
    """Overlapping deformation labels for one check (see module docstring)."""
    deformation_kinds = set(check_record.deformation_kinds)
    if not deformation_kinds:
        return ["none"]
    if deformation_kinds == {"waves"}:
        return ["waves only"]
    return [f"has {kind}" for kind in ("fold", "curl", "corner_lift") if kind in deformation_kinds]


CHECK_ATTRIBUTE_GROUPERS = {
    "deformation": deformation_groups,
    "fully_in_frame": lambda record: [str(record.fully_in_frame)],
    "overlapped": lambda record: [str(record.overlapped)],
    "orientation_class": lambda record: [str(record.orientation_class)],
    "size_kind": lambda record: [record.size_kind],
}


def check_group_row(check_records: list[CheckScoreRecord]) -> dict:
    """Recall at 0.5 / 0.9, mean IoU and median corner error for one group of checks."""
    primary_key = threshold_key(PRIMARY_MATCHING_IOU_THRESHOLD)
    strict_key = threshold_key(STRICT_RECALL_IOU_THRESHOLD)
    matched_records = [record for record in check_records if is_primary_match(record)]
    corner_errors_px = [
        record.corner_error_mean_px for record in matched_records if record.corner_error_mean_px is not None
    ]
    return {
        "checks": len(check_records),
        f"recall@{primary_key}": safe_ratio(len(matched_records), len(check_records)),
        f"recall@{strict_key}": safe_ratio(
            sum(strict_key in record.matched_iou_threshold_keys for record in check_records), len(check_records)
        ),
        "mean_iou_with_outline": (
            float(np.mean([record.iou_with_outline for record in matched_records])) if matched_records else None
        ),
        "median_corner_error_px": float(np.median(corner_errors_px)) if corner_errors_px else None,
    }


def breakdown_by_scene_attribute(
    attribute_name: str, scene_records: list[SceneScoreRecord], check_records: list[CheckScoreRecord]
) -> dict[str, dict]:
    """Group value -> row with scenes, precision@0.5 and the check-group stats."""
    primary_key = threshold_key(PRIMARY_MATCHING_IOU_THRESHOLD)
    scene_records_by_value: dict[str, list[SceneScoreRecord]] = {}
    for scene_record in scene_records:
        scene_records_by_value.setdefault(str(getattr(scene_record, attribute_name)), []).append(scene_record)
    value_by_scene_id = {record.scene_id: str(getattr(record, attribute_name)) for record in scene_records}
    rows = {}
    for group_value in sorted(scene_records_by_value, key=natural_sort_key):
        group_scene_records = scene_records_by_value[group_value]
        group_check_records = [
            record for record in check_records if value_by_scene_id[record.scene_id] == group_value
        ]
        row = {"scenes": len(group_scene_records)}
        row.update(check_group_row(group_check_records))
        row[f"precision@{primary_key}"] = safe_ratio(
            sum(record.true_positives_by_threshold[primary_key] for record in group_scene_records),
            sum(record.predicted_count for record in group_scene_records),
        )
        rows[group_value] = row
    return rows


def breakdown_by_check_attribute(attribute_name: str, check_records: list[CheckScoreRecord]) -> dict[str, dict]:
    """Group value -> check-group stats (no precision: false positives have no check)."""
    grouper = CHECK_ATTRIBUTE_GROUPERS[attribute_name]
    check_records_by_value: dict[str, list[CheckScoreRecord]] = {}
    for check_record in check_records:
        for group_value in grouper(check_record):
            check_records_by_value.setdefault(group_value, []).append(check_record)
    return {
        group_value: check_group_row(check_records_by_value[group_value])
        for group_value in sorted(check_records_by_value, key=natural_sort_key)
    }


def all_breakdowns(scene_records: list[SceneScoreRecord], check_records: list[CheckScoreRecord]) -> dict[str, dict]:
    """Every breakdown table, keyed by attribute name (scene attributes first)."""
    breakdowns = {
        attribute_name: breakdown_by_scene_attribute(attribute_name, scene_records, check_records)
        for attribute_name in SCENE_ATTRIBUTE_NAMES
    }
    for attribute_name in CHECK_ATTRIBUTE_GROUPERS:
        breakdowns[attribute_name] = breakdown_by_check_attribute(attribute_name, check_records)
    return breakdowns
