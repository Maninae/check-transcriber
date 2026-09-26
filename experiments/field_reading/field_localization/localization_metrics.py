"""Score localization predictions against ground truth: per-row records, then grouped summaries.

Per (check, target field) the GT is one of:
- scored box (status ok / too_small): IoU, hit (IoU >= 0.5), coverage (pred covers >= 95% of GT area),
  area ratio pred/GT (only when a box was predicted). A missing prediction scores IoU 0, no hit, no coverage.
- absent (no row for the field on the check): a predicted box is a false box.
- anything else (occluded, partially out of frame, not in frame): excluded and counted.
"""

import pandas as pd

from experiments.field_reading.config import TARGET_FIELD_NAMES
from experiments.field_reading.data_access.field_manifest import make_row_key
from experiments.field_reading.field_localization.box_geometry import box_area, box_coverage_of_target, box_iou
from experiments.field_reading.field_localization.localization_config import SCORED_STATUSES

HIT_IOU_THRESHOLD = 0.5
TEXT_COVERAGE_THRESHOLD = 0.95


def score_prediction_rows(check_records: list[dict], predictions_by_key: dict[str, dict]) -> pd.DataFrame:
    """One scored record per (check, target field); `gt_kind` is scored / absent / excluded."""
    scored_records = []
    for check_record in check_records:
        for field_name in TARGET_FIELD_NAMES:
            row_key = make_row_key(check_record["scene_id"], check_record["check_index"], field_name)
            prediction = predictions_by_key.get(row_key, {})
            predicted_box = prediction.get("pred_box")
            field_target = check_record["fields"].get(field_name)
            record = {"row_key": row_key, "field_name": field_name, "layout_family": check_record["layout_family"],
                      "template_id": check_record["template_id"], "has_prediction": predicted_box is not None,
                      "confidence": prediction.get("confidence")}
            if field_target is None:
                record.update(gt_kind="absent", status="absent", handwritten=None)
            elif field_target["status"] in SCORED_STATUSES and field_target["box"] is not None:
                target_box = field_target["box"]
                record.update(gt_kind="scored", status=field_target["status"], handwritten=field_target["handwritten"])
                if predicted_box is None:
                    record.update(iou=0.0, hit=False, covered=False, area_ratio=None)
                else:
                    iou = box_iou(predicted_box, target_box)
                    record.update(iou=iou, hit=iou >= HIT_IOU_THRESHOLD,
                                  covered=box_coverage_of_target(predicted_box, target_box) >= TEXT_COVERAGE_THRESHOLD,
                                  area_ratio=box_area(predicted_box) / box_area(target_box))
            else:
                record.update(gt_kind="excluded", status=field_target["status"], handwritten=field_target["handwritten"])
            scored_records.append(record)
    return pd.DataFrame.from_records(scored_records)


def summarize_scored_records(scored_records: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Aggregate metrics per group: mean IoU, hit rate, coverage rate, median area ratio, miss rate, false-box rate."""
    scored = scored_records[scored_records.gt_kind == "scored"]
    absent = scored_records[scored_records.gt_kind == "absent"]
    summary = scored.groupby(group_columns, dropna=False).agg(
        scored_count=("iou", "size"), mean_iou=("iou", "mean"), hit_rate=("hit", "mean"),
        coverage_rate=("covered", "mean"), median_area_ratio=("area_ratio", "median"),
        mean_area_ratio=("area_ratio", "mean"), no_box_rate=("has_prediction", lambda has: 1 - has.mean()))
    if group_columns == ["field_name"] or "status" not in group_columns and "handwritten" not in group_columns:
        false_boxes = absent.groupby(group_columns, dropna=False).agg(
            absent_count=("has_prediction", "size"), false_box_rate=("has_prediction", "mean"))
        summary = summary.join(false_boxes, how="outer")
    return summary.reset_index()
