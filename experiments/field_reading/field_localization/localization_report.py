"""Score every localization method on a split and write a metrics JSON plus a markdown report.

Outputs under LOCALIZATION_REPORTS_ROOT:
    localization_metrics__split=<split>.json   every breakdown (field, family x field, handwritten x field, status)
    localization_report__split=<split>.md      the human tables

Run: python -m experiments.field_reading.field_localization.localization_report --split eval
"""

import argparse
import json
import logging

import pandas as pd

from experiments.field_reading.config import TARGET_FIELD_NAMES
from experiments.field_reading.field_localization.localization_config import (LOCALIZATION_REPORTS_ROOT,
                                                                              localization_prediction_path)
from experiments.field_reading.field_localization.localization_metrics import (score_prediction_rows,
                                                                               summarize_scored_records)
from experiments.field_reading.field_localization.localization_targets import load_localization_checks
from experiments.field_reading.field_localization.prediction_io import read_prediction_rows_by_key

logger = logging.getLogger(__name__)

BREAKDOWNS: dict[str, list[str]] = {
    "by_field": ["field_name"],
    "by_family_field": ["layout_family", "field_name"],
    "by_handwritten_field": ["handwritten", "field_name"],
}


def summarize_method(scored_records: pd.DataFrame) -> dict[str, list[dict]]:
    """All breakdowns for one method: status ok rows, plus the too_small rows by field."""
    ok_or_absent = scored_records[scored_records.status.isin(["ok", "absent"])]
    summaries = {name: summarize_scored_records(ok_or_absent, columns).to_dict("records")
                 for name, columns in BREAKDOWNS.items()}
    summaries["too_small_by_field"] = summarize_scored_records(
        scored_records[scored_records.status == "too_small"], ["field_name"]).to_dict("records")
    summaries["excluded_counts"] = scored_records[scored_records.gt_kind == "excluded"].status.value_counts().to_dict()
    return summaries


def format_cell(summary_row: pd.Series) -> str:
    """'IoU / hit / coverage / area ratio' for one (method, field)."""
    return (f"{summary_row.mean_iou:.3f} / {summary_row.hit_rate:.3f} / {summary_row.coverage_rate:.3f} / "
            f"{summary_row.median_area_ratio:.2f}")


def method_by_field_table(summaries_by_method: dict[str, dict], breakdown: str) -> str:
    """Markdown table: rows = methods, columns = fields, cells = IoU / hit / coverage / area ratio."""
    lines = ["| method | " + " | ".join(TARGET_FIELD_NAMES) + " | mean IoU | mean hit |",
             "|---|" + "---:|" * (len(TARGET_FIELD_NAMES) + 2)]
    for method_id, summaries in summaries_by_method.items():
        by_field = pd.DataFrame(summaries[breakdown]).set_index("field_name")
        cells = [format_cell(by_field.loc[field]) if field in by_field.index else "-" for field in TARGET_FIELD_NAMES]
        lines.append(f"| {method_id} | " + " | ".join(cells) +
                     f" | {by_field.mean_iou.mean():.3f} | {by_field.hit_rate.mean():.3f} |")
    return "\n".join(lines)


def grouped_mean_iou_table(summaries_by_method: dict[str, dict], breakdown: str, group_column: str) -> str:
    """Markdown table: rows = methods, columns = group values, cells = field-averaged mean IoU / hit rate."""
    frames = {method: pd.DataFrame(summaries[breakdown]) for method, summaries in summaries_by_method.items()}
    group_values = sorted(set().union(*[set(frame[group_column].astype(str)) for frame in frames.values()]))
    lines = ["| method | " + " | ".join(group_values) + " |", "|---|" + "---:|" * len(group_values)]
    for method_id, frame in frames.items():
        frame = frame.assign(group_value=frame[group_column].astype(str))
        per_group = frame.groupby("group_value")[["mean_iou", "hit_rate"]].mean()
        lines.append(f"| {method_id} | " + " | ".join(
            f"{per_group.loc[value].mean_iou:.3f} / {per_group.loc[value].hit_rate:.3f}" if value in per_group.index
            else "-" for value in group_values) + " |")
    return "\n".join(lines)


def false_box_table(summaries_by_method: dict[str, dict]) -> str:
    """Markdown table of false-box rate on genuinely absent fields (in practice: memo)."""
    lines = ["| method | absent fields scored | false-box rate |", "|---|---:|---:|"]
    for method_id, summaries in summaries_by_method.items():
        by_field = pd.DataFrame(summaries["by_field"]).dropna(subset=["absent_count"])
        absent_total = by_field.absent_count.sum()
        false_boxes = (by_field.absent_count * by_field.false_box_rate).sum()
        lines.append(f"| {method_id} | {int(absent_total)} | {false_boxes / max(absent_total, 1):.3f} |")
    return "\n".join(lines)


def write_localization_report(split_name: str, method_ids: list[str]) -> None:
    """Score each available method on the split and write the JSON + markdown."""
    check_records = load_localization_checks(split_name)
    summaries_by_method = {}
    for method_id in method_ids:
        prediction_path = localization_prediction_path(split_name, method_id)
        if not prediction_path.exists():
            logger.warning("no predictions for %s on %s; skipping", method_id, split_name)
            continue
        scored = score_prediction_rows(check_records, read_prediction_rows_by_key(prediction_path))
        summaries_by_method[method_id] = summarize_method(scored)
    LOCALIZATION_REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    json_path = LOCALIZATION_REPORTS_ROOT / f"localization_metrics__split={split_name}.json"
    json_path.write_text(json.dumps(summaries_by_method, indent=1, default=str))
    markdown = [
        f"# Field localization, split={split_name} ({len(check_records)} checks)", "",
        "Cells: mean IoU / hit rate (IoU >= 0.5) / text coverage rate (pred covers >= 95% of GT) / median area "
        "ratio pred/GT. A missing box scores IoU 0.", "",
        "## Status ok", "", method_by_field_table(summaries_by_method, "by_field"), "",
        "## Status too_small (tiny in the photo; boxes still exact)", "",
        method_by_field_table(summaries_by_method, "too_small_by_field"), "",
        "## False boxes on absent fields", "", false_box_table(summaries_by_method), "",
        "## By layout family (field-averaged mean IoU / hit rate, status ok)", "",
        grouped_mean_iou_table(summaries_by_method, "by_family_field", "layout_family"), "",
        "## By handwritten (field-averaged mean IoU / hit rate, status ok)", "",
        grouped_mean_iou_table(summaries_by_method, "by_handwritten_field", "handwritten"), "",
    ]
    markdown_path = LOCALIZATION_REPORTS_ROOT / f"localization_report__split={split_name}.md"
    markdown_path.write_text("\n".join(markdown))
    logger.info("wrote %s and %s", json_path, markdown_path)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", default="eval")
    parser.add_argument("--methods", nargs="+", default=None)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    method_ids = arguments.methods or sorted(path.stem for path in
                                             localization_prediction_path(arguments.split, "x").parent.glob("*.jsonl"))
    write_localization_report(arguments.split, method_ids)


if __name__ == "__main__":
    main()
