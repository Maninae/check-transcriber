"""Assemble the full report (JSON-able dict + markdown) from scored rows of one or more methods.

Row sets:
- headline set: status ok rows (`subset=all`), ok rows in the hard set (`hard`), or too_small rows (`too_small`)
- too_small rows are always scored in their own section, never mixed into the headline
Sections: headline (per field x method), gating (per field x method, on the headline set),
hard set (overall + per flag + the easy remainder), too_small, breakdowns, status.
"""

import datetime
import json

import pandas as pd

from experiments.field_reading.config import TARGET_FIELD_NAMES
from experiments.field_reading.metrics.breakdowns import (
    ALL_FIELDS_LABEL,
    TEXT_HEIGHT_BIN_LABELS,
    build_breakdown_tables,
    summarize_per_field_with_total,
    summarize_scored_rows,
)
from experiments.field_reading.metrics.confidence_gating import summarize_gating
from experiments.field_reading.metrics.hard_set import HARD_SET_FLAG_NAMES
from experiments.field_reading.metrics.report_tables import (
    accuracy_and_cer_pivot,
    format_percent,
    gating_table,
    headline_detail_table,
)

SUBSET_NAMES = ["all", "hard", "too_small"]
FIELD_ROW_ORDER = [*TARGET_FIELD_NAMES, ALL_FIELDS_LABEL]
EASY_ROWS_LABEL = "none (easy)"


def frame_to_records(frame: pd.DataFrame) -> list[dict]:
    """DataFrame -> list of JSON-safe dicts (NaN -> null)."""
    return json.loads(frame.to_json(orient="records"))


def order_by_method_then_field(summary: pd.DataFrame) -> pd.DataFrame:
    """Sort a per-field summary by method label, then the canonical field order (ALL last)."""
    field_rank = {field_name: rank for rank, field_name in enumerate(FIELD_ROW_ORDER)}
    return summary.assign(field_rank=summary.field_name.map(field_rank)).sort_values(["method_label", "field_rank"]).drop(columns="field_rank")


def select_headline_rows(scored_rows: pd.DataFrame, subset_name: str) -> pd.DataFrame:
    """The rows the headline, gating and breakdowns are computed on for this subset."""
    if subset_name == "too_small":
        return scored_rows[scored_rows.status.eq("too_small")]
    ok_rows = scored_rows[scored_rows.status.eq("ok")]
    return ok_rows[ok_rows.in_hard_set.astype(bool)] if subset_name == "hard" else ok_rows


def build_gating_entries(headline_rows: pd.DataFrame) -> list[dict]:
    """Gating summary per method x field on the headline rows."""
    entries = []
    for (method_label, field_name), group in headline_rows.groupby(["method_label", "field_name"], sort=True):
        gating = summarize_gating(group.confidence.astype(float).to_numpy(), group.field_correct.to_numpy(bool),
                                  group.covered.to_numpy(bool))
        entries.append({"method_label": method_label, "field_name": field_name, "gating": gating})
    return entries


def build_hard_set_flag_summary(ok_rows: pd.DataFrame) -> pd.DataFrame:
    """Pooled (all fields) summary per method for each hard flag, plus the easy remainder."""
    per_flag_frames = [summarize_scored_rows(ok_rows[ok_rows[flag_name].astype(bool)], ["method_label"]).assign(slice=flag_name)
                       for flag_name in HARD_SET_FLAG_NAMES]
    easy_rows = ok_rows[~ok_rows.in_hard_set.astype(bool)]
    per_flag_frames.append(summarize_scored_rows(easy_rows, ["method_label"]).assign(slice=EASY_ROWS_LABEL))
    return pd.concat(per_flag_frames, ignore_index=True)


def build_methods_by_fields_frame(headline_summary: pd.DataFrame, metric_name: str = "accuracy") -> pd.DataFrame:
    """Methods (rows) x fields (columns, ALL last) pivot of one metric from a per-field summary."""
    pivot = headline_summary.pivot(index="method_label", columns="field_name", values=metric_name)
    return pivot[[name for name in FIELD_ROW_ORDER if name in pivot.columns]]


def build_report(scored_rows: pd.DataFrame, split_name: str, subset_name: str, run_name: str,
                 prediction_paths: list[str], excluded_row_counts: dict[str, int]) -> tuple[dict, str]:
    """Everything as a JSON-able dict, and the compact markdown rendering of it."""
    headline_rows = select_headline_rows(scored_rows, subset_name)
    ok_rows = scored_rows[scored_rows.status.eq("ok")]
    too_small_rows = scored_rows[scored_rows.status.eq("too_small")]
    headline_summary = summarize_per_field_with_total(headline_rows)
    hard_set_summary = summarize_per_field_with_total(ok_rows[ok_rows.in_hard_set.astype(bool)])
    hard_flag_summary = build_hard_set_flag_summary(ok_rows)
    too_small_summary = summarize_per_field_with_total(too_small_rows)
    status_summary = summarize_scored_rows(scored_rows, ["method_label", "status"])
    gating_entries = build_gating_entries(headline_rows)
    breakdown_tables = build_breakdown_tables(headline_rows)
    report = {
        "run_name": run_name,
        "split": split_name,
        "subset": subset_name,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "prediction_files": prediction_paths,
        "method_labels": sorted(scored_rows.method_label.unique().tolist()),
        "headline_row_count_per_method": int(len(headline_rows) / max(scored_rows.method_label.nunique(), 1)),
        "excluded_row_counts": excluded_row_counts,
        "headline": frame_to_records(headline_summary),
        "gating": gating_entries,
        "hard_set": frame_to_records(hard_set_summary),
        "hard_set_by_flag": frame_to_records(hard_flag_summary),
        "too_small": frame_to_records(too_small_summary),
        "status": frame_to_records(status_summary),
        "breakdowns": {name: frame_to_records(table) for name, table in breakdown_tables.items()},
    }
    sections = [
        f"# Field reading: {run_name}",
        f"split `{split_name}`, subset `{subset_name}`, {report['headline_row_count_per_method']} headline rows per method; "
        f"excluded (not scored): {excluded_row_counts}. acc = correct / all rows (blank = wrong); CER includes blanks at 1.0.",
        "## Headline (acc% / CER)", accuracy_and_cer_pivot(headline_summary, "field_name", FIELD_ROW_ORDER),
        "## Headline detail", headline_detail_table(order_by_method_then_field(headline_summary)),
        "## Confidence gating", "cov@X%acc = max coverage keeping accuracy >= X, @ confidence threshold.",
        gating_table(gating_entries),
        "## Hard set (ok rows with any hard flag)", accuracy_and_cer_pivot(hard_set_summary, "field_name", FIELD_ROW_ORDER),
        "### By hard flag (all fields pooled; a row can carry several flags)",
        accuracy_and_cer_pivot(hard_flag_summary, "slice", [*HARD_SET_FLAG_NAMES, EASY_ROWS_LABEL]),
        "## too_small rows (text < 14 px, scored separately)", accuracy_and_cer_pivot(too_small_summary, "field_name", FIELD_ROW_ORDER),
        "## Status (ok vs too_small, all fields pooled)", accuracy_and_cer_pivot(status_summary, "status"),
        "## Breakdowns (headline rows, all fields pooled)",
        "### Handwritten", accuracy_and_cer_pivot(breakdown_tables["handwritten"], "handwritten"),
        "### Pen font", accuracy_and_cer_pivot(breakdown_tables["pen_font_id"], "pen_font_id"),
        "### Text height in photo (px)", accuracy_and_cer_pivot(breakdown_tables["text_height_bin"], "text_height_bin", TEXT_HEIGHT_BIN_LABELS),
        "### Layout family", accuracy_and_cer_pivot(breakdown_tables["layout_family"], "layout_family"),
    ]
    amount_words_rows = headline_summary[headline_summary.field_name.eq("amount_words")]
    if not amount_words_rows.empty:
        sections.append("amount_words value-level vs text-level acc%: " + "; ".join(
            f"{record.method_label} {format_percent(record.accuracy)} vs {format_percent(record.text_accuracy)}"
            for record in amount_words_rows.itertuples()))
    return report, "\n\n".join(sections) + "\n"
