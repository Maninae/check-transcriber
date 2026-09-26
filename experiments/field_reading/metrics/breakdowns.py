"""Group scored rows into summary tables: n, coverage, accuracy, CER per method x slice.

`summarize_scored_rows` is the one aggregation every table uses, so headline, hard-set,
too-small and breakdown numbers are all computed the same way:
- accuracy = correct / n (blank counts wrong); accuracy_on_filled = correct / covered
- mean_cer / mean_wer include blanks at 1.0
"""

import numpy as np
import pandas as pd

TEXT_HEIGHT_BIN_EDGES_PX = [0.0, 14.0, 18.0, 24.0, 32.0, np.inf]
TEXT_HEIGHT_BIN_LABELS = ["<14", "14-18", "18-24", "24-32", ">=32"]
ALL_FIELDS_LABEL = "ALL"
# Breakdown dimension name -> column in the scored rows (text_height_bin is derived here).
BREAKDOWN_DIMENSION_COLUMNS: dict[str, str] = {
    "handwritten": "handwritten",
    "pen_font_id": "pen_font_id",
    "text_height_bin": "text_height_bin",
    "layout_family": "layout_family",
}


def add_breakdown_columns(scored_rows: pd.DataFrame) -> pd.DataFrame:
    """Add `text_height_bin` and make pen_font_id groupable (printed rows -> "printed")."""
    enriched_rows = scored_rows.copy()
    enriched_rows["text_height_bin"] = pd.cut(enriched_rows.text_height_in_photo_px, TEXT_HEIGHT_BIN_EDGES_PX,
                                              labels=TEXT_HEIGHT_BIN_LABELS, right=False).astype(str)
    enriched_rows["pen_font_id"] = enriched_rows.pen_font_id.fillna("printed")
    enriched_rows["handwritten"] = enriched_rows.handwritten.fillna(False).astype(bool)
    return enriched_rows


def summarize_scored_rows(scored_rows: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """One row per group: n, coverage, accuracy, accuracy_on_filled, exact_raw, text_accuracy, mean_cer, mean_wer."""
    if scored_rows.empty:
        return pd.DataFrame(columns=[*group_columns, "n", "coverage", "accuracy", "accuracy_on_filled",
                                     "exact_raw", "text_accuracy", "mean_cer", "mean_wer"])
    numeric_rows = scored_rows.assign(
        covered_value=scored_rows.covered.astype(float), correct_value=scored_rows.field_correct.astype(float),
        exact_value=scored_rows.exact_raw.astype(float), text_value=scored_rows.text_correct.astype(float),
    )
    summary = numeric_rows.groupby(group_columns, dropna=False, observed=True).agg(
        n=("correct_value", "size"), coverage=("covered_value", "mean"), accuracy=("correct_value", "mean"),
        correct_count=("correct_value", "sum"), covered_count=("covered_value", "sum"),
        exact_raw=("exact_value", "mean"), text_accuracy=("text_value", "mean"),
        mean_cer=("cer", "mean"), mean_wer=("wer", "mean"),
    ).reset_index()
    summary["accuracy_on_filled"] = np.where(summary.covered_count > 0,
                                             summary.correct_count / summary.covered_count.clip(lower=1), np.nan)
    return summary.drop(columns=["correct_count", "covered_count"])


def summarize_per_field_with_total(scored_rows: pd.DataFrame, extra_group_columns: list[str] | None = None) -> pd.DataFrame:
    """Per method x field summary plus an ALL-fields pooled row per method."""
    group_columns = ["method_label", *(extra_group_columns or [])]
    per_field = summarize_scored_rows(scored_rows, [*group_columns, "field_name"])
    pooled = summarize_scored_rows(scored_rows, group_columns).assign(field_name=ALL_FIELDS_LABEL)
    return pd.concat([per_field, pooled], ignore_index=True)


def build_breakdown_tables(scored_rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Every breakdown dimension pooled over fields (method x slice) and per field (method x field x slice)."""
    enriched_rows = add_breakdown_columns(scored_rows)
    tables: dict[str, pd.DataFrame] = {"field_name": summarize_per_field_with_total(enriched_rows)}
    for dimension_name, column_name in BREAKDOWN_DIMENSION_COLUMNS.items():
        tables[dimension_name] = summarize_scored_rows(enriched_rows, ["method_label", column_name])
        tables[f"{dimension_name}__per_field"] = summarize_scored_rows(enriched_rows, ["method_label", "field_name", column_name])
    return tables
