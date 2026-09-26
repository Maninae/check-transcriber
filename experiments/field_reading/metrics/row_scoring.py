"""Score each (ground-truth row, prediction) pair: correctness, coverage, raw exactness, CER, WER.

Input is the ground-truth field rows joined with one method's predictions (`pred_text`,
`confidence`); output adds one column per metric. Rules:
- `covered` = the method filled the field (`pred_text != ""`). Uncovered rows are never correct.
- `field_correct` = normalized-value equality (`field_value_parsing.normalize_field_value`). The
  truth value is parsed from the GT text; if that fails it falls back to the canonical column.
- `exact_raw` = verbatim string equality.
- `text_correct` = normalized-text equality (the text-level view; differs from `field_correct`
  mainly for amount_words, where value-level ignores spelling/format and text-level does not).
- `cer` / `wer` on normalized text; a blank prediction scores 1.0.
"""

import pandas as pd

from experiments.field_reading.metrics.field_value_parsing import (
    FIELD_NAME_TO_CANONICAL_COLUMN,
    canonical_value_for_field,
    normalize_field_value,
    normalize_free_text,
)
from experiments.field_reading.metrics.text_error_rates import character_error_rate, word_error_rate

SCORE_COLUMN_NAMES = ["covered", "field_correct", "exact_raw", "text_correct", "cer", "wer"]


def ground_truth_value_for_row(field_name: str, ground_truth_text: str, row: pd.Series) -> int | str | None:
    """Parsed GT text, falling back to the check's canonical value when the text does not parse."""
    parsed_value = normalize_field_value(field_name, ground_truth_text)
    if parsed_value is not None or field_name not in FIELD_NAME_TO_CANONICAL_COLUMN:
        return parsed_value
    return canonical_value_for_field(field_name, row.get(FIELD_NAME_TO_CANONICAL_COLUMN[field_name]))


def score_single_row(row: pd.Series) -> dict[str, object]:
    """All per-row metrics for one joined row (needs field_name, text, pred_text, canonical columns)."""
    field_name = row["field_name"]
    ground_truth_text = row["text"]
    predicted_text = row["pred_text"] if isinstance(row["pred_text"], str) else ""
    covered = predicted_text != ""
    normalized_prediction = normalize_free_text(predicted_text)
    normalized_ground_truth = normalize_free_text(ground_truth_text)
    ground_truth_value = ground_truth_value_for_row(field_name, ground_truth_text, row)
    predicted_value = normalize_field_value(field_name, predicted_text) if covered else None
    return {
        "covered": covered,
        "field_correct": covered and predicted_value is not None and predicted_value == ground_truth_value,
        "exact_raw": covered and predicted_text == ground_truth_text,
        "text_correct": covered and normalized_prediction == normalized_ground_truth,
        "cer": character_error_rate(normalized_prediction, normalized_ground_truth),
        "wer": word_error_rate(normalized_prediction, normalized_ground_truth),
        "pred_value": None if predicted_value is None else str(predicted_value),
        "gt_value": None if ground_truth_value is None else str(ground_truth_value),
    }


def score_joined_rows(joined_rows: pd.DataFrame) -> pd.DataFrame:
    """Copy of `joined_rows` with the score columns (plus `pred_value` / `gt_value` for debugging) added."""
    scored_rows = joined_rows.copy()
    if scored_rows.empty:
        for column_name in SCORE_COLUMN_NAMES + ["pred_value", "gt_value"]:
            scored_rows[column_name] = pd.Series(dtype=object)
        return scored_rows
    score_records = [score_single_row(row) for _, row in scored_rows.iterrows()]
    score_frame = pd.DataFrame.from_records(score_records, index=scored_rows.index)
    for column_name in score_frame.columns:
        scored_rows[column_name] = score_frame[column_name]
    return scored_rows
