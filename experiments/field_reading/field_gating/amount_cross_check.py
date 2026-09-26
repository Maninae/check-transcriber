"""Spec stage 7 amount cross-check: the courtesy amount is trusted when the legal (words) line agrees.

For each check, parse the courtesy read and the words read to cents with the metrics parsers.
- Both parse and agree -> amount confidence = AGREEMENT_CONFIDENCE_FLOOR + (1 - floor) * reader confidence
  (always above any non-agreeing row, so agreeing rows fill first).
- Otherwise -> reader confidence * DISAGREEMENT_CONFIDENCE_SCALE (shown unsure / blank in the app).
The courtesy text itself is unchanged; only its confidence moves. When the courtesy read does not
parse but the words line does, the words value is used as the courtesy text (amount recovered from
the legal line), at the scaled-down confidence.

Writes `<numeric method>+xcheck__loc=<loc>.jsonl` next to the inputs.

Run: python -m experiments.field_reading.field_gating.amount_cross_check --split eval \
        --numeric-method crnn_general__loc=oracle --words-method crnn_general__loc=oracle
"""

import argparse
import json
import logging

import pandas as pd

from experiments.field_reading.config import PREDICTIONS_ROOT
from experiments.field_reading.metrics.money_amount_parsing import parse_amount_numeric_to_cents, parse_amount_words_to_cents

logger = logging.getLogger(__name__)

AGREEMENT_CONFIDENCE_FLOOR = 0.9
DISAGREEMENT_CONFIDENCE_SCALE = 0.5


def cents_to_amount_text(cents: int) -> str:
    """Two-decimal amount text, as the app would copy it."""
    return f"{cents // 100}.{cents % 100:02d}"


def cross_checked_amount_rows(numeric_rows: pd.DataFrame, words_rows: pd.DataFrame) -> pd.DataFrame:
    """Courtesy-amount prediction rows with confidence re-scored by agreement with the words line."""
    words_by_check = {(r.scene_id, r.check_index): r for r in words_rows.itertuples()}
    output = []
    for row in numeric_rows.itertuples():
        record = row._asdict()
        record.pop("Index", None)
        numeric_cents = parse_amount_numeric_to_cents(row.pred_text or "")
        words_row = words_by_check.get((row.scene_id, row.check_index))
        words_cents = parse_amount_words_to_cents(words_row.pred_text or "") if words_row is not None else None
        confidence = float(row.confidence) if pd.notna(row.confidence) else 0.0
        if numeric_cents is not None and numeric_cents == words_cents:
            record["confidence"] = AGREEMENT_CONFIDENCE_FLOOR + (1 - AGREEMENT_CONFIDENCE_FLOOR) * confidence
        else:
            record["confidence"] = confidence * DISAGREEMENT_CONFIDENCE_SCALE
            if numeric_cents is None and words_cents is not None:
                record["pred_text"] = cents_to_amount_text(words_cents)
        output.append(record)
    return pd.DataFrame(output)


def main() -> None:
    """Write the cross-checked courtesy-amount predictions (other fields copied from the numeric method)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", required=True)
    parser.add_argument("--numeric-method", required=True, help="prediction file stem for amount_numeric")
    parser.add_argument("--words-method", required=True, help="prediction file stem for amount_words")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    directory = PREDICTIONS_ROOT / arguments.split
    numeric_all = pd.read_json(directory / f"{arguments.numeric_method}.jsonl", lines=True)
    words_all = pd.read_json(directory / f"{arguments.words_method}.jsonl", lines=True)
    numeric_rows = numeric_all[numeric_all.field_name == "amount_numeric"]
    words_rows = words_all[words_all.field_name == "amount_words"]
    checked = cross_checked_amount_rows(numeric_rows, words_rows)
    method_stem, localization = arguments.numeric_method.split("__loc=")
    checked["method"] = f"{method_stem}+xcheck"
    others = numeric_all[numeric_all.field_name != "amount_numeric"].assign(method=f"{method_stem}+xcheck")
    combined = pd.concat([others, checked], ignore_index=True)
    output_path = directory / f"{method_stem}+xcheck__loc={localization}.jsonl"
    output_path.write_text("".join(json.dumps(r, default=str) + "\n" for r in combined.to_dict("records")))
    agreeing = (checked.confidence >= AGREEMENT_CONFIDENCE_FLOOR).mean()
    logger.info("wrote %s; numeric and words agree on %.1f%% of amount rows", output_path, 100 * agreeing)


if __name__ == "__main__":
    main()
