"""Sanity check on real handwriting: score reading methods on the 78 hand-labelled SSBI field crops.

Not a benchmark (78 crops, CC BY-NC 4.0: crops never leave vega). Rules from the lead:
- rows with null `text` are skipped; SSBI `amount` = our `amount_numeric`;
- boxes are tight, so each crop is padded with the app's field-crop margin (15% of height,
  >= 6 px) by edge replication before reading;
- date and amount compare the literal string with whitespace removed (dates are DD/MM/YYYY, so
  no parsing); amount_words and payee compare casefolded, whitespace-collapsed text; CER on the
  same normalized strings.
- dates also get `date_digits_any_order`: the prediction's digits equal the truth's DD MM YYYY
  digits or the swapped MM DD YYYY (SSBI is day-first, US checks month-first; the order is a
  formatting fact, not a reading error).

Run: python -m experiments.field_reading.learned.real_ssbi_scoring --methods crnn_general trocr_small_hw_ft
"""

import argparse
import json
import logging
import re

import cv2
import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

from experiments.field_reading.config import FIELD_READING_OUTPUT_ROOT, REPORTS_ROOT
from experiments.field_reading.data_access.field_crop import FIELD_CROP_MARGIN_FRACTION, FIELD_CROP_MIN_MARGIN_PX
from experiments.field_reading.learned.line_crop_dataset import read_rgb_image
from experiments.field_reading.learned.quick_text_scoring import normalize_text_for_comparison
from experiments.field_reading.learned.reading_methods import READING_METHOD_REGISTRY, ReaderPool

logger = logging.getLogger(__name__)

SSBI_ROOT = FIELD_READING_OUTPUT_ROOT / "real_ssbi"
SSBI_FIELD_TO_OUR_FIELD = {"amount": "amount_numeric", "amount_words": "amount_words", "date": "date", "payee": "payee"}
WHITESPACE_FREE_FIELDS = {"amount_numeric", "date"}
NON_DIGIT = re.compile(r"\D")


def date_digits_match_any_order(predicted_text: str, truth_text: str) -> bool:
    """Digits of the prediction equal day-month-year or month-day-year digits of a DD/MM/YYYY truth."""
    day, month, year = truth_text.strip().split("/")
    return NON_DIGIT.sub("", predicted_text) in {day + month + year, month + day + year}


def load_ssbi_rows() -> pd.DataFrame:
    """Labelled rows with our field names; every row is handwritten by construction."""
    rows = pd.DataFrame(json.loads((SSBI_ROOT / "crops_index.json").read_text()))
    rows = rows[rows.text.notna()].copy()
    rows["field_name"] = rows.field.map(SSBI_FIELD_TO_OUR_FIELD)
    rows["handwritten"] = True
    rows["row_key"] = "ssbi_" + rows.idx.astype(str)
    rows["crop_path"] = rows.crop.map(lambda name: str(SSBI_ROOT / "crops" / name))
    return rows.reset_index(drop=True)


def pad_tight_crop(rgb_crop: np.ndarray) -> np.ndarray:
    """Add the app's field-crop margin around a tight box by replicating the edge pixels."""
    margin = int(round(max(FIELD_CROP_MIN_MARGIN_PX, FIELD_CROP_MARGIN_FRACTION * rgb_crop.shape[0])))
    return cv2.copyMakeBorder(rgb_crop, margin, margin, margin, margin, cv2.BORDER_REPLICATE)


def comparable_text(field_name: str, text: str) -> str:
    """The string compared for exact match / CER on this field."""
    normalized = normalize_text_for_comparison(text)
    return normalized.replace(" ", "") if field_name in WHITESPACE_FREE_FIELDS else normalized


def score_method_on_ssbi(method_id: str, rows: pd.DataFrame, crops: list[np.ndarray], pool: ReaderPool) -> pd.DataFrame:
    """Per-row prediction, exact flag and CER for one method."""
    results = READING_METHOD_REGISTRY[method_id](rows, crops, pool)
    records = []
    for row, (text, confidence, _) in zip(rows.itertuples(), results):
        truth = comparable_text(row.field_name, row.text)
        predicted = comparable_text(row.field_name, text)
        records.append({"method": method_id, "row_key": row.row_key, "field": row.field_name, "truth": row.text, "pred": text,
                        "confidence": confidence, "exact": predicted == truth,
                        "cer": Levenshtein.distance(predicted, truth) / max(1, len(truth)),
                        "date_digits_any_order": date_digits_match_any_order(text, row.text) if row.field_name == "date" else None})
    return pd.DataFrame.from_records(records)


def main() -> None:
    """Score the given methods and print/write the per-field table."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", nargs="+", required=True, choices=sorted(READING_METHOD_REGISTRY))
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rows = load_ssbi_rows()
    crops = [pad_tight_crop(read_rgb_image(path)) for path in rows.crop_path]
    pool = ReaderPool(arguments.device)
    scored = pd.concat([score_method_on_ssbi(method_id, rows, crops, pool) for method_id in arguments.methods])
    table = scored.groupby(["method", "field"]).agg(n=("exact", "size"), exact=("exact", "mean"), cer=("cer", "mean")).round(3)
    print(f"real handwriting (SSBI, n={len(rows)})")
    print(table.unstack("field").to_string())
    dates = scored[scored.field == "date"].groupby("method").agg(cer=("cer", "mean"), digits_any_order=("date_digits_any_order", "mean")).round(3)
    print("SSBI dates: CER and digits correct ignoring day/month order")
    print(dates.to_string())
    output_directory = REPORTS_ROOT / "learned"
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"real_ssbi__methods={'+'.join(arguments.methods)}.csv"
    scored.to_csv(output_path, index=False)
    logger.info("per-row results: %s", output_path)


if __name__ == "__main__":
    main()
