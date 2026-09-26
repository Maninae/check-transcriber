"""Write reading predictions for a split in the PLAN.md contract.

Examples (from the worktree root):
    python -m experiments.field_reading.learned.predict --method crnn_general --split val
    python -m experiments.field_reading.learned.predict --method crnn_amount_route --split eval --localization segnet_v1
    python -m experiments.field_reading.learned.predict --method trocr_small_printed_zs --split val --rows-per-field 300

- Rows: every target-field row with status ok or too_small (one prediction row each).
- `--rows-per-field N` screens a stratified subset (half handwritten where the field has any) and
  writes under `field-reading/learned_screening/` instead of the contract directory.
- `--cleanup` applies ocr_baselines' `clean_field_text` to every prediction (method id gets `+clean`).
- MPS runs hold the area MPS lock.
"""

import argparse
import contextlib
import logging

import pandas as pd
import torch

from experiments.field_reading.config import FIELD_READING_OUTPUT_ROOT
from experiments.field_reading.data_access.field_manifest import load_field_rows
from experiments.field_reading.learned.context_crop_export import SCORED_STATUSES
from experiments.field_reading.learned.field_crop_sources import ORACLE_LOCALIZATION, load_field_crops
from experiments.field_reading.learned.mps_lock import hold_mps_lock
from experiments.field_reading.learned.prediction_rows import (ReadingPredictionRow, prediction_file_path,
                                                               write_prediction_rows)
from experiments.field_reading.learned.reading_methods import READING_METHOD_REGISTRY, ReaderPool
from experiments.field_reading.ocr_baselines.field_text_cleanup import clean_field_text

logger = logging.getLogger(__name__)

SCREENING_ROOT = FIELD_READING_OUTPUT_ROOT / "learned_screening"
SCREENING_SEED = 11


def select_scored_rows(split_name: str, rows_per_field: int | None) -> pd.DataFrame:
    """Scored target-field rows, optionally a per-field subset balanced between handwritten and printed."""
    rows = load_field_rows(split_name)
    rows = rows[rows.status.isin(SCORED_STATUSES)]
    if rows_per_field is None:
        return rows.reset_index(drop=True)
    sampled_indices = []
    for _, field_rows in rows.groupby("field_name"):
        styles = field_rows.groupby("handwritten").groups
        share = rows_per_field // len(styles)
        for style_index in styles.values():
            sampled_indices += field_rows.loc[style_index].sample(min(share, len(style_index)),
                                                                  random_state=SCREENING_SEED).index.tolist()
    return rows.loc[sampled_indices].reset_index(drop=True)


def predict_rows(method_id: str, rows: pd.DataFrame, split_name: str, localization: str, device: str,
                 apply_cleanup: bool, pool: ReaderPool | None = None) -> list[ReadingPredictionRow]:
    """Run one method over rows and build contract rows (blank where there is no crop)."""
    crops, predicted_boxes = load_field_crops(rows, split_name, localization)
    has_crop = [crop is not None for crop in crops]
    readable_rows = rows[has_crop].reset_index(drop=True)
    readable_crops = [crop for crop in crops if crop is not None]
    results = READING_METHOD_REGISTRY[method_id](readable_rows, readable_crops, pool or ReaderPool(device))
    result_iterator = iter(results)
    method_label = f"{method_id}+clean" if apply_cleanup else method_id
    prediction_rows = []
    for row, crop_present, predicted_box in zip(rows.itertuples(), has_crop, predicted_boxes):
        text, confidence, latency_ms = next(result_iterator) if crop_present else ("", 0.0, None)
        if apply_cleanup and text:
            text = clean_field_text(row.field_name, text)
        prediction_rows.append(ReadingPredictionRow(
            row_key=row.row_key, scene_id=row.scene_id, check_index=int(row.check_index), field_name=row.field_name,
            method=method_label, localization=localization, pred_text=text,
            confidence=confidence if text else 0.0, pred_box=predicted_box,
            latency_ms=None if latency_ms is None else round(latency_ms, 3)))
    return prediction_rows


def main() -> None:
    """Parse arguments, run the method, write the prediction file."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", required=True, choices=sorted(READING_METHOD_REGISTRY))
    parser.add_argument("--split", required=True, choices=["val", "eval"])
    parser.add_argument("--localization", default=ORACLE_LOCALIZATION)
    parser.add_argument("--rows-per-field", type=int, default=None)
    parser.add_argument("--cleanup", action="store_true")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rows = select_scored_rows(arguments.split, arguments.rows_per_field)
    logger.info("%s on %s: %d rows, localization %s", arguments.method, arguments.split, len(rows), arguments.localization)
    lock = hold_mps_lock(f"predict {arguments.method}") if arguments.device == "mps" else contextlib.nullcontext()
    with lock:
        prediction_rows = predict_rows(arguments.method, rows, arguments.split, arguments.localization, arguments.device,
                                       arguments.cleanup)
    method_label = prediction_rows[0].method
    if arguments.rows_per_field is None:
        output_path = prediction_file_path(arguments.split, method_label, arguments.localization)
    else:
        output_path = SCREENING_ROOT / arguments.split / f"{method_label}__loc={arguments.localization}.jsonl"
    write_prediction_rows(prediction_rows, output_path)


if __name__ == "__main__":
    main()
