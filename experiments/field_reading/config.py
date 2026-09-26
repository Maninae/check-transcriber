"""Paths and constants shared by every field-reading module.

- The v1 synthetic dataset is read in place on vega; nothing here writes into it.
- Everything this area produces (train crops, predictions, reports, weights) lives under
  `FIELD_READING_OUTPUT_ROOT` / `FIELD_READING_MODEL_ROOT` on vega, never the internal disk.
"""

from enum import Enum
from pathlib import Path

SYNTH_V1_ROOT = Path("/Volumes/vega/datasets/check-transcriber/synth/v1")
REAL_SAMPLES_ROOT = Path("/Volumes/vega/datasets/check-transcriber/samples")
FIELD_READING_OUTPUT_ROOT = Path("/Volumes/vega/datasets/check-transcriber/field-reading")
# Train split OCR crops, cut by data_access/train_crop_export.py in the same layout as synth v1's ocr/.
TRAIN_CROPS_ROOT = FIELD_READING_OUTPUT_ROOT / "derived"
PREDICTIONS_ROOT = FIELD_READING_OUTPUT_ROOT / "predictions"
REPORTS_ROOT = FIELD_READING_OUTPUT_ROOT / "reports"
FIELD_READING_MODEL_ROOT = Path("/Volumes/vega/ai-models/field-reading")
TESSDATA_ROOT = FIELD_READING_MODEL_ROOT / "tessdata"   # eng.traineddata = the app's 4.0.0_best_int


class CheckField(str, Enum):
    """The fields the app reads. MICR is never read (spec section 7); signature/bank/address are not shown."""

    PAYER_NAME = "payer_name"
    PAYEE = "payee"
    AMOUNT_NUMERIC = "amount_numeric"
    AMOUNT_WORDS = "amount_words"
    DATE = "date"
    MEMO = "memo"
    CHECK_NUMBER = "check_number"


TARGET_FIELD_NAMES: list[str] = [field.value for field in CheckField]
# Printed-only fields present on every check; used as extra printed training text, never scored.
AUXILIARY_PRINTED_FIELD_NAMES: list[str] = ["bank_name", "payer_address"]
NEVER_READ_FIELD_NAMES: list[str] = ["micr"]
