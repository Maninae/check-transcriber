"""Paths, method ids and constants for field localization (unit U2).

- Every artifact lands on vega: predictions under PREDICTIONS_ROOT/localization, reports under
  REPORTS_ROOT/localization, weights and ONNX under FIELD_READING_MODEL_ROOT/localization.
- Box convention everywhere: [x0, y0, x1, y1] pixel-edge coordinates in the 1600 px rectified check crop.
"""

from pathlib import Path

from experiments.field_reading.config import (FIELD_READING_MODEL_ROOT, FIELD_READING_OUTPUT_ROOT, PREDICTIONS_ROOT,
                                              REPORTS_ROOT)

LOCALIZATION_PREDICTIONS_ROOT = PREDICTIONS_ROOT / "localization"
LOCALIZATION_REPORTS_ROOT = REPORTS_ROOT / "localization"
LOCALIZATION_MODEL_ROOT = FIELD_READING_MODEL_ROOT / "localization"
LAYOUT_PRIOR_JSON_PATH = LOCALIZATION_MODEL_ROOT / "layout_priors__split=train.json"
FIELD_READING_LOGS_ROOT = FIELD_READING_OUTPUT_ROOT / "logs"
MPS_LOCK_DIRECTORY = FIELD_READING_LOGS_ROOT / "MPS_LOCK"
PRETRAINED_WEIGHTS_CACHE_ROOT = FIELD_READING_MODEL_ROOT / "hf"

METHOD_LAYOUT_PRIOR_FAMILY_ORACLE = "layout_prior_family_oracle"
METHOD_LAYOUT_PRIOR_SIZE_KIND = "layout_prior_size_kind"
# Prior grouping column per classical method: oracle layout family (upper bound) vs aspect-inferable size kind.
LAYOUT_PRIOR_GROUP_COLUMN_BY_METHOD: dict[str, str] = {
    METHOD_LAYOUT_PRIOR_FAMILY_ORACLE: "layout_family",
    METHOD_LAYOUT_PRIOR_SIZE_KIND: "size_kind",
}

RECTIFIED_CHECK_WIDTH_PX = 1600
# Statuses whose GT box is trustworthy for scoring (occluded / partially out of frame are excluded).
SCORED_STATUSES = ("ok", "too_small")
# Statuses whose box is still a valid training / prior target (the box is right even if the ink is hidden).
BOX_TARGET_STATUSES = ("ok", "too_small", "occluded", "partially_out_of_frame")


def localization_prediction_path(split_name: str, method_id: str) -> Path:
    """Contract path for one method's localization predictions on a split."""
    return LOCALIZATION_PREDICTIONS_ROOT / split_name / f"{method_id}.jsonl"
