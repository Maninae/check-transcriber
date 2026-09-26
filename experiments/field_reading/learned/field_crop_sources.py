"""Where a reader's crops come from: synth's ground-truth field crops or a localizer's boxes.

- `oracle`: the GT `field_crop_path` PNG; pred_box = the GT box (for the contract's pred_box field).
- `<localizer id>`: boxes from `predictions/localization/<split>/<id>.jsonl`, cut from the rectified
  check with `crop_field_from_check` (the app's margin rule). A null or missing box -> crop None,
  which the caller turns into a blank prediction.
"""

import logging

import numpy as np
import pandas as pd

from experiments.field_reading.config import PREDICTIONS_ROOT
from experiments.field_reading.data_access.field_crop import crop_field_from_check
from experiments.field_reading.learned.line_crop_dataset import read_rgb_image

logger = logging.getLogger(__name__)

ORACLE_LOCALIZATION = "oracle"
MIN_CROP_SIDE_PX = 2


def load_localizer_boxes(split_name: str, localizer_id: str) -> dict[str, list[float] | None]:
    """row_key -> predicted box (or None) from a localization prediction file."""
    path = PREDICTIONS_ROOT / "localization" / split_name / f"{localizer_id}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"no localization predictions for {localizer_id} on {split_name}: {path}")
    boxes = pd.read_json(path, lines=True)
    return dict(zip(boxes.row_key, boxes.pred_box.map(lambda box: box if isinstance(box, list) else None)))


def load_field_crops(rows: pd.DataFrame, split_name: str, localization: str) -> tuple[list[np.ndarray | None], list]:
    """RGB crop (or None) and pred_box per row, in row order."""
    if localization == ORACLE_LOCALIZATION:
        return [read_rgb_image(path) for path in rows.field_crop_path], rows.box_in_check_crop.tolist()
    boxes_by_row_key = load_localizer_boxes(split_name, localization)
    missing = int((~rows.row_key.isin(boxes_by_row_key)).sum())
    if missing:
        logger.warning("%d rows have no localization row for %s; they will be blank", missing, localization)
    crops: list[np.ndarray | None] = [None] * len(rows)
    predicted_boxes = [boxes_by_row_key.get(key) for key in rows.row_key]
    for check_path, group in rows.reset_index(drop=True).groupby("check_crop_path"):
        check_image = read_rgb_image(check_path)
        for index in group.index:
            box = predicted_boxes[index]
            if box is None:
                continue
            crop = crop_field_from_check(check_image, box)
            if min(crop.shape[:2]) >= MIN_CROP_SIDE_PX:
                crops[index] = crop
    return crops, predicted_boxes
