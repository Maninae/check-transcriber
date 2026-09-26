"""Batch inference of the field-mask net over check records -> contract prediction rows.

Logits are post-processed batch by batch (a full split of raw logits would be several GB).
Presence gating has two stages: a field needs a component above the 0.5 mask threshold at all, and
its confidence (mean probability inside the box) must reach `min_confidence`, which is selected on val.
"""

import torch
from torch.utils.data import DataLoader

from experiments.field_reading.field_localization.prediction_io import make_prediction_row
from experiments.field_reading.field_localization.segnet_config import CANVAS_WIDTH_PX, FIELD_CHANNEL_NAMES
from experiments.field_reading.field_localization.segnet_dataset import FieldMaskDataset
from experiments.field_reading.field_localization.segnet_postprocess import MASK_THRESHOLD, logits_to_field_boxes

INFERENCE_BATCH_SIZE = 16
INFERENCE_WORKER_COUNT = 2


def predict_field_rows(model: torch.nn.Module, check_records: list[dict], device: torch.device) -> list[dict]:
    """One contract row per (check, field), ungated beyond the mask threshold, in check_records order."""
    loader = DataLoader(FieldMaskDataset(check_records, augment=False), batch_size=INFERENCE_BATCH_SIZE,
                        shuffle=False, num_workers=INFERENCE_WORKER_COUNT)
    model.eval()
    prediction_rows = []
    with torch.no_grad():
        for batch in loader:
            batch_logits = model(batch["canvas_image"].to(device)).float().cpu().numpy()
            for check_position, field_logits in zip(batch["check_index_in_split"].tolist(), batch_logits):
                check_record = check_records[check_position]
                crop_size = tuple(check_record["check_crop_size"])
                field_boxes = logits_to_field_boxes(field_logits, CANVAS_WIDTH_PX / crop_size[0], MASK_THRESHOLD,
                                                    crop_size)
                for field_name, (box, confidence) in zip(FIELD_CHANNEL_NAMES, field_boxes):
                    prediction_rows.append(make_prediction_row(check_record["scene_id"], check_record["check_index"],
                                                               field_name, box, confidence))
    return prediction_rows


def apply_confidence_gate(prediction_rows: list[dict], min_confidence: float) -> list[dict]:
    """Copy of the rows with boxes whose confidence is under `min_confidence` replaced by null."""
    return [row if row["pred_box"] is None or row["confidence"] >= min_confidence else {**row, "pred_box": None}
            for row in prediction_rows]
