"""Run the trained field-mask net on val and eval, select the confidence gate on val, write predictions.

Gate selection: the highest `min_confidence` whose val mean IoU (status ok) stays within
GATE_IOU_TOLERANCE of the best, which buys the lowest false-box rate on absent fields for ~free.
The selected gate is saved next to the checkpoint (`<method>__gate.json`) for the ONNX/browser path.

Run: python -m experiments.field_reading.field_localization.segnet_predict
"""

import json
import logging
import os

import numpy as np
import torch

from experiments.field_reading.field_localization.localization_config import (LOCALIZATION_MODEL_ROOT,
                                                                              PRETRAINED_WEIGHTS_CACHE_ROOT,
                                                                              localization_prediction_path)
from experiments.field_reading.field_localization.localization_metrics import (score_prediction_rows,
                                                                               summarize_scored_records)
from experiments.field_reading.field_localization.localization_targets import load_localization_checks
from experiments.field_reading.field_localization.mps_lock import hold_mps_lock
from experiments.field_reading.field_localization.prediction_io import write_prediction_rows
from experiments.field_reading.field_localization.segnet_config import SEGNET_METHOD_ID
from experiments.field_reading.field_localization.segnet_inference import apply_confidence_gate, predict_field_rows
from experiments.field_reading.field_localization.segnet_model import FieldMaskSegmentationNet

logger = logging.getLogger(__name__)

CANDIDATE_CONFIDENCE_GATES = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
GATE_IOU_TOLERANCE = 0.002


def load_trained_field_mask_net(device: torch.device) -> torch.nn.Module:
    """The best checkpoint from segnet_train, on `device`, in eval mode."""
    checkpoint = torch.load(LOCALIZATION_MODEL_ROOT / f"{SEGNET_METHOD_ID}.pt", map_location="cpu")
    model = FieldMaskSegmentationNet(pretrained_encoder=False)
    model.load_state_dict(checkpoint["model_state"])
    logger.info("loaded checkpoint from epoch %d (val mean IoU %.4f)", checkpoint["epoch"], checkpoint["val_mean_iou"])
    return model.to(device).eval()


def select_confidence_gate(val_checks: list[dict], val_rows: list[dict]) -> tuple[float, list[dict]]:
    """Pick the gate (see module docstring); returns it and the per-gate val summary for the log."""
    gate_summaries = []
    for gate in CANDIDATE_CONFIDENCE_GATES:
        gated_rows = apply_confidence_gate(val_rows, gate)
        scored = score_prediction_rows(val_checks, {row["row_key"]: row for row in gated_rows})
        by_field = summarize_scored_records(scored[scored.status.isin(["ok", "absent"])], ["field_name"])
        absent = scored[scored.gt_kind == "absent"]
        gate_summaries.append({"gate": gate, "val_mean_iou": float(by_field.mean_iou.mean()),
                               "val_false_box_rate": float(absent.has_prediction.mean())})
    best_iou = max(summary["val_mean_iou"] for summary in gate_summaries)
    chosen = max(summary["gate"] for summary in gate_summaries if summary["val_mean_iou"] >= best_iou - GATE_IOU_TOLERANCE)
    return chosen, gate_summaries


def main() -> None:
    """CLI entry point: predict val + eval under the MPS lock, then gate and write."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.environ.setdefault("HF_HOME", str(PRETRAINED_WEIGHTS_CACHE_ROOT))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    checks_by_split = {split_name: load_localization_checks(split_name) for split_name in ("val", "eval")}
    with hold_mps_lock("segnet val/eval inference"):
        model = load_trained_field_mask_net(device)
        rows_by_split = {split_name: predict_field_rows(model, checks, device)
                         for split_name, checks in checks_by_split.items()}
    chosen_gate, gate_summaries = select_confidence_gate(checks_by_split["val"], rows_by_split["val"])
    for summary in gate_summaries:
        logger.info("gate %.2f: val mean IoU %.4f, false-box rate %.3f", summary["gate"], summary["val_mean_iou"],
                    summary["val_false_box_rate"])
    logger.info("selected confidence gate %.2f", chosen_gate)
    gate_path = LOCALIZATION_MODEL_ROOT / f"{SEGNET_METHOD_ID}__gate.json"
    gate_path.write_text(json.dumps({"min_confidence": chosen_gate, "selection": gate_summaries}, indent=1))
    for split_name, rows in rows_by_split.items():
        output_path = localization_prediction_path(split_name, SEGNET_METHOD_ID)
        write_prediction_rows(apply_confidence_gate(rows, chosen_gate), output_path)
        logger.info("wrote %d rows to %s", len(rows), output_path)


if __name__ == "__main__":
    np.seterr(over="ignore")
    main()
