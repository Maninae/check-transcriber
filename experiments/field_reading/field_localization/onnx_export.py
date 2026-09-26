"""Export the trained field-mask net to ONNX and verify it: equivalence, size, CPU latency.

- Equivalence on 20 val crops: max abs logit difference PyTorch (CPU) vs onnxruntime (CPU), and whether
  the post-processed boxes are identical for every field.
- Latency: onnxruntime CPU session, median over the 20 crops after warm-up (model forward only, and
  forward + post-process).
Writes <method>.onnx under LOCALIZATION_MODEL_ROOT and onnx_export__<method>.json under the reports dir.

Run: python -m experiments.field_reading.field_localization.onnx_export
"""

import json
import logging
import time

import numpy as np
import onnxruntime
import torch

from experiments.field_reading.field_localization.localization_config import (LOCALIZATION_MODEL_ROOT,
                                                                              LOCALIZATION_REPORTS_ROOT)
from experiments.field_reading.field_localization.localization_targets import load_localization_checks
from experiments.field_reading.field_localization.segnet_config import (CANVAS_HEIGHT_PX, CANVAS_WIDTH_PX,
                                                                        ENCODER_NAME, SEGNET_METHOD_ID)
from experiments.field_reading.field_localization.segnet_dataset import FieldMaskDataset
from experiments.field_reading.field_localization.segnet_postprocess import MASK_THRESHOLD, logits_to_field_boxes
from experiments.field_reading.field_localization.segnet_predict import load_trained_field_mask_net

logger = logging.getLogger(__name__)

ONNX_OPSET = 17
EQUIVALENCE_CHECK_COUNT = 20
LATENCY_WARMUP_RUNS = 3


def main() -> None:
    """Export, compare and time (see module docstring)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    model = load_trained_field_mask_net(torch.device("cpu"))
    onnx_path = LOCALIZATION_MODEL_ROOT / f"{SEGNET_METHOD_ID}.onnx"
    torch.onnx.export(model, torch.zeros(1, 3, CANVAS_HEIGHT_PX, CANVAS_WIDTH_PX), str(onnx_path),
                      input_names=["canvas_rgb"], output_names=["field_logits"], opset_version=ONNX_OPSET,
                      dynamic_axes={"canvas_rgb": {0: "batch"}, "field_logits": {0: "batch"}}, dynamo=False)
    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    val_checks = load_localization_checks("val")[::190][:EQUIVALENCE_CHECK_COUNT]
    dataset = FieldMaskDataset(val_checks, augment=False)
    max_abs_differences, identical_box_checks, forward_ms, end_to_end_ms = [], 0, [], []
    for sample_index, check_record in enumerate(val_checks):
        canvas = dataset[sample_index]["canvas_image"][None]
        with torch.no_grad():
            torch_logits = model(canvas).numpy()[0]
        for _ in range(LATENCY_WARMUP_RUNS if sample_index == 0 else 0):
            session.run(None, {"canvas_rgb": canvas.numpy()})
        start = time.perf_counter()
        onnx_logits = session.run(None, {"canvas_rgb": canvas.numpy()})[0][0]
        forward_ms.append((time.perf_counter() - start) * 1000)
        crop_size = tuple(check_record["check_crop_size"])
        scale = CANVAS_WIDTH_PX / crop_size[0]
        onnx_boxes = logits_to_field_boxes(onnx_logits, scale, MASK_THRESHOLD, crop_size)
        end_to_end_ms.append((time.perf_counter() - start) * 1000)
        torch_boxes = logits_to_field_boxes(torch_logits, scale, MASK_THRESHOLD, crop_size)
        max_abs_differences.append(float(np.abs(torch_logits - onnx_logits).max()))
        identical_box_checks += all(torch_box == onnx_box for (torch_box, _), (onnx_box, _) in zip(torch_boxes, onnx_boxes))
    export_summary = {
        "onnx_path": str(onnx_path), "opset": ONNX_OPSET, "input": [1, 3, CANVAS_HEIGHT_PX, CANVAS_WIDTH_PX],
        "file_size_mb": round(onnx_path.stat().st_size / 1e6, 2),
        "parameter_count_millions": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
        "equivalence_checks": len(val_checks), "max_abs_logit_difference": max(max_abs_differences),
        "checks_with_identical_boxes": identical_box_checks,
        "ort_cpu_forward_ms_median": round(float(np.median(forward_ms)), 1),
        "ort_cpu_forward_plus_postprocess_ms_median": round(float(np.median(end_to_end_ms)), 1),
        "onnxruntime_version": onnxruntime.__version__,
        "pretrained_weights": f"timm {ENCODER_NAME}, Apache-2.0 (huggingface.co/timm/mobilenetv3_large_100.ra_in1k), "
                              "trained on ImageNet-1k",
    }
    LOCALIZATION_REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    (LOCALIZATION_REPORTS_ROOT / f"onnx_export__{SEGNET_METHOD_ID}.json").write_text(json.dumps(export_summary, indent=1))
    logger.info("%s", json.dumps(export_summary, indent=1))


if __name__ == "__main__":
    main()
