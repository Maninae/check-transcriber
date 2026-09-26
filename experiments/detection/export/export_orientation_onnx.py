"""Export the upside-down check classifier to ONNX for the browser app, with a parity check.

    python -m experiments.detection.export.export_orientation_onnx [--parity-crops 512]

- Input `crop`: (1, 1, 96, 224) float32 in [0, 1], a grayscale landscape crop from
  `rectify_check_crop.warp_quadrilateral_to_crop`. Output `upside_down_logit`: (1,),
  > 0 means upside down. Static batch of 1: the app classifies one check at a time.
- Parity: the stored eval crops run through the PyTorch module and the onnxruntime
  session; the logits must agree within `MAXIMUM_LOGIT_DIFFERENCE` and every decision
  must match. Eval crops are only used for parity here, never for choices.
- Writes `<experiments>/export/upside_down_classifier.onnx`; the app ships a copy in
  `app/models/`.
"""

import argparse
import json
import logging

import numpy as np
import onnxruntime
import torch

from experiments.detection.config.paths import DETECTION_EXPERIMENTS_ROOT
from experiments.detection.orientation.build_orientation_crops import ORIENTATION_EXPERIMENTS_DIRECTORY
from experiments.detection.orientation.rectify_check_crop import ORIENTATION_CROP_HEIGHT, ORIENTATION_CROP_WIDTH
from experiments.detection.orientation.train_upside_down_classifier import CLASSIFIER_WEIGHTS_PATH
from experiments.detection.orientation.upside_down_classifier import UpsideDownCheckClassifier

logger = logging.getLogger(__name__)

EXPORT_DIRECTORY = DETECTION_EXPERIMENTS_ROOT / "export"
ONNX_PATH = EXPORT_DIRECTORY / "upside_down_classifier.onnx"
ONNX_OPSET = 17
INPUT_NAME = "crop"
OUTPUT_NAME = "upside_down_logit"
MAXIMUM_LOGIT_DIFFERENCE = 1e-3


def export_classifier_to_onnx() -> UpsideDownCheckClassifier:
    """Load the trained weights, write the ONNX file, return the eval-mode module."""
    classifier = UpsideDownCheckClassifier()
    classifier.load_state_dict(torch.load(CLASSIFIER_WEIGHTS_PATH, map_location="cpu"))
    classifier.eval()
    example_crop = torch.zeros(1, 1, ORIENTATION_CROP_HEIGHT, ORIENTATION_CROP_WIDTH)
    EXPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        classifier, (example_crop,), str(ONNX_PATH), input_names=[INPUT_NAME], output_names=[OUTPUT_NAME],
        opset_version=ONNX_OPSET, dynamo=False,
    )
    return classifier


def check_parity(classifier: UpsideDownCheckClassifier, crop_count: int) -> dict:
    """Largest logit difference and decision agreement over the first `crop_count` eval crops."""
    crops = np.load(ORIENTATION_EXPERIMENTS_DIRECTORY / "eval__crops.npy", mmap_mode="r")[:crop_count]
    labels = np.load(ORIENTATION_EXPERIMENTS_DIRECTORY / "eval__labels.npy")[:crop_count]
    session = onnxruntime.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    torch_logits, onnx_logits = [], []
    for crop in crops:
        crop_tensor = (crop.astype(np.float32) / 255.0)[None, None]
        with torch.no_grad():
            torch_logits.append(float(classifier(torch.from_numpy(crop_tensor))[0]))
        onnx_logits.append(float(session.run([OUTPUT_NAME], {INPUT_NAME: crop_tensor})[0][0]))
    torch_logits, onnx_logits = np.array(torch_logits), np.array(onnx_logits)
    return {
        "crops": int(len(crops)),
        "maximum_logit_difference": float(np.abs(torch_logits - onnx_logits).max()),
        "decisions_agree": bool(np.array_equal(torch_logits > 0, onnx_logits > 0)),
        "onnx_accuracy": float(np.mean((onnx_logits > 0) == (labels > 0))),
        "onnx_bytes": ONNX_PATH.stat().st_size,
    }


def main() -> None:
    """Export, check parity, fail loud on a mismatch."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parity-crops", type=int, default=512)
    arguments = parser.parse_args()
    classifier = export_classifier_to_onnx()
    parity = check_parity(classifier, arguments.parity_crops)
    (EXPORT_DIRECTORY / "upside_down_classifier__parity.json").write_text(json.dumps(parity, indent=2))
    logger.info("parity: %s", parity)
    if parity["maximum_logit_difference"] > MAXIMUM_LOGIT_DIFFERENCE or not parity["decisions_agree"]:
        raise SystemExit(f"ONNX parity failed: {parity}")


if __name__ == "__main__":
    main()
