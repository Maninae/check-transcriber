"""Export a CenterNet checkpoint to ONNX (opset 17, static 1x3xSxS) and check parity with PyTorch.

    python -m experiments.detection.learned.centernet.onnx_export --weights <best.pt> [--parity-scenes 5]

Writes `<weights stem>.onnx` next to the checkpoint. Graph I/O:
- input  `input_image` float32 (1, 3, S, S), ImageNet-normalized RGB, letterboxed.
- output `center_heatmap` float32 (1, 1, S/4, S/4), sigmoid already applied.
- output `corner_offsets` float32 (1, 8, S/4, S/4), cells from each cell center.
Decoding (3x3 peak pick, top-K, corner math) stays outside the graph in
`centernet_decoding.py`, the reference for the JS port.

Parity: onnxruntime CPU vs PyTorch CPU on val scenes (1280 copies), reporting the
max absolute map difference and the max decoded corner difference in input pixels.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import torch

from experiments.detection.dataset.scene_annotations import load_split_scene_annotations
from experiments.detection.learned.centernet.centernet_decoding import decode_checks_in_input_pixels
from experiments.detection.learned.centernet.centernet_inference import load_centernet_checkpoint
from experiments.detection.learned.centernet.centernet_model import CenterNetExportWrapper
from experiments.detection.learned.centernet.check_scene_dataset import load_downscaled_scene, normalize_image_to_tensor
from experiments.detection.learned.centernet.scene_augmentation import letterbox_scene_for_evaluation

logger = logging.getLogger(__name__)

ONNX_OPSET_VERSION = 17
INPUT_NAME = "input_image"
OUTPUT_NAMES = ["center_heatmap", "corner_offsets"]
BYTES_PER_MEGABYTE = 1024 * 1024


def export_centernet_to_onnx(weights_path: Path, onnx_path: Path) -> Path:
    """Trace the export wrapper at the training input size and save a checked ONNX model."""
    model, training_config = load_centernet_checkpoint(weights_path, "cpu")
    example_input = torch.zeros(1, 3, training_config.input_size_pixels, training_config.input_size_pixels)
    torch.onnx.export(
        CenterNetExportWrapper(model).eval(),
        (example_input,),
        str(onnx_path),
        input_names=[INPUT_NAME],
        output_names=OUTPUT_NAMES,
        opset_version=ONNX_OPSET_VERSION,
        dynamo=False,  # TorchScript exporter: honours opset 17 exactly and emits a single self-contained file
    )
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    return onnx_path


def check_onnx_parity(weights_path: Path, onnx_path: Path, scene_count: int, split_name: str = "val") -> dict:
    """Compare onnxruntime and PyTorch outputs on real scenes; returns the worst differences."""
    model, training_config = load_centernet_checkpoint(weights_path, "cpu")
    wrapper = CenterNetExportWrapper(model).eval()
    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    worst = {"max_heatmap_abs_diff": 0.0, "max_offset_abs_diff": 0.0, "max_corner_diff_input_px": 0.0, "detection_count_mismatches": 0}
    for scene in load_split_scene_annotations(split_name, limit=scene_count):
        data_dict = letterbox_scene_for_evaluation(load_downscaled_scene(scene), training_config.input_size_pixels)
        input_batch = normalize_image_to_tensor(data_dict["input_image_rgb"])[None]
        with torch.no_grad():
            torch_heatmap, torch_offsets = (tensor.numpy() for tensor in wrapper(input_batch))
        onnx_heatmap, onnx_offsets = session.run(OUTPUT_NAMES, {INPUT_NAME: input_batch.numpy()})
        worst["max_heatmap_abs_diff"] = max(worst["max_heatmap_abs_diff"], float(np.abs(torch_heatmap - onnx_heatmap).max()))
        worst["max_offset_abs_diff"] = max(worst["max_offset_abs_diff"], float(np.abs(torch_offsets - onnx_offsets).max()))
        torch_corners, _ = decode_checks_in_input_pixels(torch_heatmap[0, 0], torch_offsets[0])
        onnx_corners, _ = decode_checks_in_input_pixels(onnx_heatmap[0, 0], onnx_offsets[0])
        if len(torch_corners) != len(onnx_corners):
            worst["detection_count_mismatches"] += 1
        elif len(torch_corners):
            worst["max_corner_diff_input_px"] = max(worst["max_corner_diff_input_px"], float(np.abs(torch_corners - onnx_corners).max()))
    worst["scenes_compared"] = scene_count
    worst["onnx_size_megabytes"] = round(onnx_path.stat().st_size / BYTES_PER_MEGABYTE, 2)
    return worst


def main() -> None:
    """Export, then report parity and size."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None, help="default: <weights>.onnx")
    parser.add_argument("--parity-scenes", type=int, default=5)
    arguments = parser.parse_args()
    onnx_path = export_centernet_to_onnx(arguments.weights, arguments.output or arguments.weights.with_suffix(".onnx"))
    logger.info("exported %s", onnx_path)
    logger.info("parity: %s", check_onnx_parity(arguments.weights, onnx_path, arguments.parity_scenes))


if __name__ == "__main__":
    main()
