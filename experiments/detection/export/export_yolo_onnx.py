"""Export the YOLO26n-OBB checkpoint to ONNX, check parity with PyTorch, time it on CPU.

    python -m experiments.detection.export.export_yolo_onnx --weights <best.pt> [--parity-scenes 20]

- Export: Ultralytics `format="onnx"`, static 1x3xSxS input, opset 17, onnxslim.
  The default (`nms=None`) keeps the one-to-many head: output (1, 6, A) raw anchors
  (cx, cy, w, h, score, angle), so the browser must threshold and run rotated NMS.
  YOLO26's NMS-free one-to-one head (`nms=False`, output (1, 300, 7)) was measured
  and rejected: on val it left duplicates/misses in 6-8% of scenes at any threshold,
  versus 0% for one-to-many + NMS.
- Parity, two levels on eval scenes:
  1. strict: the SAME square-letterboxed tensor through the PyTorch module and the
     onnxruntime session; raw head outputs compared element-wise;
  2. end to end: `YoloCheckDetector` with the .pt vs the .onnx. This differs by up to
     ~20 px on a few checks because Ultralytics letterboxes a .pt to a minimal
     rectangle (1024x768) but the static .onnx to a 1024 square, i.e. different
     inputs, not a numerical mismatch. The browser uses the square (as scored on val).
- Latency: a bare onnxruntime session (no Ultralytics) on a letterboxed input, at 1
  and at 4 intra-op threads, as the proxy for onnxruntime-web WASM (expect WASM to be
  roughly 1.5-3x slower than native single-thread, SIMD+threads narrowing the gap).
"""

import argparse
import json
import logging
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

from experiments.detection.config.paths import DETECTION_EXPERIMENTS_ROOT
from experiments.detection.dataset.scene_annotations import load_split_scene_annotations
from experiments.detection.learned.yolo_inference import YoloCheckDetector, resize_to_training_scale

logger = logging.getLogger(__name__)

EXPORT_DIRECTORY = DETECTION_EXPERIMENTS_ROOT / "export"
ONNX_OPSET = 17
LATENCY_WARMUP_RUNS = 3
LATENCY_TIMED_RUNS = 20


def export_checkpoint_to_onnx(weights_path: Path, image_size: int, model_name: str) -> Path:
    """Export and copy the .onnx next to the other export artifacts; returns its path."""
    exported_path = Path(
        YOLO(str(weights_path)).export(
            format="onnx", imgsz=image_size, opset=ONNX_OPSET, simplify=True, dynamic=False
        )
    )
    EXPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    onnx_path = EXPORT_DIRECTORY / f"{model_name}__imgsz={image_size}.onnx"
    shutil.copyfile(exported_path, onnx_path)
    return onnx_path


def match_corner_sets(reference_corner_sets: list[np.ndarray], candidate_corner_sets: list[np.ndarray]) -> list[float]:
    """Max per-corner distance for each reference quad against its nearest candidate quad.

    A rotated box has no canonical first corner (an angle near 0 or 90 degrees can flip
    which corner is listed first), so every cyclic shift of the candidate is tried.
    """
    distances = []
    for reference_corners in reference_corner_sets:
        best = min(
            (
                np.linalg.norm(np.roll(candidate, shift, axis=0) - reference_corners, axis=1).max()
                for candidate in candidate_corner_sets
                for shift in range(4)
            ),
            default=float("inf"),
        )
        distances.append(float(best))
    return distances


def measure_end_to_end_parity(weights_path: Path, onnx_path: Path, image_size: int, scene_count: int) -> dict:
    """Compare PyTorch and ONNX detections on the first `scene_count` eval scenes."""
    torch_detector = YoloCheckDetector(str(weights_path), image_size, device="cpu")
    onnx_detector = YoloCheckDetector(str(onnx_path), image_size, device="cpu")
    onnx_detector.task_name = "obb"  # Ultralytics cannot always infer the task from an .onnx
    corner_distances, count_mismatches, score_differences = [], 0, []
    for scene in load_split_scene_annotations("eval", limit=scene_count):
        image_bgr = cv2.imread(str(scene.image_path), cv2.IMREAD_COLOR)
        torch_detections = torch_detector.detect_checks(image_bgr)
        onnx_detections = onnx_detector.detect_checks(image_bgr)
        count_mismatches += int(len(torch_detections) != len(onnx_detections))
        corner_distances += match_corner_sets(
            [detection.corners for detection in torch_detections], [detection.corners for detection in onnx_detections]
        )
        score_differences += [
            abs(a.score - b.score)
            for a, b in zip(
                sorted(torch_detections, key=lambda d: d.score), sorted(onnx_detections, key=lambda d: d.score)
            )
        ]
    return {
        "scenes": scene_count,
        "checks_compared": len(corner_distances),
        "scenes_with_count_mismatch": count_mismatches,
        "max_corner_difference_full_res_pixels": max(corner_distances),
        "median_corner_difference_full_res_pixels": float(np.median(corner_distances)),
        "max_score_difference": max(score_differences),
    }


def measure_raw_output_parity(weights_path: Path, onnx_path: Path, image_size: int, scene_count: int) -> dict:
    """Same letterboxed tensor through PyTorch and onnxruntime; max abs difference of raw outputs."""
    torch_module = YOLO(str(weights_path)).model.float().eval()
    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    letterbox = LetterBox((image_size, image_size), auto=False)
    box_differences, score_differences = [], []
    for scene in load_split_scene_annotations("eval", limit=scene_count):
        resized_bgr, _ = resize_to_training_scale(cv2.imread(str(scene.image_path), cv2.IMREAD_COLOR))
        letterboxed_rgb = letterbox(image=resized_bgr)[:, :, ::-1].transpose(2, 0, 1).copy()
        input_tensor = torch.from_numpy(letterboxed_rgb).float().unsqueeze(0) / 255.0
        with torch.no_grad():
            torch_output = torch_module(input_tensor)
        while isinstance(torch_output, (tuple, list)):
            torch_output = torch_output[0]
        onnx_output = session.run(None, {input_name: input_tensor.numpy()})[0]
        difference = np.abs(torch_output.numpy() - onnx_output)  # rows: cx, cy, w, h, score, angle
        box_differences.append(float(difference[:, :4].max()))
        score_differences.append(float(difference[:, 4].max()))
    return {
        "scenes": scene_count,
        "max_box_difference_input_pixels": max(box_differences),
        "max_score_difference": max(score_differences),
    }


def measure_cpu_latency(onnx_path: Path, image_size: int) -> dict:
    """Median bare-onnxruntime latency of one forward pass at 1 and 4 threads."""
    input_tensor = np.random.default_rng(0).random((1, 3, image_size, image_size), dtype=np.float32)
    latency_by_threads = {}
    for thread_count in (1, 4):
        session_options = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = thread_count
        session = onnxruntime.InferenceSession(str(onnx_path), session_options, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        for _ in range(LATENCY_WARMUP_RUNS):
            session.run(None, {input_name: input_tensor})
        timings = []
        for _ in range(LATENCY_TIMED_RUNS):
            start_time = time.perf_counter()
            session.run(None, {input_name: input_tensor})
            timings.append(time.perf_counter() - start_time)
        latency_by_threads[f"threads={thread_count}_median_ms"] = 1000 * float(np.median(timings))
    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    latency_by_threads["input"] = [(i.name, i.shape) for i in session.get_inputs()]
    latency_by_threads["outputs"] = [(o.name, o.shape) for o in session.get_outputs()]
    return latency_by_threads


def main() -> None:
    """Export, verify, time, and write a JSON summary."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--parity-scenes", type=int, default=20)
    parser.add_argument("--name", default="yolo26n-obb", help="artifact name prefix")
    arguments = parser.parse_args()
    onnx_path = export_checkpoint_to_onnx(arguments.weights, arguments.image_size, arguments.name)
    summary = {
        "onnx_path": str(onnx_path),
        "onnx_size_megabytes": onnx_path.stat().st_size / 1e6,
        "pytorch_checkpoint_size_megabytes": arguments.weights.stat().st_size / 1e6,
        "raw_output_parity": measure_raw_output_parity(
            arguments.weights, onnx_path, arguments.image_size, arguments.parity_scenes
        ),
        "end_to_end_parity": measure_end_to_end_parity(arguments.weights, onnx_path, arguments.image_size, arguments.parity_scenes),
        "cpu_latency": measure_cpu_latency(onnx_path, arguments.image_size),
    }
    summary_path = EXPORT_DIRECTORY / f"{arguments.name}__imgsz={arguments.image_size}__summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("%s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
