"""Check exported ONNX CRNNs against PyTorch and time them on CPU with onnxruntime.

Shared helpers (val crop sample, CPU sessions, median latency) are reused by trocr_onnx_benchmark.
For each model: identical decoded text on EQUIVALENCE_CROP_COUNT val crops (fp32 and int8),
max abs difference of the outputs (CRNN log-probs; TrOCR encoder states + first-step logits),
file size in MB, and batch-1 CPU latency per crop (median over LATENCY_CROP_COUNT crops,
preprocessing excluded) with 1 intra-op thread and with onnxruntime's default thread count.

Run: python -m experiments.field_reading.learned.onnx_benchmark --crnn crnn_general_h32 crnn_amount_h32
"""

import argparse
import json
import logging
import time

import numpy as np
import onnxruntime
import pandas as pd

from experiments.field_reading.config import REPORTS_ROOT
from experiments.field_reading.data_access.field_manifest import load_field_rows
from experiments.field_reading.learned.crnn_model import timesteps_for_width
from experiments.field_reading.learned.crnn_reader import CrnnCropReader
from experiments.field_reading.learned.ctc_decoding import decode_ctc_batch
from experiments.field_reading.learned.line_crop_dataset import read_rgb_image
from experiments.field_reading.learned.onnx_export import ONNX_ROOT
from experiments.field_reading.learned.reading_methods import RECOGNIZER_ROOT

logger = logging.getLogger(__name__)

EQUIVALENCE_CROP_COUNT = 20
LATENCY_CROP_COUNT = 50
SAMPLE_SEED = 5


def sample_val_crops(count: int, field_names: list[str] | None) -> tuple[pd.DataFrame, list[np.ndarray]]:
    """A fixed sample of status-ok val rows and their RGB field crops."""
    rows = load_field_rows("val")
    rows = rows[rows.status == "ok"]
    if field_names:
        rows = rows[rows.field_name.isin(field_names)]
    rows = rows.sample(count, random_state=SAMPLE_SEED)
    return rows, [read_rgb_image(path) for path in rows.field_crop_path]


def ort_session(path, threads: int | None) -> onnxruntime.InferenceSession:
    """CPU session; threads=None keeps onnxruntime's default."""
    options = onnxruntime.SessionOptions()
    if threads is not None:
        options.intra_op_num_threads = threads
    return onnxruntime.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def merge_into_benchmark_report(reports: list[dict]) -> None:
    """Add/replace entries (keyed by model) in reports/learned/onnx_benchmark.json."""
    output_path = REPORTS_ROOT / "learned" / "onnx_benchmark.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(output_path.read_text()) if output_path.exists() else {}
    existing.update({report["model"]: report for report in reports})
    output_path.write_text(json.dumps(existing, indent=1))


def median_latency_ms(run_once, inputs: list) -> float:
    """Median wall time of `run_once(item)` over inputs, after one warm-up call."""
    run_once(inputs[0])
    timings = []
    for item in inputs:
        started = time.perf_counter()
        run_once(item)
        timings.append((time.perf_counter() - started) * 1000)
    return float(np.median(timings))


def benchmark_crnn(run_name: str, field_names: list[str] | None) -> dict:
    """Equivalence, size and latency for one CRNN export."""
    reader = CrnnCropReader(RECOGNIZER_ROOT / run_name / "best.pt", "cpu")
    _, crops = sample_val_crops(LATENCY_CROP_COUNT, field_names)
    lines = [reader.preprocess(crop)[None, None] for crop in crops]
    report = {"model": run_name}
    for precision, path in [("fp32", ONNX_ROOT / f"{run_name}.onnx"), ("int8", ONNX_ROOT / f"{run_name}.int8.onnx")]:
        session = ort_session(path, None)
        identical, max_difference = 0, 0.0
        for line in lines[:EQUIVALENCE_CROP_COUNT]:
            torch_output, _ = reader.log_probabilities_for_lines([line[0, 0]])
            onnx_output = session.run(None, {"line_images": line})[0]
            steps = [timesteps_for_width(line.shape[-1])]
            identical += decode_ctc_batch(torch_output, steps, reader.charset)[0][0] == decode_ctc_batch(onnx_output, steps, reader.charset)[0][0]
            max_difference = max(max_difference, float(np.abs(torch_output - onnx_output).max()))
        report[precision] = {"size_mb": round(path.stat().st_size / 1e6, 2), "identical_text": f"{identical}/{EQUIVALENCE_CROP_COUNT}",
                             "max_abs_diff_log_probs": round(max_difference, 6)}
        for threads, label in [(1, "latency_ms_1thread"), (None, "latency_ms_default_threads")]:
            timed_session = ort_session(path, threads)
            report[precision][label] = round(median_latency_ms(lambda line: timed_session.run(None, {"line_images": line}), lines), 2)
    return report


def main() -> None:
    """Benchmark the requested exports and write a JSON report."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--crnn", nargs="*", default=[])
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    reports = [benchmark_crnn(run, ["amount_numeric"] if "amount" in run else None) for run in arguments.crnn]
    merge_into_benchmark_report(reports)
    print(json.dumps(reports, indent=1))


if __name__ == "__main__":
    main()
