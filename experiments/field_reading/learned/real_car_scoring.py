"""Real courtesy-amount check: score amount readers on ORAND-CAR-2014 test crops (eval only).

CC BY-NC-ND 4.0, never trained on; crops stay on vega. Rules from the lead:
- a fixed random sample of SAMPLE_PER_SUBSET crops per subset (CAR-A Uruguay, CAR-B Chile), seed 0;
- ground truth is digits only (thousands dots dropped); exact = the prediction with every
  non-digit stripped equals it; CER on the digit strings; CAR-A and CAR-B reported separately;
- crops are tight, so they get the app's field-crop margin by edge replication (as SSBI).
Also reports `maxconf(<a>,<b>)` rows composed from two readers' outputs (no extra reading).

Run: python -m experiments.field_reading.learned.real_car_scoring --methods crnn_general trocr_small_handwritten_zs_grey
"""

import argparse
import contextlib
import logging
import re

import pandas as pd
import torch
from rapidfuzz.distance import Levenshtein

from experiments.field_reading.config import REAL_SAMPLES_ROOT, REPORTS_ROOT
from experiments.field_reading.learned.line_crop_dataset import read_rgb_image
from experiments.field_reading.learned.mps_lock import hold_mps_lock
from experiments.field_reading.learned.reading_methods import READING_METHOD_REGISTRY, ReaderPool
from experiments.field_reading.learned.real_ssbi_scoring import pad_tight_crop

logger = logging.getLogger(__name__)

CAR_ROOT = REAL_SAMPLES_ROOT / "orand-car-2014" / "ORAND-CAR-2014"
SAMPLE_PER_SUBSET = 500
SAMPLE_SEED = 0
NON_DIGIT = re.compile(r"\D")


def load_car_rows() -> pd.DataFrame:
    """Sampled test rows of both subsets with crop paths and digit ground truth."""
    frames = []
    for subset in ("A", "B"):
        directory = CAR_ROOT / f"CAR-{subset}"
        truth = pd.read_csv(directory / f"{subset.lower()}_test_gt.txt", sep="\t", header=None, names=["file", "text"], dtype=str)
        truth = truth.sample(min(SAMPLE_PER_SUBSET, len(truth)), random_state=SAMPLE_SEED)
        truth["subset"] = f"CAR-{subset}"
        truth["crop_path"] = truth.file.map(lambda name: str(directory / f"{subset.lower()}_test_images" / name))
        frames.append(truth)
    rows = pd.concat(frames, ignore_index=True)
    rows["field_name"], rows["handwritten"] = "amount_numeric", True
    rows["row_key"] = rows.subset + "_" + rows.file
    return rows


def score_reads(rows: pd.DataFrame, texts: list[str], confidences: list[float], method_label: str) -> pd.DataFrame:
    """Per-row digits-only exact match and CER."""
    predicted_digits = [NON_DIGIT.sub("", text) for text in texts]
    return pd.DataFrame({"method": method_label, "row_key": rows.row_key.values, "field": "amount_numeric", "subset": rows.subset.values, "truth": rows.text.values, "pred": texts,
                         "confidence": confidences, "exact": [p == t for p, t in zip(predicted_digits, rows.text)],
                         "cer": [Levenshtein.distance(p, t) / max(1, len(t)) for p, t in zip(predicted_digits, rows.text)]})


def main() -> None:
    """Read the sample with each method, add maxconf pairs, print and write the table."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", nargs="+", required=True, choices=sorted(READING_METHOD_REGISTRY))
    parser.add_argument("--maxconf-pairs", nargs="*", default=[], help="pairs as <method_a>+<method_b>")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu",
                        help="torch readers' device; cpu skips the MPS lock (ONNX readers are always CPU)")
    parser.add_argument("--output-name", default="real_car__per_row.csv")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rows = load_car_rows()
    crops = [pad_tight_crop(read_rgb_image(path)) for path in rows.crop_path]
    outputs = {}
    lock = hold_mps_lock("real CAR scoring") if arguments.device == "mps" else contextlib.nullcontext()
    with lock:
        pool = ReaderPool(arguments.device)
        for method_id in arguments.methods:
            results = READING_METHOD_REGISTRY[method_id](rows, crops, pool)
            outputs[method_id] = ([text for text, _, _ in results], [confidence for _, confidence, _ in results])
    scored = [score_reads(rows, texts, confidences, method_id) for method_id, (texts, confidences) in outputs.items()]
    for pair in arguments.maxconf_pairs:
        first, second = pair.split("+")
        picks = [a if ca >= cb else b for a, b, ca, cb in zip(outputs[first][0], outputs[second][0], outputs[first][1], outputs[second][1])]
        scored.append(score_reads(rows, picks, [max(a, b) for a, b in zip(outputs[first][1], outputs[second][1])], f"maxconf({pair})"))
    scored_rows = pd.concat(scored, ignore_index=True)
    table = scored_rows.groupby(["method", "subset"]).agg(n=("exact", "size"), exact=("exact", "mean"), cer=("cer", "mean")).round(3)
    print("real courtesy amounts (ORAND-CAR-2014 test sample)")
    print(table.unstack("subset").to_string())
    output_path = REPORTS_ROOT / "learned" / arguments.output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored_rows.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
