"""Compose routed methods offline from base readers' prediction files (same split, same localization).

Every base reader is run once per split with `predict`; routers are row-wise selections:
- `route_oraclehw`: GT handwritten rows from --handwritten-file, others from --printed-file
  (ORACLE: the app has no handwritten flag; the output method id must end in `_oraclehw`).
- `route_field`: rows of --fields from --field-file, the rest from --default-file.
- `maxconf`: per row the higher-confidence answer of the --inputs files.
- `cascade`: --primary answer unless its confidence < --threshold, then --fallback.
latency_ms is the sum over readers that had to run for the row (both for maxconf, primary +
fallback when the cascade falls through).

Run: python -m experiments.field_reading.learned.prediction_combination --split val --combine maxconf \
        --inputs a.jsonl b.jsonl --method-id trocr_ft_crnn_maxconf
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.field_reading.data_access.field_manifest import load_field_rows
from experiments.field_reading.learned.prediction_rows import PREDICTION_ROW_KEYS, ReadingPredictionRow, write_prediction_rows

logger = logging.getLogger(__name__)


def load_predictions(path: Path) -> pd.DataFrame:
    """Prediction file indexed by row_key."""
    return pd.read_json(path, lines=True).set_index("row_key", drop=False)


def combine(frames: list[pd.DataFrame], pick_index: np.ndarray, latency_ms: np.ndarray, method_id: str) -> list[ReadingPredictionRow]:
    """Rows taken from frames[pick_index[i]] at row i, relabelled with `method_id`."""
    rows = []
    for position, row_key in enumerate(frames[0].index):
        source = frames[pick_index[position]].loc[row_key]
        record = {key: source[key] for key in PREDICTION_ROW_KEYS}
        record.update(method=method_id, latency_ms=float(latency_ms[position]), check_index=int(record["check_index"]),
                      pred_text=record["pred_text"] or "", confidence=None if pd.isna(record["confidence"]) else float(record["confidence"]),
                      pred_box=record["pred_box"] if isinstance(record["pred_box"], list) else None)
        rows.append(ReadingPredictionRow(**record))
    return rows


def main() -> None:
    """Build one combined prediction file."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", required=True)
    parser.add_argument("--combine", required=True, choices=["route_oraclehw", "route_field", "maxconf", "cascade"])
    parser.add_argument("--method-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", nargs="*", type=Path, default=[])
    parser.add_argument("--handwritten-file", type=Path)
    parser.add_argument("--printed-file", type=Path)
    parser.add_argument("--field-file", type=Path)
    parser.add_argument("--default-file", type=Path)
    parser.add_argument("--fields", nargs="*", default=[])
    parser.add_argument("--primary", type=Path)
    parser.add_argument("--fallback", type=Path)
    parser.add_argument("--threshold", type=float, default=0.9)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if arguments.combine == "route_oraclehw" and not arguments.method_id.endswith("_oraclehw"):
        raise ValueError("routes on the ground-truth handwritten flag must be labelled *_oraclehw")
    paths = {"route_oraclehw": [arguments.handwritten_file, arguments.printed_file],
             "route_field": [arguments.field_file, arguments.default_file],
             "maxconf": arguments.inputs, "cascade": [arguments.primary, arguments.fallback]}[arguments.combine]
    frames = [load_predictions(path) for path in paths]
    common_keys = frames[0].index
    for frame in frames[1:]:
        common_keys = common_keys.intersection(frame.index)
    frames = [frame.loc[common_keys] for frame in frames]
    confidences = np.stack([frame.confidence.fillna(0).to_numpy(float) for frame in frames])
    latencies = np.stack([frame.latency_ms.fillna(0).to_numpy(float) for frame in frames])
    if arguments.combine == "route_oraclehw":
        handwritten = load_field_rows(arguments.split).set_index("row_key").handwritten.reindex(common_keys).fillna(False)
        pick = np.where(handwritten.to_numpy(bool), 0, 1)
        latency = latencies[pick, np.arange(len(pick))]
    elif arguments.combine == "route_field":
        pick = np.where(frames[0].field_name.isin(arguments.fields).to_numpy(), 0, 1)
        latency = latencies[pick, np.arange(len(pick))]
    elif arguments.combine == "maxconf":
        pick = confidences.argmax(axis=0)
        latency = latencies.sum(axis=0)
    else:
        falls_through = confidences[0] < arguments.threshold
        pick = falls_through.astype(int)
        latency = latencies[0] + np.where(falls_through, latencies[1], 0.0)
    write_prediction_rows(combine(frames, pick, latency, arguments.method_id), arguments.output)
    logger.info("%s: %d rows, picks per source %s", arguments.method_id, len(pick), np.bincount(pick, minlength=len(frames)).tolist())


if __name__ == "__main__":
    main()
