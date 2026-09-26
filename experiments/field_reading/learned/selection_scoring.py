"""Model-selection tables for learned readers, built on the shared metrics harness.

Correctness is the harness's `field_correct` (normalize_field_value: cents, ISO dates, digits,
normalized text), CER its normalized CER. Tables are method x field, split by style
(handwritten / printed) on status ok rows, with too_small reported separately, plus the
gating number the app cares about: max coverage at >= 95% accuracy on handwritten ok rows.

Run: python -m experiments.field_reading.learned.selection_scoring --split val --predictions <jsonl> ...
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.field_reading.data_access.field_manifest import load_field_rows
from experiments.field_reading.learned.context_crop_export import SCORED_STATUSES
from experiments.field_reading.metrics.confidence_gating import build_gating_operating_points
from experiments.field_reading.metrics.row_scoring import score_joined_rows

logger = logging.getLogger(__name__)

GATING_ACCURACY_TARGET = 0.95


def score_prediction_file_against_split(prediction_path: Path, ground_truth_rows: pd.DataFrame) -> pd.DataFrame:
    """Scored rows (only rows present in the prediction file) with method, style and status labels."""
    predictions = pd.read_json(prediction_path, lines=True)
    joined = ground_truth_rows.merge(predictions[["row_key", "method", "pred_text", "confidence"]], on="row_key", how="inner")
    joined["pred_text"] = joined.pred_text.fillna("")
    scored = score_joined_rows(joined)
    scored["style"] = np.where(scored.handwritten.fillna(False), "hw", "printed")
    return scored


def max_coverage_at_accuracy(scored_rows: pd.DataFrame, accuracy_target: float = GATING_ACCURACY_TARGET) -> float:
    """Largest fraction of rows that can be filled with accuracy >= target by thresholding confidence."""
    points = build_gating_operating_points(scored_rows.confidence.to_numpy(float), scored_rows.field_correct.to_numpy(bool),
                                           scored_rows.covered.to_numpy(bool))
    reachable = points[points.accuracy >= accuracy_target]
    return float(reachable.coverage.max()) if len(reachable) else 0.0


def build_selection_table(scored_rows: pd.DataFrame) -> pd.DataFrame:
    """method x field x slice: n, accuracy, CER, coverage@95 (slices: ok-hw, ok-printed, too_small)."""
    scored_rows = scored_rows.assign(slice=np.where(scored_rows.status == "ok", "ok_" + scored_rows["style"], "too_small"))
    records = []
    for (method, field_name, slice_name), group in scored_rows.groupby(["method", "field_name", "slice"]):
        records.append({"method": method, "field": field_name, "slice": slice_name, "n": len(group),
                        "acc": group.field_correct.mean(), "cer": group.cer.mean(),
                        "cov@95": max_coverage_at_accuracy(group)})
    return pd.DataFrame.from_records(records)


def pivot_for_display(selection_table: pd.DataFrame, value_column: str) -> pd.DataFrame:
    """Rows = method, columns = (field, slice)."""
    return selection_table.pivot_table(index="method", columns=["field", "slice"], values=value_column).round(3)


def main() -> None:
    """Print accuracy / CER / coverage@95 tables for the given prediction files."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", required=True)
    parser.add_argument("--predictions", nargs="+", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ground_truth_rows = load_field_rows(arguments.split)
    ground_truth_rows = ground_truth_rows[ground_truth_rows.status.isin(SCORED_STATUSES)]
    scored = pd.concat([score_prediction_file_against_split(path, ground_truth_rows) for path in arguments.predictions])
    table = build_selection_table(scored)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    for value_column in ["acc", "cer", "cov@95"]:
        print(f"\n== {value_column} ==")
        print(pivot_for_display(table, value_column).T.to_string())
    if arguments.output_csv:
        table.to_csv(arguments.output_csv, index=False)


if __name__ == "__main__":
    main()
