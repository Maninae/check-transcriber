"""Pick the CTC confidence statistic per field: which one gates best on val.

Reads every scored val row (oracle crops) with a CRNN once, keeps all confidence kinds
(mean_char, min_char, path), and reports per field x style the max coverage at >= 95% and 98%
accuracy for each kind (harness `field_correct`). Writes reports/learned/confidence_kinds__<run>.csv.

Run: python -m experiments.field_reading.learned.confidence_kind_selection --run crnn_general_h32
"""

import argparse
import logging

import numpy as np
import pandas as pd
import torch

from experiments.field_reading.config import REPORTS_ROOT
from experiments.field_reading.learned.crnn_reader import CrnnCropReader
from experiments.field_reading.learned.ctc_decoding import CtcConfidenceKind
from experiments.field_reading.learned.line_crop_dataset import read_rgb_image
from experiments.field_reading.learned.mps_lock import hold_mps_lock
from experiments.field_reading.learned.predict import select_scored_rows
from experiments.field_reading.learned.recognizer_paths import RECOGNIZER_ROOT
from experiments.field_reading.learned.selection_scoring import max_coverage_at_accuracy
from experiments.field_reading.metrics.row_scoring import score_joined_rows

logger = logging.getLogger(__name__)

ACCURACY_TARGETS = (0.95, 0.98)


def main() -> None:
    """Read val once, compare confidence kinds, write and print the table."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="crnn_general_h32")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rows = select_scored_rows("val", None)
    crops = [read_rgb_image(path) for path in rows.field_crop_path]
    with hold_mps_lock(f"confidence kinds {arguments.run}"):
        reader = CrnnCropReader(RECOGNIZER_ROOT / arguments.run / "best.pt", "mps" if torch.backends.mps.is_available() else "cpu")
        results = reader.read_crops(crops)
    rows = rows.assign(pred_text=[text for text, _ in results])
    for kind in CtcConfidenceKind:
        rows[f"confidence_{kind.value}"] = [confidences[kind.value] for _, confidences in results]
    scored = score_joined_rows(rows.assign(confidence=np.nan))
    scored["style"] = np.where(scored.handwritten.fillna(False), "hw", "printed")
    scored = scored[scored.status == "ok"]
    records = []
    for (field_name, style), group in scored.groupby(["field_name", "style"]):
        for kind in CtcConfidenceKind:
            for target in ACCURACY_TARGETS:
                records.append({"field": field_name, "style": style, "kind": kind.value, "target": target,
                                "coverage": max_coverage_at_accuracy(group.assign(confidence=group[f"confidence_{kind.value}"]), target)})
    table = pd.DataFrame.from_records(records)
    output_path = REPORTS_ROOT / "learned" / f"confidence_kinds__{arguments.run}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    print(table.pivot_table(index=["field", "style", "target"], columns="kind", values="coverage").round(3).to_string())


if __name__ == "__main__":
    main()
