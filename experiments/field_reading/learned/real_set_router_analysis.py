"""Do the routers pick the right reader on REAL handwriting? (SSBI and ORAND-CAR, CPU only.)

Inputs are the per-row outputs of real_ssbi_scoring / real_car_scoring (each with the base readers
TrOCR-hw-grey and the CRNN) plus style-classifier probabilities. Every real crop is handwritten,
so the oracle-hw route = TrOCR-hw alone. Routers composed per row:
- maxconf_raw: higher raw confidence.
- maxconf_rank: each reader's confidence replaced by its percentile among that reader's val
  (synth, oracle-crop) predictions of the same field, then the higher one (a common scale).
- fieldroute: TrOCR-hw on payee/amount/amount_words/date/memo, CRNN elsewhere (all real fields route to TrOCR).
- styleroute: TrOCR-hw when the style classifier says p_handwritten > 0.5, else the CRNN.
Reported per set x field: exact, share of rows answered by TrOCR, and accuracy on the 50% most
confident reads (the router's chosen confidence), a crude real-data gating check.

Run: python -m experiments.field_reading.learned.real_set_router_analysis \
        --ssbi-csv <per-row csv> --car-csv <per-row csv> --crnn-method crnn_amount_route
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.field_reading.config import PREDICTIONS_ROOT
from experiments.field_reading.learned.handwriting_style_classifier import STYLE_OUTPUT_ROOT

logger = logging.getLogger(__name__)

TROCR_METHOD = "trocr_small_handwritten_zs_grey"
FIELD_ROUTE_TO_TROCR = {"payee", "amount_numeric", "amount_words", "date", "memo"}
VAL_PREDICTION_FILES = {"trocr": PREDICTIONS_ROOT / "val" / f"{TROCR_METHOD}__loc=oracle.jsonl",
                        "crnn": PREDICTIONS_ROOT / "val" / "crnn_general__loc=oracle.jsonl"}
TOP_CONFIDENCE_FRACTION = 0.5


def val_confidence_percentile(reader: str, field_name: str, confidences: np.ndarray, val_tables: dict) -> np.ndarray:
    """Percentile of each confidence within the reader's val confidences for that field."""
    reference = np.sort(val_tables[reader].loc[val_tables[reader].field_name == field_name, "confidence"].to_numpy(float))
    return np.searchsorted(reference, confidences, side="right") / max(1, len(reference))


def top_confidence_accuracy(exact: np.ndarray, confidence: np.ndarray) -> float:
    """Accuracy over the most confident TOP_CONFIDENCE_FRACTION of rows."""
    keep = np.argsort(-confidence, kind="stable")[:max(1, int(len(exact) * TOP_CONFIDENCE_FRACTION))]
    return float(exact[keep].mean())


def analyse_set(per_row: pd.DataFrame, crnn_method: str, style: pd.Series, val_tables: dict, set_name: str) -> pd.DataFrame:
    """One summary row per (field, method) for a real set."""
    trocr = per_row[per_row.method == TROCR_METHOD].set_index("row_key")
    crnn = per_row[per_row.method == crnn_method].set_index("row_key").loc[trocr.index]
    records = []
    for field_name, trocr_rows in trocr.groupby("field"):
        crnn_rows = crnn.loc[trocr_rows.index]
        rank_trocr = val_confidence_percentile("trocr", field_name, trocr_rows.confidence.to_numpy(float), val_tables)
        rank_crnn = val_confidence_percentile("crnn", field_name, crnn_rows.confidence.to_numpy(float), val_tables)
        choices = {
            "trocr_hwgrey (= oraclehw route)": np.ones(len(trocr_rows), bool),
            crnn_method: np.zeros(len(trocr_rows), bool),
            "maxconf_raw": trocr_rows.confidence.to_numpy(float) >= crnn_rows.confidence.to_numpy(float),
            "maxconf_rank": rank_trocr >= rank_crnn,
            "fieldroute": np.full(len(trocr_rows), field_name in FIELD_ROUTE_TO_TROCR),
            "styleroute": style.reindex(trocr_rows.index).fillna(0).to_numpy(float) > 0.5,
        }
        for label, use_trocr in choices.items():
            exact = np.where(use_trocr, trocr_rows.exact.to_numpy(bool), crnn_rows.exact.to_numpy(bool))
            confidence = np.where(use_trocr, trocr_rows.confidence.to_numpy(float), crnn_rows.confidence.to_numpy(float))
            records.append({"set": set_name, "field": field_name, "method": label, "n": len(exact), "exact": exact.mean(),
                            "share_trocr": use_trocr.mean(), "acc_top50%conf": top_confidence_accuracy(exact, confidence)})
    return pd.DataFrame.from_records(records)


def main() -> None:
    """Print the router table for both real sets."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ssbi-csv", type=Path, required=True)
    parser.add_argument("--car-csv", type=Path, required=True)
    parser.add_argument("--crnn-method", default="crnn_general")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    val_tables = {reader: pd.read_json(path, lines=True)[["field_name", "confidence"]] for reader, path in VAL_PREDICTION_FILES.items()}
    tables = []
    for set_name, csv_path, style_name in [("SSBI", arguments.ssbi_csv, "real_SSBI"), ("ORAND-CAR", arguments.car_csv, "real_ORAND-CAR")]:
        style_path = STYLE_OUTPUT_ROOT / f"{style_name}.jsonl"
        style = pd.read_json(style_path, lines=True).set_index("row_key").p_handwritten if style_path.exists() else pd.Series(dtype=float)
        tables.append(analyse_set(pd.read_csv(csv_path), arguments.crnn_method, style, val_tables, set_name))
    table = pd.concat(tables).round(3)
    pd.set_option("display.width", 200)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
