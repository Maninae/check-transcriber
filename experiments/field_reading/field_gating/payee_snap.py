"""Snap a payee read to a known payee list and score identification (spec 4.3: payee is a sanity check).

In the app the list is BACLT's configured co-ops; here each split's own held-out payee pool stands
in for it (synth v1 manifest `split_pools.<split>.payee_names`, 4 names, so chance is 25%). Reads
use abbreviated variants ("Blue Heron CLT"), so matching is rapidfuzz WRatio against each canonical
name; below SNAP_MIN_SCORE the payee stays blank.

Run: python -m experiments.field_reading.field_gating.payee_snap --split eval --methods crnn_general__loc=oracle ...
"""

import argparse
import json
import logging

import pandas as pd
from rapidfuzz import fuzz, process

from experiments.field_reading.config import PREDICTIONS_ROOT, REPORTS_ROOT, SYNTH_V1_ROOT
from experiments.field_reading.data_access.field_manifest import load_field_rows

logger = logging.getLogger(__name__)

SNAP_MIN_SCORE = 60


def known_payee_names(split_name: str) -> list[str]:
    """The split's payee pool, standing in for the app's configured list."""
    return json.loads((SYNTH_V1_ROOT / "manifest.json").read_text())["split_pools"][split_name]["payee_names"]


def snap_payee(read_text: str, known_names: list[str]) -> tuple[str, float]:
    """(canonical name or "", match score in [0, 1])."""
    if not read_text:
        return "", 0.0
    name, score, _ = process.extractOne(read_text, known_names, scorer=fuzz.WRatio)
    return (name, score / 100) if score >= SNAP_MIN_SCORE else ("", score / 100)


def main() -> None:
    """Payee identification rate (ok rows, all and handwritten) per method, before and after snapping."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", default="eval")
    parser.add_argument("--methods", nargs="+", required=True)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    truth = load_field_rows(arguments.split)
    truth = truth[(truth.field_name == "payee") & (truth.status == "ok")].set_index("row_key")
    names = known_payee_names(arguments.split)
    records = []
    for method in arguments.methods:
        predictions = pd.read_json(PREDICTIONS_ROOT / arguments.split / f"{method}.jsonl", lines=True).set_index("row_key")
        joined = truth.join(predictions[["pred_text"]], how="left")
        snapped = [snap_payee(text if isinstance(text, str) else "", names) for text in joined.pred_text]
        joined["identified"] = [name == canonical for (name, _), canonical in zip(snapped, joined.canonical_payee)]
        joined["filled"] = [bool(name) for name, _ in snapped]
        for label, rows in (("all", joined), ("handwritten", joined[joined.handwritten]), ("printed", joined[~joined.handwritten])):
            records.append({"method": method, "rows": label, "n": len(rows), "filled%": round(100 * rows.filled.mean(), 1),
                            "identified%": round(100 * rows.identified.mean(), 1),
                            "precision%": round(100 * rows.identified[rows.filled].mean(), 1) if rows.filled.any() else None})
    table = pd.DataFrame(records)
    output_path = REPORTS_ROOT / "final" / f"payee_snap__split={arguments.split}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(f"# Payee identification after snapping to the known list ({len(names)} names; chance 25%)\n\n" + table.to_markdown(index=False) + "\n")
    logger.info("\n%s", table.to_string(index=False))


if __name__ == "__main__":
    main()
