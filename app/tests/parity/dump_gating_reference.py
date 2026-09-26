"""Dump Python parser / scorer results on real eval strings for run_gating_parity.mjs.

    /Volumes/vega/datasets/check-transcriber/venv/bin/python app/tests/parity/dump_gating_reference.py

Strings: every ground-truth and research-prediction text of the eval split (courtesy amounts,
legal lines, dates, payees). Writes /tmp/check-transcriber-m4/gating_reference.json with the
experiments' parse results and rapidfuzz WRatio scores of payee reads against the eval co-op pool.
"""

import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from rapidfuzz import fuzz  # noqa: E402

from experiments.field_reading.metrics.date_parsing import parse_date_to_iso  # noqa: E402
from experiments.field_reading.metrics.field_value_parsing import normalize_free_text  # noqa: E402
from experiments.field_reading.metrics.money_amount_parsing import parse_amount_numeric_to_cents, parse_amount_words_to_cents  # noqa: E402

PREDICTIONS_DIRECTORY = Path("/Volumes/vega/datasets/check-transcriber/field-reading/predictions/eval")
MANIFEST_PATH = Path("/Volumes/vega/datasets/check-transcriber/synth/v1/ocr/ocr_fields__split=eval.jsonl")
SYNTH_MANIFEST = Path("/Volumes/vega/datasets/check-transcriber/synth/v1/manifest.json")
OUTPUT_PATH = Path("/tmp/check-transcriber-m4/gating_reference.json")
PAYEE_PAIR_LIMIT = 4000


def collect_texts() -> dict[str, set[str]]:
    """field name -> every distinct text seen for it (ground truth + every method's predictions)."""
    texts: dict[str, set[str]] = {}
    for line in MANIFEST_PATH.read_text().splitlines():
        row = json.loads(line)
        texts.setdefault(row["field_name"], set()).add(row["text"] or "")
    for path in PREDICTIONS_DIRECTORY.glob("*.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            texts.setdefault(row["field_name"], set()).add(row["pred_text"] or "")
    return texts


def main() -> None:
    """Write the reference file."""
    texts = collect_texts()
    pool = json.loads(SYNTH_MANIFEST.read_text())["split_pools"]["eval"]["payee_names"]
    payee_reads = sorted(texts["payee"])[:PAYEE_PAIR_LIMIT]
    reference = {
        "courtesy": [[text, parse_amount_numeric_to_cents(text)] for text in sorted(texts["amount_numeric"])],
        "legal": [[text, parse_amount_words_to_cents(text)] for text in sorted(texts["amount_words"])],
        "date": [[text, parse_date_to_iso(text)] for text in sorted(texts["date"])],
        "free_text": [[text, normalize_free_text(text)] for text in sorted(texts["payer_name"])],
        "wratio": [[read, name, fuzz.WRatio(read, name)] for read in payee_reads for name in pool],
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(reference))
    print({key: len(value) for key, value in reference.items()}, "->", OUTPUT_PATH)


if __name__ == "__main__":
    main()
