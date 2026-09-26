"""The app's field gate (app/js/fields/field_gating.js) re-stated in Python on the research parsers.

Used by the regression test to check the browser's gate states against an independent
implementation: money/date/check-number parsing from experiments/field_reading/metrics, the payee
snap with rapidfuzz WRatio (field_gating/payee_snap.py), thresholds read from the JS config file so
the two can never silently disagree on a number.
"""

import datetime
import re
from pathlib import Path

from rapidfuzz import fuzz, process

from experiments.field_reading.metrics.date_parsing import parse_date_to_iso
from experiments.field_reading.metrics.field_value_parsing import normalize_free_text
from experiments.field_reading.metrics.money_amount_parsing import parse_amount_numeric_to_cents, parse_amount_words_to_cents

CONFIG_PATH = Path(__file__).resolve().parents[2] / "js" / "fields" / "field_gating_config.js"
CONFIDENT, UNSURE, BLANK = "confident", "unsure", "blank"


def read_gate_config() -> dict:
    """Thresholds parsed out of field_gating_config.js."""
    source = CONFIG_PATH.read_text()
    thresholds = {name: (float(filled.replace("Infinity", "inf")), float(unsure))
                  for name, filled, unsure in re.findall(r"(\w+): \{ filled: ([\w.]+), unsure: ([\d.]+) \}", source)}
    def number(name: str) -> float:
        expression = re.search(rf"export const {name} = ([^;]+);", source).group(1)
        numerator, _, denominator = expression.partition("/")
        return float(numerator) / float(denominator) if denominator else float(numerator)

    return {"thresholds": thresholds, "handwritten": number("HANDWRITTEN_PROBABILITY_THRESHOLD"),
            "handwriting_unsure": number("HANDWRITING_READ_UNSURE_MIN_CONFIDENCE"), "payee_snap": number("PAYEE_SNAP_MIN_SCORE"),
            "edits_per_character": number("PAYER_SNAP_EDITS_PER_CHARACTER"), "max_edits": number("PAYER_SNAP_MAXIMUM_EDITS"),
            "window_days": number("PLAUSIBLE_DATE_WINDOW_DAYS")}


CONFIG = read_gate_config()


def levenshtein(first: str, second: str) -> int:
    """Unit-cost edit distance."""
    previous = list(range(len(second) + 1))
    for i, a in enumerate(first, 1):
        current = [i]
        for j, b in enumerate(second, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def has_text(read) -> bool:
    return bool(read) and bool(read["text"]) and bool(read["text"].strip())


def unread_handwriting(read) -> bool:
    return bool(read) and read["handwrittenProbability"] > CONFIG["handwritten"] and read["reader"] != "trocr"


def state_from_confidence(read, field_name: str) -> str:
    if not has_text(read):
        return BLANK
    if read["reader"] == "trocr":
        return UNSURE if read["confidence"] >= CONFIG["handwriting_unsure"] else BLANK
    filled, unsure = CONFIG["thresholds"][field_name]
    return CONFIDENT if read["confidence"] >= filled else UNSURE if read["confidence"] >= unsure else BLANK


def cents_text(cents: int) -> str:
    return f"{cents // 100}.{cents % 100:02d}"


def gate(raw_reads: dict, known_payer_names: list[str], known_payee_names: list[str], today_iso: str) -> dict[str, tuple[str, str]]:
    """Field key -> (state, value), same keys and rules as gateCheckFields."""
    out = {}
    read = raw_reads.get("check_number")
    digits = re.sub(r"^(?:no\.?|#|nº)\s*", "", read["text"].strip().lower()).replace(" ", "") if has_text(read) else ""
    if not (digits.isdigit() and digits.isascii()) or unread_handwriting(read):
        out["checkNumber"] = (BLANK, "")
    else:
        state = state_from_confidence(read, "check_number")
        out["checkNumber"] = (state, "" if state == BLANK else digits)

    read = raw_reads.get("payer_name")
    state = BLANK if unread_handwriting(read) else state_from_confidence(read, "payer_name")
    if state == BLANK:
        out["payer"] = (BLANK, "")
    else:
        text = read["text"].strip()
        normalized = normalize_free_text(text)
        allowed = min(CONFIG["max_edits"], max(1, int(len(normalized) * CONFIG["edits_per_character"])))
        candidates = [(levenshtein(normalized, normalize_free_text(name)), index, name) for index, name in enumerate(known_payer_names)]
        candidates = [c for c in candidates if c[0] <= allowed]
        if not candidates:
            out["payer"] = (state, text)
        else:
            distance, _, name = min(candidates)
            out["payer"] = (state, name) if distance == 0 else (UNSURE, name)

    numeric, words = raw_reads.get("amount_numeric"), raw_reads.get("amount_words")
    numeric_cents = parse_amount_numeric_to_cents(numeric["text"]) if has_text(numeric) else None
    words_cents = parse_amount_words_to_cents(words["text"]) if has_text(words) else None
    if numeric_cents is not None and numeric_cents == words_cents:
        out["amount"] = (CONFIDENT, cents_text(numeric_cents))
    elif numeric_cents is not None:
        out["amount"] = (UNSURE, cents_text(numeric_cents))
    elif words_cents is not None:
        out["amount"] = (UNSURE, cents_text(words_cents))
    else:
        out["amount"] = (BLANK, "")

    read = raw_reads.get("date")
    iso = parse_date_to_iso(read["text"]) if has_text(read) and not unread_handwriting(read) else None
    in_window = iso is not None and abs((datetime.date.fromisoformat(iso) - datetime.date.fromisoformat(today_iso)).days) <= CONFIG["window_days"]
    state = state_from_confidence(read, "date") if in_window else BLANK
    out["date"] = (state, "" if state == BLANK else iso)

    read = raw_reads.get("memo")
    state = BLANK if unread_handwriting(read) else state_from_confidence(read, "memo")
    out["memo"] = (state, "" if state == BLANK else read["text"].strip())

    read = raw_reads.get("payee")
    if not has_text(read) or unread_handwriting(read) or not known_payee_names or (read["reader"] == "trocr" and read["confidence"] < CONFIG["handwriting_unsure"]):
        out["payee"] = (BLANK, "")
    else:
        name, score, _ = process.extractOne(read["text"].strip(), known_payee_names, scorer=fuzz.WRatio)
        out["payee"] = (UNSURE, name) if score >= CONFIG["payee_snap"] else (BLANK, "")
    return out
