"""Field-aware cleanup of raw OCR text: strip border/rule artifacts, keep the field-shaped token.

Tesseract reads the courtesy/date box edges as `[`, `|`, `]` and turns check-protector
asterisks into `+`, `#`, `%`. These rules are simple regexes the app can mirror in JS:
- every field: drop leading/trailing border-like characters (`[]|{}()"'_~` and stray punctuation).
- amount_numeric: keep the last money-shaped token (`1,234.56`, `1234 56/100`, `1,234.xx`, `1234.-`),
  dropping any protector garbage in front of it.
- check_number: digits only.
- date: unchanged beyond the edge strip (the date parser tolerates the formats).
"""

import re

EDGE_ARTIFACT_CHARACTERS = "[]|{}()\"'`_~‘’“”«»;:!^"
MONEY_TOKEN_PATTERN = re.compile(r"\d[\d,]*(?:[.,]\s?(?:\d{2}|xx|XX|-{1,2})|\s+(?:\d{2}|xx|XX|no)\s?/\s?100|\s?=)?")


def strip_edge_artifacts(text: str) -> str:
    """Remove border-like characters from both ends, then collapse whitespace."""
    return " ".join(text.strip().strip(EDGE_ARTIFACT_CHARACTERS + " ").split())


def clean_amount_numeric(text: str) -> str:
    """The last money-shaped token in the text, or the edge-stripped text if none is found."""
    tokens = [match.group(0).strip() for match in MONEY_TOKEN_PATTERN.finditer(text)]
    return tokens[-1] if tokens else strip_edge_artifacts(text)


def clean_field_text(field_name: str, raw_text: str) -> str:
    """Apply the cleanup rules for one field."""
    text = strip_edge_artifacts(raw_text)
    if field_name == "amount_numeric":
        return clean_amount_numeric(text)
    if field_name == "check_number":
        return re.sub(r"\D", "", text)
    return text
