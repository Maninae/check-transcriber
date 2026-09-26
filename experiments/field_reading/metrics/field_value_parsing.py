"""Turn a field's text (ground truth or prediction) into the value a correctness check compares.

`normalize_field_value(field_name, text)` is THE definition of "this read is correct":
- amount_numeric, amount_words -> integer cents (money equality, so `$453.00` == `453 00/100`)
- date -> ISO date string (so `8.12.26` == `Aug 12, 2026`)
- check_number -> digit string without leading zeros
- payer_name, payee, memo -> normalized text (casefold, collapsed whitespace, edge `*`/punctuation stripped)

Unparseable text gives None, which never equals anything (a None ground truth falls back to the
check's canonical value in scoring). Keep these rules simple: the app mirrors them in JS.
"""

import re
import string

from experiments.field_reading.config import CheckField
from experiments.field_reading.metrics.date_parsing import parse_date_to_iso
from experiments.field_reading.metrics.money_amount_parsing import (
    parse_amount_numeric_to_cents,
    parse_amount_words_to_cents,
)

EDGE_STRIP_CHARACTERS = string.punctuation + string.whitespace
CHECK_NUMBER_PREFIX_PATTERN = re.compile(r"^(?:no\.?|#|nº)\s*")

# Which manifest column holds each field's canonical (annotation-derived) value; text fields have none.
FIELD_NAME_TO_CANONICAL_COLUMN: dict[str, str] = {
    CheckField.AMOUNT_NUMERIC.value: "canonical_amount_cents",
    CheckField.AMOUNT_WORDS.value: "canonical_amount_cents",
    CheckField.DATE.value: "canonical_date_iso",
    CheckField.CHECK_NUMBER.value: "canonical_check_number",
}


def normalize_free_text(text: str) -> str:
    """Casefold, collapse whitespace runs to one space, strip `*`, punctuation and spaces at both ends."""
    return " ".join(text.casefold().split()).strip(EDGE_STRIP_CHARACTERS)


def parse_check_number_digits(check_number_text: str) -> str | None:
    """`0012345`, `No. 2921`, `# 29 21` -> `12345` / `2921`; any other non-digit character -> None."""
    compact_text = CHECK_NUMBER_PREFIX_PATTERN.sub("", check_number_text.strip().lower()).replace(" ", "")
    if not compact_text.isdigit() or not compact_text.isascii():
        return None
    return compact_text.lstrip("0") or "0"


def normalize_field_value(field_name: str, text: str) -> int | str | None:
    """The comparable value of `text` read from field `field_name` (see module docstring)."""
    match field_name:
        case CheckField.AMOUNT_NUMERIC.value:
            return parse_amount_numeric_to_cents(text)
        case CheckField.AMOUNT_WORDS.value:
            return parse_amount_words_to_cents(text)
        case CheckField.DATE.value:
            return parse_date_to_iso(text)
        case CheckField.CHECK_NUMBER.value:
            return parse_check_number_digits(text)
        case _:
            normalized_text = normalize_free_text(text)
            return normalized_text or None


def canonical_value_for_field(field_name: str, canonical_raw_value: object) -> int | str | None:
    """A manifest canonical value in the same form `normalize_field_value` produces (None if absent)."""
    if canonical_raw_value is None or (isinstance(canonical_raw_value, float) and canonical_raw_value != canonical_raw_value):
        return None
    if field_name in (CheckField.AMOUNT_NUMERIC.value, CheckField.AMOUNT_WORDS.value):
        return int(canonical_raw_value)
    if field_name == CheckField.CHECK_NUMBER.value:
        return str(canonical_raw_value).lstrip("0") or "0"
    return str(canonical_raw_value)
