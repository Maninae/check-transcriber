"""Parse check amounts (courtesy box digits and legal-line words) into integer cents.

Both parsers are deliberately small, explicit regex/token rules so the app can mirror them in JS.
They return None for anything they do not fully understand: an unparseable prediction scores as
wrong, never as a crash and never as a lucky partial match.

Courtesy-box forms handled (after stripping `$`, `*`, commas and spaces at the edges):
- `453.00`, `2,489.xx`, `2,489.-`  (dot cents; `xx` / dashes mean zero cents)
- `2,475 00/100`, `694 xx/100`, `12 no/100`  (fraction cents)
- `1234 =`, `2,200`, `681`  (whole dollars)

Legal-line forms: number words (hyphenated tens, "fifteen hundred"), optional "dollars", "and" / "&",
then a fraction `NN/100`, `xx/100`, `no/100` or `NN cents`. A missing fraction means zero cents.
"""

import re

CENTS_PER_DOLLAR = 100

# Courtesy box: one alternative per written convention, each fully anchored.
DOT_CENTS_PATTERN = re.compile(r"(\d+)\.(\d{2})")
DOT_ZERO_CENTS_PATTERN = re.compile(r"(\d+)\.(?:xx|-+)")
FRACTION_CENTS_PATTERN = re.compile(r"(\d+)\s*(\d{2}|xx|no)\s*/\s*100")
WHOLE_DOLLARS_PATTERN = re.compile(r"(\d+)\s*=*")
COURTESY_EDGE_NOISE_CHARACTERS = "$* \t"

# Legal line: the fraction part is found first and cut out, the rest must be number words.
WORDS_FRACTION_PATTERN = re.compile(r"(\d{1,2}|xx|no)\s*/\s*100")
WORDS_CENTS_PATTERN = re.compile(r"(\d{1,2})\s*cents?\b")
WORDS_FILLER_TOKENS = {"and", "dollars", "dollar", "only", "exactly"}
WORDS_PUNCTUATION_TOKENS = {"*", ".", ",", "/", ":", ";"}
UNIT_WORD_VALUES = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
}
TEEN_WORD_VALUES = {
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
TENS_WORD_VALUES = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
SCALE_WORD_VALUES = {
    "thousand": 1_000,
    "million": 1_000_000,
}


def fraction_token_to_cents(fraction_token: str) -> int:
    """`07` -> 7; `xx`, `no` and dash runs mean zero cents."""
    return int(fraction_token) if fraction_token.isdigit() else 0


def parse_amount_numeric_to_cents(courtesy_text: str) -> int | None:
    """Courtesy-box text (e.g. `$***1,245.76`, `694 xx/100`, `1234 =`) -> integer cents, or None."""
    compact_text = courtesy_text.strip(COURTESY_EDGE_NOISE_CHARACTERS).replace(",", "").lower()
    compact_text = compact_text.replace("*", "").replace("$", "").strip()
    if match := DOT_CENTS_PATTERN.fullmatch(compact_text):
        return int(match.group(1)) * CENTS_PER_DOLLAR + int(match.group(2))
    if match := DOT_ZERO_CENTS_PATTERN.fullmatch(compact_text):
        return int(match.group(1)) * CENTS_PER_DOLLAR
    if match := FRACTION_CENTS_PATTERN.fullmatch(compact_text):
        return int(match.group(1)) * CENTS_PER_DOLLAR + fraction_token_to_cents(match.group(2))
    if match := WHOLE_DOLLARS_PATTERN.fullmatch(compact_text):
        return int(match.group(1)) * CENTS_PER_DOLLAR
    return None


def number_words_to_integer(number_words: list[str]) -> int | None:
    """["two", "thousand", "forty", "one"] -> 2041, or None if the sequence is not a well-formed number.

    Grammar per thousand-group: [unit|teen [hundred]] [tens] [unit] | teen; `fifteen hundred` is allowed.
    The previous-token kind is tracked so "five twenty" or "twenty twenty" are rejected, not summed.
    """
    if not number_words:
        return None
    total_value = 0
    group_value = 0
    previous_kind = None  # kind of the previous word inside the current thousand-group
    for word in number_words:
        if word in UNIT_WORD_VALUES and previous_kind in (None, "tens", "hundred"):
            group_value += UNIT_WORD_VALUES[word]
            previous_kind = "unit"
        elif word in TEEN_WORD_VALUES and previous_kind in (None, "hundred"):
            group_value += TEEN_WORD_VALUES[word]
            previous_kind = "teen"
        elif word in TENS_WORD_VALUES and previous_kind in (None, "hundred"):
            group_value += TENS_WORD_VALUES[word]
            previous_kind = "tens"
        elif word == "hundred" and previous_kind in ("unit", "teen") and group_value < 100:
            group_value *= 100
            previous_kind = "hundred"
        elif word in SCALE_WORD_VALUES and previous_kind is not None:
            total_value += group_value * SCALE_WORD_VALUES[word]
            group_value = 0
            previous_kind = None
        else:
            return None
    return total_value + group_value


def parse_amount_words_to_cents(legal_line_text: str) -> int | None:
    """Legal-line text (e.g. `***FOUR HUNDRED FIFTY-THREE AND 00/100***`) -> integer cents, or None."""
    lowered_text = legal_line_text.lower().replace("&", " and ").replace("-", " ")
    cents_value = 0
    fraction_matches = WORDS_FRACTION_PATTERN.findall(lowered_text) or WORDS_CENTS_PATTERN.findall(lowered_text)
    if len(fraction_matches) > 1:
        return None
    if fraction_matches:
        cents_value = fraction_token_to_cents(fraction_matches[0])
        lowered_text = WORDS_CENTS_PATTERN.sub(" ", WORDS_FRACTION_PATTERN.sub(" ", lowered_text))
    word_tokens = re.findall(r"[a-z]+|\S", lowered_text)
    number_words = [token for token in word_tokens if token not in WORDS_FILLER_TOKENS | WORDS_PUNCTUATION_TOKENS]
    dollar_value = number_words_to_integer(number_words)
    if dollar_value is None:
        return None
    return dollar_value * CENTS_PER_DOLLAR + cents_value
