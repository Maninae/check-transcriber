"""Parse handwritten or printed check dates into ISO `YYYY-MM-DD` strings.

Forms handled (US month-first for all-numeric dates; two-digit years mean 20YY):
- numeric with `/`, `-` or `.` separators: `02/17/2025`, `6/28/25`, `8.12.26`, `09-04-2026`
- ISO year-first: `2025-06-29`
- month name first: `May 19, 2026`, `Aug 3rd 2026`, `Aug. 3, 2026`, `JUN 02 2027`, `Sept 5 2026`
- day first with a month name: `3 August 2026`, `3rd Aug, 2026`

Returns None for anything else, including impossible calendar dates (`2/30/2026`).
"""

import datetime
import re

TWO_DIGIT_YEAR_CENTURY = 2000
MONTH_PREFIX_TO_NUMBER = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
FULL_MONTH_NAMES = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
]

YEAR_FIRST_NUMERIC_PATTERN = re.compile(r"(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})")
MONTH_FIRST_NUMERIC_PATTERN = re.compile(r"(\d{1,2})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{2}|\d{4})")
DAY_WITH_ORDINAL = r"(\d{1,2})(?:st|nd|rd|th)?"
MONTH_WORD = r"([a-z]{3,9})\.?"
MONTH_NAME_FIRST_PATTERN = re.compile(MONTH_WORD + r"\s*" + DAY_WITH_ORDINAL + r"\s*,?\s*(\d{4}|\d{2})")
DAY_FIRST_MONTH_NAME_PATTERN = re.compile(DAY_WITH_ORDINAL + r"\s+" + MONTH_WORD + r"\s*,?\s*(\d{4}|\d{2})")


def month_word_to_number(month_word: str) -> int | None:
    """`aug`, `august`, `sept` -> 8, 8, 9 (any 3+ letter prefix of a month name); anything not a prefix of a real month name -> None."""
    for month_index, full_name in enumerate(FULL_MONTH_NAMES):
        if len(month_word) >= 3 and full_name.startswith(month_word):
            return month_index + 1
    return None


def expand_year(year_text: str) -> int:
    """`26` -> 2026, `2026` -> 2026."""
    year_value = int(year_text)
    return year_value + TWO_DIGIT_YEAR_CENTURY if len(year_text) == 2 else year_value


def iso_date_or_none(year_value: int, month_value: int | None, day_value: int) -> str | None:
    """ISO string for a real calendar date, None for impossible ones."""
    if month_value is None:
        return None
    try:
        return datetime.date(year_value, month_value, day_value).isoformat()
    except ValueError:
        return None


def parse_date_to_iso(date_text: str) -> str | None:
    """Date text as written on a check -> `YYYY-MM-DD`, or None."""
    compact_text = " ".join(date_text.lower().replace("*", " ").split()).strip(" ,")
    if match := YEAR_FIRST_NUMERIC_PATTERN.fullmatch(compact_text):
        return iso_date_or_none(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    if match := MONTH_FIRST_NUMERIC_PATTERN.fullmatch(compact_text):
        return iso_date_or_none(expand_year(match.group(3)), int(match.group(1)), int(match.group(2)))
    if match := MONTH_NAME_FIRST_PATTERN.fullmatch(compact_text):
        month_value = month_word_to_number(match.group(1))
        return iso_date_or_none(expand_year(match.group(3)), month_value, int(match.group(2)))
    if match := DAY_FIRST_MONTH_NAME_PATTERN.fullmatch(compact_text):
        month_value = month_word_to_number(match.group(2))
        return iso_date_or_none(expand_year(match.group(3)), month_value, int(match.group(1)))
    return None
