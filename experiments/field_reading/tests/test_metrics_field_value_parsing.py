"""Unit tests for the field-value parsers: every GT format seen in train/val/eval, plus rejections."""

import pytest

from experiments.field_reading.metrics.field_value_parsing import normalize_field_value, normalize_free_text


@pytest.mark.parametrize("courtesy_text, expected_cents", [
    ("$***453.00", 45300),
    ("2,475 00/100", 247500),
    ("694 xx/100", 69400),
    ("2,489.xx", 248900),
    ("1575 00/100", 157500),
    ("**827.00**", 82700),
    ("2,489.-", 248900),
    ("1234 =", 123400),
    ("$1,245.76", 124576),
    ("2,200", 220000),
    ("681", 68100),
    ("$*****1,234.50", 123450),
    ("*****2,350.00", 235000),
    ("951.xx", 95100),
    ("12 no/100", 1200),
    ("1,622 00/100", 162200),
])
def test_amount_numeric_parses_every_gt_format(courtesy_text: str, expected_cents: int) -> None:
    """Each courtesy-box convention in the dataset maps to its integer cents."""
    assert normalize_field_value("amount_numeric", courtesy_text) == expected_cents


@pytest.mark.parametrize("unparseable_text", ["12.5", "1 2 3", "abc", "$", "45O.00", "1.2.3"])
def test_amount_numeric_rejects_malformed_text(unparseable_text: str) -> None:
    """Malformed reads give None (scored wrong), never a partial number."""
    assert normalize_field_value("amount_numeric", unparseable_text) is None


@pytest.mark.parametrize("legal_line_text, expected_cents", [
    ("***FOUR HUNDRED FIFTY-THREE AND 00/100***", 45300),
    ("Two thousand four hundred eighty-nine and no/100", 248900),
    ("Six hundred ninety-four & xx/100", 69400),
    ("SEVEN HUNDRED FORTY-ONE DOLLARS AND 00 CENTS", 74100),
    ("Eight Hundred Twenty-Two Dollars 77 Cents", 82277),
    ("***TWO THOUSAND TWENTY-SEVEN and 38/100***", 202738),
    ("One thousand five hundred and xx/100", 150000),
    ("Two thousand & xx/100", 200000),
    ("one thousand three hundred one and 00/100", 130100),
    ("Three thousand seventy-five and no/100", 307500),
    ("fifteen hundred and 00/100", 150000),
    ("Nine hundred twelve", 91200),
])
def test_amount_words_parses_every_gt_format(legal_line_text: str, expected_cents: int) -> None:
    """Number words (hyphenated tens, scales, `&`, all fraction styles) map to integer cents."""
    assert normalize_field_value("amount_words", legal_line_text) == expected_cents


@pytest.mark.parametrize("unparseable_text", [
    "five twenty and 00/100",
    "twenty twenty and 00/100",
    "hundred and 00/100",
    "four hundered and 00/100",
    "one thousand and 00/100 and 12/100",
    "",
])
def test_amount_words_rejects_ill_formed_numbers(unparseable_text: str) -> None:
    """Misspellings and out-of-order number words are None, not a guessed sum."""
    assert normalize_field_value("amount_words", unparseable_text) is None


@pytest.mark.parametrize("date_text, expected_iso", [
    ("May 19, 2026", "2026-05-19"),
    ("Aug 3rd 2026", "2026-08-03"),
    ("02/17/2025", "2025-02-17"),
    ("6/28/25", "2025-06-28"),
    ("8.12.26", "2026-08-12"),
    ("2025-06-29", "2025-06-29"),
    ("Aug. 3, 2026", "2026-08-03"),
    ("JUN 02 2027", "2027-06-02"),
    ("09-04-2026", "2026-09-04"),
    ("February 10th, 2027", "2027-02-10"),
    ("Nov 21st 2026", "2026-11-21"),
    ("Sept 5 2026", "2026-09-05"),
    ("3 August 2026", "2026-08-03"),
])
def test_date_parses_every_gt_format(date_text: str, expected_iso: str) -> None:
    """All-numeric dates are US month-first, two-digit years are 20YY, month names any case/abbreviation."""
    assert normalize_field_value("date", date_text) == expected_iso


@pytest.mark.parametrize("unparseable_text", ["2/30/2026", "13/01/2026", "Foo 3, 2026", "2026", ""])
def test_date_rejects_impossible_or_unknown_dates(unparseable_text: str) -> None:
    """Impossible calendar dates and non-dates are None."""
    assert normalize_field_value("date", unparseable_text) is None


@pytest.mark.parametrize("check_number_text, expected_digits", [
    ("2921", "2921"),
    ("0012345", "12345"),
    ("No. 2921", "2921"),
    ("# 29 21", "2921"),
    ("41165001200", "41165001200"),
])
def test_check_number_strips_prefix_spaces_and_leading_zeros(check_number_text: str, expected_digits: str) -> None:
    """Check numbers compare as digit strings without leading zeros."""
    assert normalize_field_value("check_number", check_number_text) == expected_digits


def test_check_number_rejects_letters() -> None:
    """An O-for-0 confusion is a wrong read, not silently fixed."""
    assert normalize_field_value("check_number", "29O1") is None


def test_free_text_normalization_casefolds_collapses_and_strips_edges() -> None:
    """Text fields ignore case, whitespace runs and edge punctuation, but keep interior punctuation."""
    assert normalize_free_text("  ***TAYLOR,  Brewer AND Carney. ") == "taylor, brewer and carney"
    assert normalize_field_value("payee", "Greenwillow Co-op") == normalize_field_value("payee", "GREENWILLOW  CO-OP.")
    assert normalize_field_value("memo", "Co-op") != normalize_field_value("memo", "Coop")
    assert normalize_field_value("memo", " *** ") is None
