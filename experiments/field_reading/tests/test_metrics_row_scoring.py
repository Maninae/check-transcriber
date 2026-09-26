"""Per-row scoring rules: value vs text vs raw correctness, blanks, CER/WER."""

import numpy as np
import pandas as pd
import pytest

from experiments.field_reading.metrics.row_scoring import score_joined_rows
from experiments.field_reading.metrics.text_error_rates import character_error_rate, word_error_rate


def score_one(field_name: str, ground_truth_text: str, predicted_text: str, **canonical_values) -> pd.Series:
    """Score a single synthetic row."""
    row = {"field_name": field_name, "text": ground_truth_text, "pred_text": predicted_text, "confidence": np.nan,
           "canonical_amount_cents": None, "canonical_date_iso": None, "canonical_check_number": None, **canonical_values}
    return score_joined_rows(pd.DataFrame([row])).iloc[0]


def test_money_equality_ignores_format_but_raw_and_text_do_not() -> None:
    """`$453.00` read as `453 00/100` is value-correct, not raw- or text-correct."""
    scored = score_one("amount_numeric", "$453.00", "453 00/100")
    assert scored.field_correct and not scored.exact_raw and not scored.text_correct


def test_amount_words_value_level_separate_from_text_level() -> None:
    """Different spelling of the same amount: value right, text wrong."""
    scored = score_one("amount_words", "***FOUR HUNDRED FIFTY-THREE AND 00/100***", "Four hundred fifty three & no/100")
    assert scored.field_correct and not scored.text_correct


def test_blank_prediction_is_uncovered_and_never_correct() -> None:
    """Blank = not covered, CER 1.0, even for an empty-normalizing GT."""
    scored = score_one("payee", "Greenwillow Co-op", "")
    assert not scored.covered and not scored.field_correct and scored.cer == 1.0 and scored.wer == 1.0


def test_unparseable_prediction_is_wrong_not_a_crash() -> None:
    """Garbage in a money field scores wrong."""
    scored = score_one("amount_numeric", "$453.00", "$4S3.OO")
    assert scored.covered and not scored.field_correct


def test_unparseable_ground_truth_falls_back_to_canonical_value() -> None:
    """If the GT text does not parse, the canonical value decides."""
    scored = score_one("date", "the third of Aug", "08/03/2026", canonical_date_iso="2026-08-03")
    assert scored.field_correct


def test_character_and_word_error_rates() -> None:
    """Levenshtein over characters / words divided by GT length."""
    assert character_error_rate("rent", "rent") == 0.0
    assert character_error_rate("rant", "rent") == pytest.approx(0.25)
    assert character_error_rate("", "") == 0.0
    assert word_error_rate("june rent due", "june rent") == pytest.approx(0.5)
    assert word_error_rate("", "june rent") == 1.0
