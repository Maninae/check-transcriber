"""Tests for the recognizer charset, CTC greedy decoding, confidence and constrained decoding."""

import numpy as np
import pytest

from experiments.field_reading.learned.ctc_decoding import (CtcConfidenceKind, decode_ctc_batch, greedy_ctc_labels,
                                                            restrict_log_probabilities_to_charset)
from experiments.field_reading.learned.quick_text_scoring import character_error_rate, is_exact_match
from experiments.field_reading.learned.text_charset import CHARSET_REGISTRY, CTC_BLANK_INDEX, TextCharset

GENERAL = CHARSET_REGISTRY["general"]


@pytest.mark.parametrize("text", ["$***453.00", "2,475 00/100", "Aug 3rd 2026", "Unit #7D & rent", "694 xx/100 ="])
def test_general_charset_round_trip(text):
    assert GENERAL.covers(text)
    assert GENERAL.decode_labels(GENERAL.encode(text)) == text


def test_amount_charset_covers_courtesy_formats_and_skips_letters():
    amount = CHARSET_REGISTRY["amount"]
    for text in ["$***453.00", "2,475 00/100", "694 xx/100", "725.-", "2979 ="]:
        assert amount.decode_labels(amount.encode(text)) == text
    assert amount.encode("Ab") == []


def test_blank_is_index_zero_and_duplicates_rejected():
    assert GENERAL.character_to_label[GENERAL.characters[0]] == 1 and CTC_BLANK_INDEX == 0
    with pytest.raises(ValueError):
        TextCharset("dup", "aab")


def one_hot_log_probabilities(label_path: list[int], num_classes: int, peak: float = 0.9) -> np.ndarray:
    """(T, C) log-probs whose argmax follows `label_path`, with max probability `peak`."""
    probabilities = np.full((len(label_path), num_classes), (1 - peak) / (num_classes - 1))
    probabilities[np.arange(len(label_path)), label_path] = peak
    return np.log(probabilities)


def test_greedy_decode_collapses_repeats_and_keeps_blank_separated_doubles():
    charset = TextCharset("toy", "ab0")
    a, b, zero = charset.encode("ab0")
    path = [0, a, a, 0, b, zero, 0, zero, zero, 0]
    labels, probabilities = greedy_ctc_labels(one_hot_log_probabilities(path, charset.num_classes))
    assert charset.decode_labels(labels) == "ab00"
    assert len(probabilities) == 4 and all(abs(p - 0.9) < 1e-9 for p in probabilities)


def test_confidence_kinds_order_and_empty_decode():
    charset = TextCharset("toy", "ab")
    log_probabilities = one_hot_log_probabilities([1, 0, 2], charset.num_classes, peak=0.9)
    log_probabilities[2] = np.log(np.array([0.2, 0.2, 0.6]))
    (text, confidences), = decode_ctc_batch(log_probabilities[None], [3], charset, CtcConfidenceKind.MIN_CHAR)
    assert text == "ab"
    assert confidences["min_char"] == pytest.approx(0.6)
    assert confidences["mean_char"] == pytest.approx(0.75)
    assert confidences["confidence"] == confidences["min_char"]
    assert 0 < confidences["path"] < 1
    (empty_text, empty_confidences), = decode_ctc_batch(one_hot_log_probabilities([0, 0], 3)[None], [2], charset)
    assert empty_text == "" and empty_confidences["confidence"] == 0.0


def test_valid_timesteps_trim_padding():
    charset = TextCharset("toy", "ab")
    log_probabilities = one_hot_log_probabilities([1, 0, 2, 2], charset.num_classes)
    (text, _), = decode_ctc_batch(log_probabilities[None], [2], charset)
    assert text == "a"


def test_constrained_decoding_masks_disallowed_classes():
    charset = TextCharset("toy", "aO0")
    probabilities = np.array([[0.05, 0.05, 0.6, 0.3]])   # 'O' wins unconstrained, '0' allowed
    restricted = restrict_log_probabilities_to_charset(np.log(probabilities), charset, "0")
    labels, _ = greedy_ctc_labels(restricted)
    assert charset.decode_labels(labels) == "0"
    assert np.exp(restricted).sum() == pytest.approx(1.0)


def test_quick_scoring_normalizes_case_and_whitespace():
    assert is_exact_match("  kettle   POND ", "Kettle Pond")
    assert character_error_rate("abc", "abd") == pytest.approx(1 / 3)
    assert character_error_rate(None, "ab") == 1.0
