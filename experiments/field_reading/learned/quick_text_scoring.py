"""Simple string scores for model selection until the shared metrics harness is used.

Exact match and CER both compare casefolded, whitespace-collapsed text. The lead re-scores
everything with `experiments/field_reading/metrics/`; these numbers are for choosing models on val.
"""

import re

from rapidfuzz.distance import Levenshtein

WHITESPACE_RUN = re.compile(r"\s+")


def normalize_text_for_comparison(text: str | None) -> str:
    """Casefold and collapse whitespace runs to single spaces."""
    return WHITESPACE_RUN.sub(" ", (text or "")).strip().casefold()


def is_exact_match(predicted_text: str | None, ground_truth_text: str) -> bool:
    """Normalized string equality."""
    return normalize_text_for_comparison(predicted_text) == normalize_text_for_comparison(ground_truth_text)


def character_error_rate(predicted_text: str | None, ground_truth_text: str) -> float:
    """Levenshtein distance over ground-truth length (normalized strings; empty GT counts as length 1)."""
    predicted = normalize_text_for_comparison(predicted_text)
    truth = normalize_text_for_comparison(ground_truth_text)
    return Levenshtein.distance(predicted, truth) / max(1, len(truth))
