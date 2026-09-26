"""Greedy CTC decoding and the confidence scores derived from the best path.

The recognizer emits per-timestep log-probabilities (T, C). Greedy decoding takes the argmax at
each step, collapses repeats, and drops blanks. Confidence comes from the same path:

- `mean_char`: mean over emitted characters of the max probability at the step that emitted
  them (the first step of each run). Blank steps are ignored, so long padded crops are not
  rewarded for confidently predicting blanks.
- `min_char`: the weakest emitted character, the strictest gate (one unsure digit = unsure field).
- `path`: exp(mean log-prob over all T steps of the argmax path), blanks included.

An empty decode gets confidence 0.0 (the app treats it as blank anyway).
"""

from enum import Enum

import numpy as np

from experiments.field_reading.learned.text_charset import CTC_BLANK_INDEX, TextCharset


class CtcConfidenceKind(str, Enum):
    """Which best-path statistic becomes the field confidence."""

    MEAN_CHAR = "mean_char"
    MIN_CHAR = "min_char"
    PATH = "path"


def greedy_ctc_labels(log_probabilities: np.ndarray) -> tuple[list[int], list[float]]:
    """Collapsed label sequence and each emitted label's max probability, for one (T, C) array."""
    best_labels = log_probabilities.argmax(axis=1)
    best_log_probabilities = log_probabilities.max(axis=1)
    emitted_labels: list[int] = []
    emitted_probabilities: list[float] = []
    previous_label = CTC_BLANK_INDEX
    for label, log_probability in zip(best_labels.tolist(), best_log_probabilities.tolist()):
        if label != CTC_BLANK_INDEX and label != previous_label:
            emitted_labels.append(label)
            emitted_probabilities.append(float(np.exp(log_probability)))
        previous_label = label
    return emitted_labels, emitted_probabilities


def ctc_confidence(log_probabilities: np.ndarray, emitted_probabilities: list[float],
                   confidence_kind: CtcConfidenceKind) -> float:
    """Confidence in [0, 1] for one decoded line."""
    if not emitted_probabilities:
        return 0.0
    if confidence_kind == CtcConfidenceKind.MEAN_CHAR:
        return float(np.mean(emitted_probabilities))
    if confidence_kind == CtcConfidenceKind.MIN_CHAR:
        return float(np.min(emitted_probabilities))
    return float(np.exp(log_probabilities.max(axis=1).mean()))


def decode_ctc_batch(log_probabilities: np.ndarray, valid_timesteps: list[int], charset: TextCharset,
                     confidence_kind: CtcConfidenceKind = CtcConfidenceKind.MEAN_CHAR
                     ) -> list[tuple[str, dict[str, float]]]:
    """Decode a (B, T, C) batch; returns (text, {confidence kind: value}) per item.

    Every confidence kind is returned so the caller can pick one after the fact (val gating);
    `confidence_kind` only chooses the one stored under the "confidence" key.
    """
    decoded = []
    for item_log_probabilities, timesteps in zip(log_probabilities, valid_timesteps):
        trimmed = item_log_probabilities[:timesteps]
        labels, probabilities = greedy_ctc_labels(trimmed)
        confidences = {kind.value: ctc_confidence(trimmed, probabilities, kind) for kind in CtcConfidenceKind}
        confidences["confidence"] = confidences[confidence_kind.value]
        decoded.append((charset.decode_labels(labels), confidences))
    return decoded


def restrict_log_probabilities_to_charset(log_probabilities: np.ndarray, full_charset: TextCharset,
                                          allowed_characters: str) -> np.ndarray:
    """Constrained decoding: mask classes outside `allowed_characters` (blank kept), renormalize per step."""
    allowed_labels = [CTC_BLANK_INDEX] + [full_charset.character_to_label[c] for c in allowed_characters
                                          if c in full_charset.character_to_label]
    masked = np.full_like(log_probabilities, -np.inf)
    masked[..., allowed_labels] = log_probabilities[..., allowed_labels]
    normalizer = np.logaddexp.reduce(masked[..., allowed_labels], axis=-1, keepdims=True)
    return masked - normalizer
