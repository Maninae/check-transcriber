"""Character and word error rates between a predicted and a ground-truth string (via rapidfuzz).

Both rates are edit distance divided by the ground-truth length, so they can exceed 1.0 when the
prediction is much longer than the truth. A blank prediction scores exactly 1.0 (every truth
character deleted). Callers pass already-normalized text (`normalize_free_text`).
"""

from rapidfuzz.distance import Levenshtein


def character_error_rate(predicted_text: str, ground_truth_text: str) -> float:
    """Levenshtein distance over characters / len(ground truth); 0.0 when both are empty."""
    if not ground_truth_text:
        return 0.0 if not predicted_text else 1.0
    return Levenshtein.distance(predicted_text, ground_truth_text) / len(ground_truth_text)


def word_error_rate(predicted_text: str, ground_truth_text: str) -> float:
    """Levenshtein distance over whitespace-split words / number of ground-truth words."""
    ground_truth_words = ground_truth_text.split()
    predicted_words = predicted_text.split()
    if not ground_truth_words:
        return 0.0 if not predicted_words else 1.0
    return Levenshtein.distance(predicted_words, ground_truth_words) / len(ground_truth_words)
