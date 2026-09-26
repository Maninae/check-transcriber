"""Ink refinement on a synthetic check: a text blob on a ruled line, a small printed label, a MICR-like line."""

import cv2
import numpy as np

from experiments.field_reading.field_localization.ink_refinement import (InkRefinementParams, compute_clean_ink_mask,
                                                                         refine_prior_to_ink_box)

CROP_WIDTH, CROP_HEIGHT = 1600, 733
TEXT_BOX = (500, 300, 900, 340)   # x0, y0, x1, y1 of the synthetic handwriting


def draw_synthetic_check(with_text: bool) -> np.ndarray:
    """Light paper with a long ruled line, a small label to the left, and optionally a 40 px tall 'word' row."""
    image = np.full((CROP_HEIGHT, CROP_WIDTH), 225, np.uint8)
    cv2.line(image, (200, 345), (1200, 345), 40, 2)                               # ruled line under the text
    cv2.putText(image, "PAY TO THE ORDER OF", (200, 335), cv2.FONT_HERSHEY_SIMPLEX, 0.35, 40, 1)
    if with_text:
        x0, y0, x1, y1 = TEXT_BOX
        for letter_x in range(x0, x1, 50):                                        # letter blobs with gaps
            cv2.rectangle(image, (letter_x, y0), (letter_x + 35, y1), 30, -1)
    return image


def payee_prior() -> dict:
    """Prior whose median box sits roughly over the text and whose region includes the label."""
    return {"median_box": [480 / CROP_WIDTH, 295 / CROP_HEIGHT, 950 / CROP_WIDTH, 345 / CROP_HEIGHT],
            "search_region": [150 / CROP_WIDTH, 260 / CROP_HEIGHT, 1300 / CROP_WIDTH, 380 / CROP_HEIGHT],
            "median_height": 40 / CROP_HEIGHT, "presence_rate": 1.0}


def test_refinement_snaps_to_text_and_ignores_line_and_label():
    """The box hugs the letter row: the ruled line and the small label are not included."""
    params = InkRefinementParams()
    clean_ink = compute_clean_ink_mask(draw_synthetic_check(with_text=True), params)
    assert clean_ink[345, 1100] == 0, "ruled line should be removed"
    box, confidence = refine_prior_to_ink_box(clean_ink, payee_prior(), params)
    assert box is not None and confidence > 0.5
    padding = params.output_padding * 40
    assert abs(box[0] - (TEXT_BOX[0] - padding)) < 4 and abs(box[2] - (TEXT_BOX[2] - 15 + padding)) < 4
    assert abs(box[1] - (TEXT_BOX[1] - padding)) < 4 and abs(box[3] - (TEXT_BOX[3] + padding)) < 6


def test_refinement_returns_no_box_without_text():
    """Only the line and the small label remain -> no box."""
    params = InkRefinementParams()
    clean_ink = compute_clean_ink_mask(draw_synthetic_check(with_text=False), params)
    box, confidence = refine_prior_to_ink_box(clean_ink, payee_prior(), params)
    assert box is None and confidence == 0.0
