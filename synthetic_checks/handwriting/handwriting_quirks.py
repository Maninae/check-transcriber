"""Rare, human mistakes and flourishes layered on top of a normal line of handwriting.

- Retrace: the writer goes over one digit twice, slightly offset (checks invite it: people
  fix or darken amounts). Picked here; drawn by the layout as a second coverage layer.
- Overrun: the text runs a little past the end of its line or dips below the printed rule.
- Flourish: a signature's closing swoosh, drawn straight onto the centreline so the pen model
  inks it like any other stroke.

None of these change the label text; the ink box is re-measured after them.
"""

from dataclasses import dataclass

import cv2
import numpy as np

RETRACE_OFFSET_MIN_RADII = 0.6
RETRACE_OFFSET_MAX_RADII = 1.6
OVERRUN_EXTRA_WIDTH = 0.07        # may write up to 7% past the end of the line
OVERRUN_BASELINE_DROP_X_HEIGHTS = 0.3
FLOURISH_SAMPLES = 60


@dataclass(frozen=True)
class FieldQuirks:
    """Which quirks fire for one field."""

    retrace_character_index: int | None
    retrace_offset_px: tuple[float, float]
    width_allowance: float        # multiplier on the field's max width
    baseline_drop_x_heights: float


def sample_field_quirks(text: str, retrace_probability: float, overrun_probability: float,
                        pen_radius_px: float, rng: np.random.Generator) -> FieldQuirks:
    """Roll the dice for one field's retrace and overrun."""
    digit_indices = [index for index, character in enumerate(text) if character.isdigit()]
    retrace_index = None
    retrace_offset = (0.0, 0.0)
    if digit_indices and rng.random() < retrace_probability:
        retrace_index = int(rng.choice(digit_indices))
        distance = pen_radius_px * rng.uniform(RETRACE_OFFSET_MIN_RADII, RETRACE_OFFSET_MAX_RADII)
        angle = rng.uniform(0, 2 * np.pi)
        retrace_offset = (float(distance * np.cos(angle)), float(distance * np.sin(angle)))
    overrun = rng.random() < overrun_probability
    width_allowance = 1 + OVERRUN_EXTRA_WIDTH * rng.random() if overrun else 1.0
    baseline_drop = OVERRUN_BASELINE_DROP_X_HEIGHTS * rng.random() if overrun and rng.random() < 0.5 else 0.0
    return FieldQuirks(retrace_index, retrace_offset, width_allowance, baseline_drop)


def quadratic_bezier_points(start: np.ndarray, control: np.ndarray, end: np.ndarray, samples: int) -> np.ndarray:
    """Points along a quadratic Bezier curve, shape (samples, 2)."""
    t = np.linspace(0, 1, samples)[:, None]
    return (1 - t) ** 2 * start + 2 * (1 - t) * t * control + t ** 2 * end


def draw_signature_flourish(skeleton: np.ndarray, baseline_y: float, x_height_px: float,
                            rng: np.random.Generator) -> None:
    """Add a one-pixel flourish to a signature centreline in place: an underline sweep or a tail."""
    rows, columns = np.nonzero(skeleton)
    if len(rows) == 0:
        return
    left, right = columns.min(), columns.max()
    height, width = skeleton.shape
    if rng.random() < 0.55:
        # Underline sweep: from under the end of the name back to the left, slightly bowed.
        start = np.array([right - rng.uniform(0, 0.15) * (right - left), baseline_y + rng.uniform(0.2, 0.6) * x_height_px])
        end = np.array([left + rng.uniform(0.0, 0.35) * (right - left), baseline_y + rng.uniform(0.3, 0.9) * x_height_px])
        control = (start + end) / 2 + np.array([0, rng.uniform(0.3, 0.9) * x_height_px])
    else:
        # Tail: leave the last letter and trail off right and up.
        last_row = rows[columns == right].mean()
        start = np.array([right, last_row])
        end = start + np.array([rng.uniform(0.8, 2.2), -rng.uniform(0.0, 0.8)]) * x_height_px
        control = (start + end) / 2 + np.array([0, rng.uniform(0.2, 0.7) * x_height_px])
    points = quadratic_bezier_points(start, control, end, FLOURISH_SAMPLES)
    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
    canvas = np.zeros(skeleton.shape, np.uint8)
    cv2.polylines(canvas, [np.round(points).astype(np.int32)], isClosed=False, color=1, thickness=1, lineType=cv2.LINE_8)
    skeleton |= canvas.astype(bool)
