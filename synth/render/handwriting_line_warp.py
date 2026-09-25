"""Bend a laid-out line the way a hand does: slant, a wandering baseline, wobbly glyph shapes.

One dense remap does all three at once (cv2.remap samples the source at a per-pixel offset):
- slant: a shear about the baseline (letters lean; the baseline itself stays put), unlike a
  rotation, which would tilt the whole line;
- baseline: a slow vertical wave along x plus a straight uphill/downhill drift;
- elastic: a smooth 2-D displacement with correlation length ~ one glyph, so every instance
  of a letter lands on a different part of the field and comes out a little different, while
  strokes stay continuous (cursive joins survive).

The same maps warp the retrace layer so a second pass follows the first.
"""

import math

import cv2
import numpy as np

from synth.render.handwriting_habits import HandHabits
from synth.render.handwriting_noise import smooth_noise_curve, smooth_noise_field

BASELINE_WAVE_CORRELATION_X_HEIGHTS = 5.0
ELASTIC_CORRELATION_X_HEIGHTS = 1.1  # longer than a stroke gap, so warps bend glyphs without squashing counters


def build_line_warp_maps(shape: tuple[int, int], baseline_y: float, origin_x: float, x_height_px: float,
                         habits: HandHabits, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Source-coordinate maps (map_x, map_y) for cv2.remap: output pixel (y, x) samples source (map_y, map_x)."""
    height, width = shape
    rows, columns = np.mgrid[0:height, 0:width].astype(np.float32)
    shear = math.tan(math.radians(habits.slant_degrees))
    baseline_wave = habits.baseline_wander * x_height_px * smooth_noise_curve(
        rng, width, BASELINE_WAVE_CORRELATION_X_HEIGHTS * x_height_px)
    baseline_shift = baseline_wave + habits.baseline_drift_slope * (np.arange(width, dtype=np.float32) - origin_x)
    elastic_amplitude = habits.elastic_warp * x_height_px
    elastic_correlation = ELASTIC_CORRELATION_X_HEIGHTS * x_height_px
    map_x = columns - shear * (baseline_y - rows) + elastic_amplitude * smooth_noise_field(rng, shape, elastic_correlation)
    map_y = rows - baseline_shift[None, :] + elastic_amplitude * smooth_noise_field(rng, shape, elastic_correlation)
    return map_x.astype(np.float32), map_y.astype(np.float32)


def apply_line_warp(coverage: np.ndarray, maps: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Warp a float coverage map with maps from `build_line_warp_maps`."""
    map_x, map_y = maps
    return cv2.remap(coverage, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
