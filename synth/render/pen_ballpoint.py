"""Re-ink a one-pixel centreline the way a ballpoint pen lays ink on paper.

A ballpoint leaves a thin trace of nearly constant width (0.3-0.45 mm, about 3.5-5.5 px at
300 dpi) whose width and darkness breathe with hand pressure. Model, per pixel:

    radius  = pen radius x (1 + width_swing x pressure)          (pressure: smooth noise field)
    alpha   = clip(radius - distance_to_centreline + 0.5, 0, 1)   (antialiased disc sweep)
    alpha  *= opacity x density(pressure, stroke-end blobs, grain, skips)

- Blobs: ink pools where the ball stops, so stroke ends and junctions get a darker, slightly
  wider spot.
- Skips: where a second, rarer noise field peaks, the ball rolled dry and the trace fades.
- Grain: fine multiplicative noise; blue ballpoint ink is translucent and uneven, black gel
  is denser and more even, so grain and opacity are per-pen.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from synth.render.handwriting_noise import smooth_noise_field
from synth.render.pen_centreline import centreline_endpoints, centreline_junctions

ANTIALIAS_HALF_PIXEL = 0.5
MIN_RADIUS_PX = 0.6
PRESSURE_CORRELATION_PER_RADIUS = 14.0   # pressure changes over ~a letter
SKIP_CORRELATION_PER_RADIUS = 3.0        # skips are short gaps
GRAIN_BLUR_SIGMA_PX = 0.6
BLOB_SIGMA_PER_RADIUS = 1.1
BLOB_RADIUS_GAIN = 0.35                  # blobs widen the trace by up to 35% of the radius
BLOB_DENSITY_GAIN = 0.22
SKIP_THRESHOLD_STD = 2.1                 # noise peaks above this many std can become skips
SKIP_RESIDUAL_ALPHA = 0.2


@dataclass(frozen=True)
class BallpointPen:
    """One physical pen: trace size, ink behaviour and how steady the hand is."""

    radius_px: float            # half the trace width at average pressure
    opacity: float              # ink alpha at full density (blue ballpoint < black gel)
    width_swing: float          # relative radius change per std of pressure (0.1-0.3)
    density_swing: float        # relative darkness change per std of pressure
    grain_std: float            # fine per-pixel darkness noise
    blob_probability: float     # chance a stroke end or junction gets an ink blob
    skip_rate: float            # 0 = never skips; 1 = skips at every pressure dip


def ink_centreline(skeleton: np.ndarray, pen: BallpointPen, rng: np.random.Generator) -> np.ndarray:
    """Float32 alpha in [0, 1] of the ballpoint trace along boolean `skeleton` (same shape)."""
    if not skeleton.any():
        return np.zeros(skeleton.shape, np.float32)
    shape = skeleton.shape
    distance_to_centreline = cv2.distanceTransform((~skeleton).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    pressure = smooth_noise_field(rng, shape, pen.radius_px * PRESSURE_CORRELATION_PER_RADIUS)
    blobs = ink_blob_field(skeleton, pen, rng)
    radius = pen.radius_px * (1 + pen.width_swing * pressure + BLOB_RADIUS_GAIN * blobs)
    radius = np.maximum(radius, MIN_RADIUS_PX)
    coverage = np.clip(radius - distance_to_centreline + ANTIALIAS_HALF_PIXEL, 0, 1)

    density = 1 + pen.density_swing * pressure + BLOB_DENSITY_GAIN * blobs
    density *= 1 + pen.grain_std * cv2.GaussianBlur(rng.standard_normal(shape).astype(np.float32), (0, 0), GRAIN_BLUR_SIGMA_PX)
    density *= ink_skip_field(shape, pen, rng)
    return np.clip(coverage * np.clip(density, 0, 1.25) * pen.opacity, 0, 1).astype(np.float32)


def ink_blob_field(skeleton: np.ndarray, pen: BallpointPen, rng: np.random.Generator) -> np.ndarray:
    """Soft spots in [0, 1] where ink pooled: a random subset of stroke ends and junctions."""
    stop_points = centreline_endpoints(skeleton) | centreline_junctions(skeleton)
    rows, columns = np.nonzero(stop_points)
    keep = rng.random(len(rows)) < pen.blob_probability
    impulses = np.zeros(skeleton.shape, np.float32)
    impulses[rows[keep], columns[keep]] = rng.uniform(0.6, 1.0, keep.sum())
    sigma = pen.radius_px * BLOB_SIGMA_PER_RADIUS
    blurred = cv2.GaussianBlur(impulses, (0, 0), sigma)
    peak_of_unit_impulse = 1 / (2 * np.pi * sigma * sigma)
    return np.clip(blurred / peak_of_unit_impulse, 0, 1)


def ink_skip_field(shape: tuple[int, int], pen: BallpointPen, rng: np.random.Generator) -> np.ndarray:
    """Multiplier in [SKIP_RESIDUAL_ALPHA, 1]: 1 almost everywhere, dipping where the ball ran dry."""
    if pen.skip_rate <= 0:
        return np.ones(shape, np.float32)
    skip_noise = smooth_noise_field(rng, shape, pen.radius_px * SKIP_CORRELATION_PER_RADIUS)
    threshold = SKIP_THRESHOLD_STD + (1 - pen.skip_rate)
    fade = np.clip(skip_noise - threshold, 0, 1)
    return (1 - (1 - SKIP_RESIDUAL_ALPHA) * fade).astype(np.float32)
