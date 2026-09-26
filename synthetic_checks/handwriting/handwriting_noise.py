"""Smooth random fields: the shared source of every wobble in the handwriting engine.

A field is white noise on a coarse grid (one sample per `correlation_px`), bicubically
upsampled to full size and renormalized to zero mean and unit standard deviation. The
correlation length sets the scale of the wobble: about a glyph for shape warps, a few
glyphs for pen pressure, a few words for baseline wander.
"""

import cv2
import numpy as np

COARSE_GRID_MARGIN_CELLS = 3


def smooth_noise_field(rng: np.random.Generator, shape: tuple[int, int], correlation_px: float) -> np.ndarray:
    """Zero-mean, unit-std float32 field of `shape` (rows, columns) varying over ~`correlation_px`."""
    height, width = shape
    coarse_rows = int(height / max(correlation_px, 1.0)) + COARSE_GRID_MARGIN_CELLS
    coarse_columns = int(width / max(correlation_px, 1.0)) + COARSE_GRID_MARGIN_CELLS
    coarse = rng.standard_normal((coarse_rows, coarse_columns)).astype(np.float32)
    field = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
    return (field - field.mean()) / (field.std() + 1e-6)


def smooth_noise_curve(rng: np.random.Generator, length: int, correlation_px: float) -> np.ndarray:
    """Zero-mean, unit-std float32 1-D curve of `length` samples varying over ~`correlation_px`."""
    return smooth_noise_field(rng, (1, length), correlation_px)[0]
