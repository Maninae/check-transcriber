"""Random fields shared by the paper, pattern and print models (all float32, deterministic given rng).

- `smooth_noise`: band-limited noise, about unit std, with features `feature_px` across.
- `fine_noise`: per-pixel noise lightly blurred, for toner grain and paper fibre.
"""

import cv2
import numpy as np


def smooth_noise(height: int, width: int, feature_px: float, rng: np.random.Generator) -> np.ndarray:
    """Noise with blobs about `feature_px` wide: a coarse random grid, bicubic-upsampled, re-normalized."""
    cells_y = max(2, int(np.ceil(height / feature_px)) + 1)
    cells_x = max(2, int(np.ceil(width / feature_px)) + 1)
    coarse = rng.standard_normal((cells_y, cells_x)).astype(np.float32)
    field = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
    return (field - field.mean()) / (field.std() + 1e-6)


def fine_noise(height: int, width: int, blur_sigma: tuple[float, float], rng: np.random.Generator) -> np.ndarray:
    """White noise blurred by (sigma_x, sigma_y) and re-normalized to unit std; anisotropic when sigmas differ."""
    field = rng.standard_normal((height, width)).astype(np.float32)
    sigma_x, sigma_y = blur_sigma
    if sigma_x > 0 or sigma_y > 0:
        field = cv2.GaussianBlur(field, (0, 0), sigmaX=max(sigma_x, 0.01), sigmaY=max(sigma_y, 0.01))
    return (field - field.mean()) / (field.std() + 1e-6)
