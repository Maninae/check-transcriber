"""How ink sits on paper: every layer multiplies the reflectance under it (subtractive), never alpha-over.

Three processes, matching how a real check is made:
- offset pre-print (stock): smooth ink with a hair of dot gain and a slow density wobble; the colour
  plate may sit up to ~2 px off the dark plate (plate misregistration), see `shift_coverage`;
- laser personalization and MICR toner (`laser_toner`): a little spread, ragged edges, grainy fill,
  and a few stray toner satellites near strokes, so a 3x crop shows toner rather than vector text;
- pen: a handwriting RGBA layer multiplied in by `multiply_rgba_layer`.
Coverage maps are float32 in 0..1; the image being built is float32 RGB reflectance in 0..1.
"""

import cv2
import numpy as np

from synthetic_checks.stock.noise_fields import fine_noise, smooth_noise

OFFSET_DOT_GAIN_SIGMA = 0.45
OFFSET_DENSITY_WOBBLE = 0.07
TONER_SPREAD_SIGMA = 0.6
TONER_EDGE_ROUGHNESS = 0.22
TONER_EDGE_SHARPNESS = 2.4
TONER_GRAIN = 0.16
TONER_MOTTLE = 0.30
TONER_SATELLITE_PROBABILITY = 0.0008
TONER_RGB = (36, 36, 40)


def multiply_ink(image: np.ndarray, coverage: np.ndarray, ink_rgb: tuple[int, int, int], x0: int = 0, y0: int = 0) -> None:
    """Darken `image` in place by an ink of colour `ink_rgb` at `coverage`, pasted with its top-left at (x0, y0)."""
    height, width = image.shape[:2]
    cov_x0, cov_y0 = max(0, -x0), max(0, -y0)
    dest_x0, dest_y0 = max(0, x0), max(0, y0)
    dest_x1 = min(width, x0 + coverage.shape[1])
    dest_y1 = min(height, y0 + coverage.shape[0])
    if dest_x1 <= dest_x0 or dest_y1 <= dest_y0:
        return
    patch = coverage[cov_y0:cov_y0 + dest_y1 - dest_y0, cov_x0:cov_x0 + dest_x1 - dest_x0]
    absorbance = 1.0 - np.array(ink_rgb, np.float32) / 255.0
    image[dest_y0:dest_y1, dest_x0:dest_x1] *= 1.0 - patch[..., None] * absorbance


def multiply_transmittance(image: np.ndarray, transmittance_rgb: np.ndarray) -> None:
    """Multiply a full-size per-channel transmittance (e.g. a scenic background) into `image` in place."""
    image *= transmittance_rgb


def offset_ink(coverage: np.ndarray, rng: np.random.Generator, textured: bool) -> np.ndarray:
    """Offset-litho look for a plate: slight dot gain and a slow density wobble; unchanged when clean."""
    if not textured:
        return coverage
    gained = cv2.GaussianBlur(coverage, (0, 0), sigmaX=OFFSET_DOT_GAIN_SIGMA)
    wobble = 1.0 + OFFSET_DENSITY_WOBBLE * smooth_noise(coverage.shape[0], coverage.shape[1], 150.0, rng)
    return np.clip(gained * wobble, 0, 1)


def shift_coverage(coverage: np.ndarray, offset_xy: tuple[float, float]) -> np.ndarray:
    """Sub-pixel translate a coverage map (plate misregistration); the uncovered border stays empty."""
    offset_x, offset_y = offset_xy
    if abs(offset_x) < 0.05 and abs(offset_y) < 0.05:
        return coverage
    transform = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
    return cv2.warpAffine(coverage, transform, (coverage.shape[1], coverage.shape[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def laser_toner(coverage: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Turn a crisp anti-aliased text coverage into fused-toner coverage (same size, 0..1).

    - spread: a small blur, then a noisy threshold, so edges grow a hair and go ragged;
    - fill: fine grain plus a slightly coarser mottle, so solid strokes are not uniformly black;
    - satellites: rare stray toner specks within a few pixels of the strokes.
    """
    height, width = coverage.shape
    spread = cv2.GaussianBlur(coverage, (0, 0), sigmaX=TONER_SPREAD_SIGMA)
    edge_noise = 0.6 * fine_noise(height, width, (0.6, 0.6), rng) + 0.4 * fine_noise(height, width, (1.4, 1.4), rng)
    ragged = np.clip((spread - 0.5 + TONER_EDGE_ROUGHNESS * edge_noise) * TONER_EDGE_SHARPNESS + 0.5, 0, 1)
    # Mottle: clumps of thinner toner read as light streaks inside solid strokes.
    mottle = np.clip(fine_noise(height, width, (1.3, 0.9), rng), 0, None)
    density = 0.97 + TONER_GRAIN * 0.5 * fine_noise(height, width, (0.45, 0.45), rng) - TONER_MOTTLE * mottle * 0.5
    toner = ragged * np.clip(density, 0.45, 1.0)
    near_strokes = (cv2.GaussianBlur(coverage, (0, 0), sigmaX=2.5) > 0.04) & (spread < 0.15)
    satellites = (rng.random((height, width)) < TONER_SATELLITE_PROBABILITY) & near_strokes
    if satellites.any():
        speck_layer = cv2.GaussianBlur(satellites.astype(np.float32), (0, 0), sigmaX=0.45) * 1.6
        toner = np.maximum(toner, np.clip(speck_layer, 0, 0.5))
    return toner.astype(np.float32)


def multiply_rgba_layer(image: np.ndarray, layer_rgba: np.ndarray, x0: int, y0: int) -> None:
    """Multiply a uint8 RGBA ink layer (pen) into `image` in place at (x0, y0): colour absorbs where alpha is on."""
    alpha = layer_rgba[..., 3].astype(np.float32) / 255.0
    absorbance = 1.0 - layer_rgba[..., :3].astype(np.float32) / 255.0
    height, width = alpha.shape
    image[y0:y0 + height, x0:x0 + width] *= 1.0 - alpha[..., None] * absorbance
