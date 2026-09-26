"""Give a flat, evenly lit fabric swatch the wrinkles and folds of cloth lying on a bed.

Material swatches (ambientCG, Poly Haven) are shot flat under even light, so a check pasted on
one reads as "check on a colour chip". This adds a procedural cloth height field and shades it:
- a few long soft folds (Gaussian ridges and troughs along gently bent lines),
- anisotropic small wrinkles (low-frequency noise stretched along one direction),
- Lambert shading from one light direction, normalized to mean 1, plus slight valley darkening.

Only brightness changes (a per-pixel gain on all channels), never geometry, so labels are unaffected.
The field is built on a coarse grid and upsampled: cloth relief is low frequency by nature.
"""

import cv2
import numpy as np

RELIEF_GRID_LONG_SIDE = 256
FOLD_COUNT_RANGE = (2, 6)
FOLD_WIDTH_FRACTION_RANGE = (0.06, 0.2)     # of the grid's long side
FOLD_HEIGHT_RANGE = (0.4, 1.0)
FOLD_BEND_FRACTION = 0.08                    # how far a fold line wanders, as a share of the long side
WRINKLE_AMPLITUDE_RANGE = (0.08, 0.25)
WRINKLE_BLUR_SIGMA = 16.0
WRINKLE_STRETCH_RANGE = (1.5, 4.0)
RELIEF_HEIGHT_SCALE = 5.0                    # grid px of height per unit: a unit fold rises 5 px over its 15-50 px width
LIGHT_ELEVATION_DEGREES_RANGE = (30.0, 60.0)
VALLEY_DARKENING = 0.08
GAIN_SOFT_RANGE = 0.14                       # gains squash smoothly into 1 +- this (no hard-edged patches)


def bent_line_distance(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """Signed distance (grid px) from every pixel to a random gently bent line across the grid."""
    height, width = shape
    y_grid, x_grid = np.mgrid[0:height, 0:width].astype(np.float32)
    angle = rng.uniform(0, np.pi)
    center_x, center_y = rng.uniform(0.1, 0.9) * width, rng.uniform(0.1, 0.9) * height
    along = (x_grid - center_x) * np.cos(angle) + (y_grid - center_y) * np.sin(angle)
    across = -(x_grid - center_x) * np.sin(angle) + (y_grid - center_y) * np.cos(angle)
    bend = FOLD_BEND_FRACTION * max(shape) * np.sin(along / max(shape) * rng.uniform(1.5, 4.0) + rng.uniform(0, 2 * np.pi))
    return across - bend


def anisotropic_wrinkles(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """Low-frequency noise stretched along one direction, like fine creases in a slept-on sheet."""
    stretch = rng.uniform(*WRINKLE_STRETCH_RANGE)
    height, width = shape
    noise = rng.standard_normal((height, int(width / stretch) + 2)).astype(np.float32)
    noise = cv2.resize(cv2.GaussianBlur(noise, (0, 0), WRINKLE_BLUR_SIGMA / stretch), (width, height),
                       interpolation=cv2.INTER_CUBIC)
    rotation = cv2.getRotationMatrix2D((width / 2, height / 2), rng.uniform(0, 180), 1.0)
    noise = cv2.warpAffine(noise, rotation, (width, height), borderMode=cv2.BORDER_REFLECT)
    return noise / max(1e-6, float(noise.std()))


def cloth_height_field(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """Height (arbitrary units) of a loosely laid sheet: folds plus wrinkles."""
    height_field = np.zeros(shape, np.float32)
    for _ in range(int(rng.integers(FOLD_COUNT_RANGE[0], FOLD_COUNT_RANGE[1] + 1))):
        width = rng.uniform(*FOLD_WIDTH_FRACTION_RANGE) * max(shape)
        sign = rng.choice([-1.0, 1.0])
        height_field += sign * rng.uniform(*FOLD_HEIGHT_RANGE) * np.exp(-(bent_line_distance(shape, rng) / width) ** 2)
    height_field += rng.uniform(*WRINKLE_AMPLITUDE_RANGE) * anisotropic_wrinkles(shape, rng)
    return height_field


def lambert_gain(height_field: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Per-pixel brightness gain from shading the height field under one light, mean 1."""
    grad_y, grad_x = np.gradient(height_field * RELIEF_HEIGHT_SCALE)
    normal = np.dstack([-grad_x, -grad_y, np.ones_like(grad_x)])
    normal /= np.linalg.norm(normal, axis=2, keepdims=True)
    azimuth = rng.uniform(0, 2 * np.pi)
    elevation = np.radians(rng.uniform(*LIGHT_ELEVATION_DEGREES_RANGE))
    light = np.array([np.cos(azimuth) * np.cos(elevation), np.sin(azimuth) * np.cos(elevation), np.sin(elevation)])
    shading = np.clip(normal @ light, 0.0, 1.0)
    shading /= max(1e-6, float(shading.mean()))
    valleys = np.clip(-(height_field - height_field.mean()), 0, None)
    shading *= 1 - VALLEY_DARKENING * valleys / max(1e-6, float(valleys.max()))
    deviation = shading / max(1e-6, float(shading.mean())) - 1
    return (1 + GAIN_SOFT_RANGE * np.tanh(deviation / GAIN_SOFT_RANGE)).astype(np.float32)


def add_cloth_relief(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return a copy of uint8 RGB `image` shaded as wrinkled, folded cloth (same size and dtype)."""
    height, width = image.shape[:2]
    scale = RELIEF_GRID_LONG_SIDE / max(height, width)
    grid_shape = (max(8, int(height * scale)), max(8, int(width * scale)))
    gain = lambert_gain(cloth_height_field(grid_shape, rng), rng)
    gain = cv2.resize(gain, (width, height), interpolation=cv2.INTER_CUBIC)
    return np.clip(image.astype(np.float32) * gain[..., None], 0, 255).astype(np.uint8)
