"""Procedural fabric-like backgrounds: a thread weave, soft folds, and low-frequency shading.

Used by tests and as an explicit fallback (`--procedural-backgrounds N`) before real
backgrounds exist. Writes files so the normal loader and split logic treat them like photos.
"""

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

FABRIC_BASE_COLORS_RGB = [
    (238, 236, 230), (214, 224, 236), (232, 222, 214), (205, 214, 200), (190, 196, 210),
    (240, 228, 232), (222, 222, 222), (170, 186, 204), (236, 232, 214), (120, 140, 170),
]
PROCEDURAL_SIZE = (3200, 2400)


def fabric_weave(width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    """Fine crossing threads with irregular thickness, as a luminance modulation around 1."""
    y_grid, x_grid = np.mgrid[0:height, 0:width].astype(np.float32)
    period = rng.uniform(2.5, 5.0)
    thickness_noise = cv2.resize(rng.standard_normal((height // 16 + 1, width // 16 + 1)).astype(np.float32), (width, height))
    warp_threads = np.sin(2 * np.pi * x_grid / period + thickness_noise * 0.8)
    weft_threads = np.sin(2 * np.pi * y_grid / period + thickness_noise * 0.8)
    return 1 + rng.uniform(0.015, 0.05) * (warp_threads * weft_threads)


def soft_folds(width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    """A few long wrinkles, each a highlight on one side and a shadow on the other."""
    y_grid, x_grid = np.mgrid[0:height, 0:width].astype(np.float32)
    shading = np.ones((height, width), np.float32)
    for _ in range(int(rng.integers(2, 7))):
        angle = rng.uniform(0, np.pi)
        offset = rng.uniform(-0.5, 0.5) * max(width, height)
        signed_distance = (x_grid - width / 2) * np.cos(angle) + (y_grid - height / 2) * np.sin(angle) - offset
        signed_distance += 40 * np.sin(2 * np.pi * ((x_grid * np.sin(angle) - y_grid * np.cos(angle)) / rng.uniform(600, 1800)))
        fold_width = rng.uniform(20, 120)
        shading += rng.uniform(0.03, 0.12) * -signed_distance / fold_width * np.exp(-(signed_distance / fold_width) ** 2)
    return shading


def generate_fabric_background(rng: np.random.Generator, size: tuple[int, int] = PROCEDURAL_SIZE) -> np.ndarray:
    """One fabric image, float32 RGB in [0, 1]."""
    width, height = size
    base = np.array(FABRIC_BASE_COLORS_RGB[int(rng.integers(len(FABRIC_BASE_COLORS_RGB)))], np.float32) / 255
    image = np.ones((height, width, 3), np.float32) * base
    if rng.random() < 0.3:
        stripe_period = rng.uniform(40, 160)
        y_grid, x_grid = np.mgrid[0:height, 0:width].astype(np.float32)
        stripes = (np.sin(2 * np.pi * (x_grid if rng.random() < 0.5 else y_grid) / stripe_period) > 0.6).astype(np.float32)
        stripe_color = np.clip(base * rng.uniform(0.75, 0.92), 0, 1)
        image = image * (1 - stripes[..., None]) + stripe_color * stripes[..., None]
    shading = fabric_weave(width, height, rng) * soft_folds(width, height, rng)
    low_frequency = cv2.resize(rng.uniform(0.94, 1.06, (4, 5)).astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC)
    return np.clip(image * (shading * low_frequency)[..., None], 0, 1)


def write_procedural_backgrounds(output_directory: Path, count: int, seed: int) -> list[Path]:
    """Generate `count` fabrics as JPEGs; existing files with the same name are reused."""
    output_directory.mkdir(parents=True, exist_ok=True)
    written_paths = []
    for index in range(count):
        file_path = output_directory / f"fabric_{seed}_{index:03d}.jpg"
        if not file_path.exists():
            fabric = generate_fabric_background(np.random.default_rng([seed, index]))
            cv2.imwrite(str(file_path), cv2.cvtColor((fabric * 255).astype(np.uint8), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        written_paths.append(file_path)
    logger.info("procedural backgrounds ready in %s (%d)", output_directory, count)
    return written_paths
