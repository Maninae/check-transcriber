"""Faded illustrated backgrounds for "scenic" personal stock, as RGB transmittance (float32, near 1).

Multiplied onto the paper like any other ink. Each scene is built from a few soft shapes and
then faded hard (to about a quarter strength) so fill-ins stay legible, as on real scenic checks.
"""

import cv2
import numpy as np

from synthetic_checks.check_templates import ScenicKind
from synthetic_checks.stock.noise_fields import smooth_noise

SCENIC_FADE_RANGE = (0.16, 0.3)


def ridge_line(width: int, base_y: float, relief_px: float, rng: np.random.Generator) -> np.ndarray:
    """Heights (one per column) of a mountain ridge: a smoothed, detrended random walk with about `relief_px` of relief."""
    steps = rng.normal(0, 1, width).cumsum()
    steps -= np.linspace(steps[0], steps[-1], width)
    smoothed = cv2.GaussianBlur(steps.astype(np.float32)[None, :], (0, 0), sigmaX=width / 80)[0]
    return base_y + relief_px * smoothed / (np.abs(smoothed).max() + 1e-6)


def paint_region(canvas: np.ndarray, mask: np.ndarray, color_rgb: tuple[int, int, int]) -> None:
    """Replace canvas colour by `color_rgb` where `mask` (0..1) is on, in place."""
    color = np.array(color_rgb, np.float32) / 255.0
    canvas[:] = canvas * (1 - mask[..., None]) + color * mask[..., None]


def mountains(width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    """Sky gradient over three ridges; each ridge fades into mist toward its base, farther ridges paler."""
    canvas = np.ones((height, width, 3), np.float32)
    sky_top = np.array([175, 205, 235], np.float32) / 255.0
    canvas[:] = sky_top + (1 - sky_top) * np.linspace(0, 1, height, dtype=np.float32)[:, None, None]
    y_grid = np.arange(height, dtype=np.float32)[:, None]
    ridge_colors = [(170, 188, 212), (140, 165, 182), (120, 155, 135)]
    for depth, color in enumerate(ridge_colors):
        ridge = ridge_line(width, height * (0.42 + 0.14 * depth), height * (0.2 - 0.04 * depth), rng)
        below_ridge = np.clip(y_grid - ridge[None, :], 0, 1.5) / 1.5
        mist = np.clip(1.0 - (y_grid - ridge[None, :]) / (height * 0.45), 0.15, 1.0)  # valley haze
        paint_region(canvas, below_ridge * mist, color)
    return canvas


def waves(width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    """Stacked ocean-wave bands in blues and teals."""
    canvas = np.ones((height, width, 3), np.float32)
    x_grid = np.arange(width, dtype=np.float32)[None, :]
    y_grid = np.arange(height, dtype=np.float32)[:, None]
    palette = [(170, 210, 230), (120, 180, 210), (80, 150, 180), (60, 120, 160)]
    for band, color in enumerate(palette):
        crest = height * (0.3 + 0.17 * band) + height * 0.05 * np.sin(2 * np.pi * x_grid / (width * rng.uniform(0.2, 0.5)) + rng.uniform(0, 6.3))
        paint_region(canvas, np.clip((y_grid - crest) / 2.0, 0, 1), color)
    return canvas


def watercolor(width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    """Two or three soft pigment washes with darker drying edges."""
    canvas = np.ones((height, width, 3), np.float32)
    palette = [(230, 150, 160), (150, 190, 230), (240, 200, 120), (160, 210, 170), (190, 160, 220)]
    for _ in range(int(rng.integers(2, 4))):
        field = smooth_noise(height, width, rng.uniform(250, 500), rng)
        wash = np.clip(field - rng.uniform(0.0, 0.6), 0, 1)
        edge = np.clip(1 - np.abs(field - 0.1) * 6, 0, 1) * 0.35  # pigment pools at the wash boundary
        color = np.array(palette[int(rng.integers(len(palette)))], np.float32) / 255.0
        absorbance = np.clip(wash * 0.7 + edge, 0, 1)[..., None]
        canvas *= 1 - absorbance * (1 - color)
    return canvas


def sunburst(width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    """Rays from a point near one edge in two alternating tones, fading outward."""
    center_x, center_y = width * rng.choice([0.1, 0.9]), height * rng.uniform(0.6, 1.1)
    y_grid, x_grid = np.mgrid[0:height, 0:width].astype(np.float32)
    theta = np.arctan2(y_grid - center_y, x_grid - center_x)
    radius = np.hypot(x_grid - center_x, y_grid - center_y) / width
    rays = (np.sin(theta * rng.integers(14, 28)) > 0).astype(np.float32)
    rays = cv2.GaussianBlur(rays, (0, 0), sigmaX=1.2)
    color = np.array([240, 190, 110], np.float32) / 255.0
    absorbance = (0.25 + 0.35 * rays) * np.clip(1.1 - radius, 0, 1)
    return 1 - absorbance[..., None] * (1 - color)


SCENE_BUILDERS = {ScenicKind.MOUNTAINS: mountains, ScenicKind.WAVES: waves,
                  ScenicKind.WATERCOLOR: watercolor, ScenicKind.SUNBURST: sunburst}


def make_scenic_transmittance(kind: ScenicKind, width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    """Faded scene as per-channel transmittance in (0, 1]; multiply it onto the paper."""
    scene = SCENE_BUILDERS[kind](width, height, rng)
    fade = rng.uniform(*SCENIC_FADE_RANGE)
    return (1.0 - fade * (1.0 - scene)).astype(np.float32)
