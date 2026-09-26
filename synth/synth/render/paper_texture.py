"""Blank check paper as an RGB reflectance map (float32, 0..1), before any ink.

Textured paper (the on-screen synthetic path) layers, all multiplicative on the tint:
- tint unevenness: a very low-frequency brightness drift of about 0.5%, with a trace of hue drift;
- formation: faint cloudy flocs of pulp, blotches a few to tens of pixels across;
- fibre: faint fine noise stretched along the machine direction (x), plus a finer isotropic grain;
- specks: a few dark dust/pulp specks, and on some stock short coloured security fibres.
Clean paper (print sheets) is the flat tint: the real sheet supplies its own texture.
"""

import cv2
import numpy as np

from synth.render.noise_fields import fine_noise, smooth_noise

TINT_DRIFT_STRENGTH = 0.005
TINT_CHROMA_DRIFT_STRENGTH = 0.003
FORMATION_STRENGTH = 0.006   # formation shows in transmitted light; reflected, it is nearly invisible (more read as parchment)
FLOC_STRENGTH = 0.012
FIBRE_STRENGTH = 0.009
GRAIN_STRENGTH = 0.012
SPECKS_PER_SQUARE_INCH = 3.0
SECURITY_FIBRE_COLORS_RGB = [(205, 60, 70), (60, 90, 190), (70, 150, 110)]


def add_specks(paper: np.ndarray, dpi: int, rng: np.random.Generator) -> None:
    """Darken a few tiny random spots in place (dust, pulp shives)."""
    height, width = paper.shape[:2]
    speck_count = int(SPECKS_PER_SQUARE_INCH * (width / dpi) * (height / dpi))
    speck_mask = np.zeros((height, width), np.float32)
    ys = rng.integers(0, height, speck_count)
    xs = rng.integers(0, width, speck_count)
    speck_mask[ys, xs] = rng.uniform(0.4, 1.8, speck_count)  # blurred below into ~1-2 px soft dots
    speck_mask = cv2.GaussianBlur(speck_mask, (0, 0), sigmaX=0.7)
    paper *= (1.0 - np.clip(speck_mask, 0, 0.3))[..., None]


def add_security_fibres(paper: np.ndarray, dpi: int, rng: np.random.Generator) -> None:
    """Draw 6-16 short curved coloured fibres in place, pale, as embedded in real check paper."""
    height, width = paper.shape[:2]
    fibre_layer = np.zeros((height, width, 3), np.float32)
    for _ in range(int(rng.integers(6, 17))):
        color_absorbance = 1.0 - np.array(SECURITY_FIBRE_COLORS_RGB[int(rng.integers(3))], np.float32) / 255.0
        length_px = rng.uniform(0.08, 0.22) * dpi
        heading = rng.uniform(0, 2 * np.pi)
        bend = rng.uniform(-0.004, 0.004)
        points = [np.array([rng.uniform(0, width), rng.uniform(0, height)])]
        for _ in range(12):
            heading += bend * 12 + rng.normal(0, 0.02)
            points.append(points[-1] + length_px / 12 * np.array([np.cos(heading), np.sin(heading)]))
        polyline = np.round(np.array(points)).astype(np.int32)
        strength = rng.uniform(0.18, 0.35)
        cv2.polylines(fibre_layer, [polyline], False, tuple(float(v * strength) for v in color_absorbance), 1, cv2.LINE_AA)
    paper *= 1.0 - fibre_layer


def make_paper(width: int, height: int, tint_rgb: tuple[int, int, int], dpi: int, rng: np.random.Generator,
               textured: bool, with_security_fibres: bool) -> np.ndarray:
    """Paper reflectance (height, width, 3) in 0..1; flat tint when `textured` is False."""
    paper = np.empty((height, width, 3), np.float32)
    paper[:] = np.array(tint_rgb, np.float32) / 255.0
    if not textured:
        return paper
    luminance_texture = (1.0
                         + FORMATION_STRENGTH * smooth_noise(height, width, 45.0, rng)
                         + FLOC_STRENGTH * fine_noise(height, width, (2.2, 1.6), rng)
                         + FIBRE_STRENGTH * fine_noise(height, width, (1.8, 0.5), rng)
                         + GRAIN_STRENGTH * fine_noise(height, width, (0.5, 0.5), rng))
    paper *= luminance_texture[..., None]
    paper *= (1.0 + TINT_DRIFT_STRENGTH * smooth_noise(height, width, 420.0, rng))[..., None]
    for channel in range(3):  # a trace of hue drift only; real stock varies in brightness, not colour
        paper[..., channel] *= 1.0 + TINT_CHROMA_DRIFT_STRENGTH * smooth_noise(height, width, 420.0, rng)
    add_specks(paper, dpi, rng)
    if with_security_fibres:
        add_security_fibres(paper, dpi, rng)
    return np.clip(paper, 0.0, 1.0)
