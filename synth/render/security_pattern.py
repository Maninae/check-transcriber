"""Security-pattern textures printed on check paper, as ink-coverage maps in [0, 1].

Each generator returns a float32 array of shape (height, width): 0 is bare paper, 1 is
full pattern ink. The renderer blends it between the paper tint and the pattern color.
Patterns are deterministic per template (seeded by the caller) and cached upstream.
"""

import numpy as np
from PIL import Image, ImageDraw

from synth.render.check_templates import SecurityPatternKind
from synth.render.fonts import load_font

MICROPRINT_WORDS = ["SECURE", "ORIGINAL", "DOCUMENT", "PROTECTED", "VALID"]


def soft_lines(phase_field: np.ndarray, period_px: float, line_fraction: float) -> np.ndarray:
    """Thin anti-aliased lines wherever `phase_field` crosses a multiple of `period_px`."""
    distance_to_line = np.abs(((phase_field / period_px) % 1.0) - 0.5) * 2.0  # 1 on the line, 0 midway
    return np.clip((distance_to_line - (1.0 - line_fraction)) / line_fraction, 0.0, 1.0).astype(np.float32)


def pixel_grids(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """x and y coordinate grids as float32."""
    y_grid, x_grid = np.mgrid[0:height, 0:width].astype(np.float32)
    return x_grid, y_grid


def diagonal_lines(width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    """Evenly spaced fine diagonal lines."""
    x_grid, y_grid = pixel_grids(width, height)
    angle = rng.uniform(0.5, 1.1)
    return soft_lines(x_grid * np.cos(angle) + y_grid * np.sin(angle), rng.uniform(6, 11), 0.35)


def guilloche_waves(width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    """Two families of wavy parallel lines whose interference reads as a guilloche."""
    x_grid, y_grid = pixel_grids(width, height)
    coverage = np.zeros((height, width), np.float32)
    for _ in range(2):
        wavelength = rng.uniform(120, 420)
        amplitude = rng.uniform(10, 40)
        phase_field = y_grid + amplitude * np.sin(2 * np.pi * x_grid / wavelength + rng.uniform(0, 6.28))
        coverage = np.maximum(coverage, soft_lines(phase_field, rng.uniform(7, 13), 0.3))
    return coverage


def dot_screen(width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    """A halftone-like dot grid with slowly varying dot size."""
    x_grid, y_grid = pixel_grids(width, height)
    period = rng.uniform(6, 10)
    dots = (np.cos(2 * np.pi * x_grid / period) * np.cos(2 * np.pi * y_grid / period) + 1) / 2
    size_modulation = 0.55 + 0.25 * np.sin(2 * np.pi * x_grid / (width * rng.uniform(0.3, 0.8)))
    return np.clip((dots - size_modulation) * 4, 0, 1).astype(np.float32)


def crosshatch(width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    """Two crossing sets of fine lines."""
    x_grid, y_grid = pixel_grids(width, height)
    period = rng.uniform(8, 14)
    first = soft_lines(x_grid + y_grid, period, 0.3)
    second = soft_lines(x_grid - y_grid, period, 0.3)
    return np.maximum(first, second)


def soft_gradient(width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    """A smooth tonal sweep with a faint large wave, like scenic check stock."""
    x_grid, y_grid = pixel_grids(width, height)
    sweep = x_grid / width * rng.uniform(0.3, 0.8)
    wave = 0.25 * (1 + np.sin(2 * np.pi * (x_grid / width * rng.uniform(1, 2.5) + y_grid / height * 0.7)))
    return np.clip(sweep + wave * 0.6, 0, 1).astype(np.float32)


def microprint(width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    """Rows of tiny repeated words, as on high-security stock (one strip drawn, then tiled)."""
    font = load_font("pt_sans", int(rng.integers(9, 13)))
    row_spacing = int(rng.integers(12, 18))
    row_shift = 17
    phrase = " ".join(MICROPRINT_WORDS[int(i)] for i in rng.integers(0, len(MICROPRINT_WORDS), 6)) + " "
    strip_width = 2 * width
    strip = Image.new("L", (strip_width, row_spacing), 0)
    ImageDraw.Draw(strip).text((0, 0), phrase * (strip_width // max(1, int(font.getlength(phrase))) + 2), font=font, fill=255)
    strip_pixels = np.asarray(strip, np.float32) / 255.0
    coverage = np.zeros((height, width), np.float32)
    for row_index, y in enumerate(range(0, height, row_spacing)):
        offset = (row_index * row_shift) % width
        rows = min(row_spacing, height - y)
        coverage[y:y + rows] = strip_pixels[:rows, offset:offset + width]
    # Microprint is dense ink; scale it down so it reads as tone, not as text wallpaper.
    return coverage * 0.45


PATTERN_GENERATORS = {
    SecurityPatternKind.DIAGONAL_LINES: diagonal_lines,
    SecurityPatternKind.GUILLOCHE_WAVES: guilloche_waves,
    SecurityPatternKind.DOT_SCREEN: dot_screen,
    SecurityPatternKind.CROSSHATCH: crosshatch,
    SecurityPatternKind.SOFT_GRADIENT: soft_gradient,
    SecurityPatternKind.MICROPRINT: microprint,
}


def make_security_pattern(kind: SecurityPatternKind, width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    """Coverage map for `kind`, modulated by a faint central emblem so the paper is not uniform."""
    coverage = PATTERN_GENERATORS[kind](width, height, rng)
    x_grid, y_grid = pixel_grids(width, height)
    emblem_center_x = width * rng.uniform(0.35, 0.65)
    radial_distance = np.sqrt(((x_grid - emblem_center_x) / (width * 0.22)) ** 2 + ((y_grid - height * 0.5) / (height * 0.38)) ** 2)
    emblem = np.clip(1.2 - radial_distance, 0, 1) * rng.uniform(0.0, 0.35)
    return np.clip(coverage + emblem, 0, 1).astype(np.float32)
