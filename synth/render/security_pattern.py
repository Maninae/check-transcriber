"""Security-pattern textures printed on check paper, as ink-coverage maps in [0, 1].

Each generator returns float32 (height, width): 0 bare paper, 1 full colour-plate ink.
Real offset security backgrounds are engraved by hand-tuned software and printed through
a slightly stretched plate, so nothing is a perfect sine or grid. Every generator therefore
works on WARPED pixel coordinates (a smooth random displacement field), lets its line
period drift across the sheet, and the result is modulated by a patchy density field.
Optional extras: a hidden "VOID" pantograph (same tone, different dot size) and a faint emblem.
"""

import numpy as np
from PIL import Image, ImageDraw

from synth.render.check_templates import SecurityPatternKind
from synth.render.fonts import load_font
from synth.render.noise_fields import smooth_noise

MICROPRINT_WORDS = ["SECURE", "ORIGINAL", "DOCUMENT", "PROTECTED", "VALID"]
WARP_FEATURE_PX = 260.0
LINE_WIDTH_PX = 1.3


def soft_lines(phase_field: np.ndarray, line_fraction: np.ndarray | float) -> np.ndarray:
    """Anti-aliased lines wherever `phase_field` crosses an integer; `line_fraction` is line width / period."""
    distance_to_line = np.abs(((phase_field + 0.5) % 1.0) - 0.5)  # 0 on the line, 0.5 midway
    half_width = np.clip(np.asarray(line_fraction, np.float32) / 2, 0.02, 0.45)
    return np.clip(1.0 - (distance_to_line - half_width) / (half_width * 0.8 + 0.02), 0.0, 1.0).astype(np.float32)


def warped_grids(width: int, height: int, rng: np.random.Generator, amplitude_px: float) -> tuple[np.ndarray, np.ndarray]:
    """Pixel x and y grids pushed around by a smooth random displacement of about `amplitude_px`."""
    y_grid, x_grid = np.mgrid[0:height, 0:width].astype(np.float32)
    x_grid += amplitude_px * smooth_noise(height, width, WARP_FEATURE_PX, rng)
    y_grid += amplitude_px * smooth_noise(height, width, WARP_FEATURE_PX, rng)
    return x_grid, y_grid


def drifting_period(width: int, height: int, base_period: float, rng: np.random.Generator) -> np.ndarray:
    """A line period that wanders about +-12% across the sheet."""
    return base_period * (1.0 + 0.12 * smooth_noise(height, width, 500.0, rng))


def diagonal_lines(x_grid, y_grid, rng):
    """Fine diagonal lines at a slightly wandering spacing."""
    angle = rng.uniform(0.5, 1.1)
    period = drifting_period(x_grid.shape[1], x_grid.shape[0], rng.uniform(6, 11), rng)
    return soft_lines((x_grid * np.cos(angle) + y_grid * np.sin(angle)) / period, LINE_WIDTH_PX / period)


def guilloche_waves(x_grid, y_grid, rng):
    """Two or three families of wavy parallel lines whose interference reads as a guilloche net."""
    coverage = np.zeros(x_grid.shape, np.float32)
    for _ in range(int(rng.integers(2, 4))):
        wavelength = rng.uniform(160, 520)
        amplitude = rng.uniform(10, 45) * (1 + 0.5 * np.sin(2 * np.pi * x_grid / rng.uniform(900, 2400)))
        period = rng.uniform(7, 13)
        phase = (y_grid + amplitude * np.sin(2 * np.pi * x_grid / wavelength + rng.uniform(0, 6.28))) / period
        coverage = np.maximum(coverage, soft_lines(phase, LINE_WIDTH_PX / period))
    return coverage


def fan_guilloche(x_grid, y_grid, rng):
    """Two fans of wavy rays from focal points off the sheet: the crossed-net look of banknote stock."""
    height, width = x_grid.shape
    coverage = np.zeros(x_grid.shape, np.float32)
    for side in (-1, 1):
        center_x = width * (0.5 + side * rng.uniform(0.1, 0.45))
        center_y = height * rng.choice([-0.4, 1.4]) * rng.uniform(0.9, 1.3)
        radius = np.hypot(x_grid - center_x, y_grid - center_y) + 1.0
        ray_count = rng.uniform(260, 520)
        theta = np.arctan2(y_grid - center_y, x_grid - center_x)
        phase = theta * ray_count / (2 * np.pi) + rng.uniform(1.5, 4) * np.sin(radius / rng.uniform(40, 120))
        local_period = 2 * np.pi * radius / ray_count
        coverage = np.maximum(coverage, soft_lines(phase, LINE_WIDTH_PX / local_period))
    return coverage


def rosette(x_grid, y_grid, rng):
    """Concentric rings with a lobed wobble around a centre, two slightly different families."""
    height, width = x_grid.shape
    center_x, center_y = width * rng.uniform(0.3, 0.7), height * rng.uniform(0.3, 0.7)
    radius = np.hypot(x_grid - center_x, y_grid - center_y)
    theta = np.arctan2(y_grid - center_y, x_grid - center_x)
    coverage = np.zeros(x_grid.shape, np.float32)
    for _ in range(2):
        period = rng.uniform(8, 13)
        lobes = int(rng.integers(5, 14))
        phase = (radius + rng.uniform(6, 18) * np.sin(lobes * theta + radius / rng.uniform(60, 200))) / period
        coverage = np.maximum(coverage, soft_lines(phase, LINE_WIDTH_PX / period))
    return coverage


def dot_screen(x_grid, y_grid, rng):
    """A halftone dot grid whose dot size swells and shrinks in patches."""
    height, width = x_grid.shape
    period = rng.uniform(6, 10)
    dots = (np.cos(2 * np.pi * x_grid / period) * np.cos(2 * np.pi * y_grid / period) + 1) / 2
    threshold = 0.62 + 0.05 * smooth_noise(height, width, 400.0, rng)
    return np.clip((dots - threshold) * 4, 0, 1).astype(np.float32)


def crosshatch(x_grid, y_grid, rng):
    """Two crossing sets of fine lines at slightly different, wandering spacings."""
    height, width = x_grid.shape
    first_period = drifting_period(width, height, rng.uniform(8, 14), rng)
    second_period = first_period * rng.uniform(0.9, 1.1)
    first = soft_lines((x_grid + y_grid) / first_period, LINE_WIDTH_PX / first_period)
    second = soft_lines((x_grid - y_grid) / second_period, LINE_WIDTH_PX / second_period)
    return np.maximum(first, second)


def soft_gradient(x_grid, y_grid, rng):
    """A smooth tonal sweep with a faint large wave, like plain pastel stock."""
    height, width = x_grid.shape
    sweep = x_grid / width * rng.uniform(0.3, 0.8)
    wave = 0.25 * (1 + np.sin(2 * np.pi * (x_grid / width * rng.uniform(1, 2.5) + y_grid / height * 0.7)))
    return np.clip(sweep + wave * 0.6, 0, 1).astype(np.float32)


def microprint(x_grid, y_grid, rng):
    """Rows of tiny repeated words, as on high-security stock (one strip drawn, then tiled with offsets)."""
    height, width = x_grid.shape
    font = load_font("pt_sans", int(rng.integers(9, 13)))
    row_spacing = int(rng.integers(12, 18))
    phrase = " ".join(MICROPRINT_WORDS[int(i)] for i in rng.integers(0, len(MICROPRINT_WORDS), 6)) + " "
    strip_width = 2 * width
    strip = Image.new("L", (strip_width, row_spacing), 0)
    ImageDraw.Draw(strip).text((0, 0), phrase * (strip_width // max(1, int(font.getlength(phrase))) + 2), font=font, fill=255)
    strip_pixels = np.asarray(strip, np.float32) / 255.0
    coverage = np.zeros((height, width), np.float32)
    for row_index, y in enumerate(range(0, height, row_spacing)):
        offset = (row_index * 17) % width
        rows = min(row_spacing, height - y)
        coverage[y:y + rows] = strip_pixels[:rows, offset:offset + width]
    return coverage * 0.45  # dense ink; scale so it reads as tone, not wallpaper


PATTERN_GENERATORS = {
    SecurityPatternKind.DIAGONAL_LINES: diagonal_lines,
    SecurityPatternKind.GUILLOCHE_WAVES: guilloche_waves,
    SecurityPatternKind.FAN_GUILLOCHE: fan_guilloche,
    SecurityPatternKind.ROSETTE: rosette,
    SecurityPatternKind.DOT_SCREEN: dot_screen,
    SecurityPatternKind.CROSSHATCH: crosshatch,
    SecurityPatternKind.SOFT_GRADIENT: soft_gradient,
    SecurityPatternKind.MICROPRINT: microprint,
}


def void_pantograph(width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    """Hidden 'VOID': fine dots outside the word, coarser dots of equal average tone inside it.

    At reading distance both areas are the same flat tint; a copier (or a 3x crop) resolves the word.
    """
    word_layer = Image.new("L", (width, height), 0)
    font = load_font("pt_sans_bold", int(height * 0.42))
    word_draw = ImageDraw.Draw(word_layer)
    for column in range(int(rng.integers(1, 4))):
        word_draw.text((width * (0.15 + 0.3 * column), height * 0.5), "VOID", font=font, fill=255, anchor="lm")
    inside_word = np.asarray(word_layer, np.float32) / 255.0
    y_grid, x_grid = np.mgrid[0:height, 0:width].astype(np.float32)
    fine = (np.cos(2 * np.pi * x_grid / 4.0) * np.cos(2 * np.pi * y_grid / 4.0) + 1) / 2
    coarse = (np.cos(2 * np.pi * x_grid / 8.0) * np.cos(2 * np.pi * y_grid / 8.0) + 1) / 2
    fine_dots = np.clip((fine - 0.72) * 5, 0, 1)
    coarse_dots = np.clip((coarse - 0.72) * 5, 0, 1)
    return (fine_dots * (1 - inside_word) + coarse_dots * inside_word).astype(np.float32)


def make_security_pattern(kind: SecurityPatternKind, width: int, height: int, rng: np.random.Generator,
                          warp_amplitude_px: float = 9.0, with_pantograph: bool = False) -> np.ndarray:
    """Coverage map for `kind` on warped coordinates, with patchy density, a faint emblem and optional pantograph."""
    x_grid, y_grid = warped_grids(width, height, rng, warp_amplitude_px)
    coverage = PATTERN_GENERATORS[kind](x_grid, y_grid, rng)
    density = np.clip(0.9 + 0.08 * smooth_noise(height, width, 320.0, rng), 0.7, 1.05)  # real stock is even; more read as parchment
    coverage = coverage * density
    radial_distance = np.hypot((x_grid - width * rng.uniform(0.35, 0.65)) / (width * 0.22),
                               (y_grid - height * 0.5) / (height * 0.38))
    coverage = coverage + np.clip(1.2 - radial_distance, 0, 1) * rng.uniform(0.0, 0.3)
    if with_pantograph:
        coverage = np.maximum(coverage, void_pantograph(width, height, rng) * 0.55)
    return np.clip(coverage, 0, 1).astype(np.float32)
