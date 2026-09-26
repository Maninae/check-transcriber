"""Lay one line of text out as a float glyph-coverage map, before any pen or warp.

Two strategies, chosen by the font:
- print hands: glyph by glyph, each with its own size, width, rotation and baseline jitter,
  so repeated letters already differ before the line warp.
- cursive hands: word by word, so the font's joins stay connected; per-letter variety comes
  from the smooth line warp applied afterwards.

The map has generous margins so the later shear, wander and warp never push ink off it.
Glyph spans (character, x0, x1 in map pixels) let quirks target one glyph (a retraced digit).
"""

from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw

from synthetic_checks.fonts.font_registry import load_font
from synthetic_checks.handwriting.handwriting_habits import HandHabits

MARGIN_X_PER_EM = 0.9           # room for slant shear and warp at both line ends
MARGIN_Y_PER_EM = 0.45
TILE_PAD_PER_EM = 0.35          # room around a glyph so rotation does not clip it
WORD_SIZE_JITTER_SHRINK = 0.5   # cursive words vary less in size than print glyphs
MIN_GLYPH_EM_PX = 6
FULL_INK = 255.0


@dataclass
class LaidOutLine:
    """Glyph coverage for one line plus where things are on it."""

    coverage: np.ndarray                  # float32 (H, W) in [0, 1]
    baseline_y: float                     # nominal baseline row
    origin_x: float                       # x where the first glyph starts
    x_height_px: float
    glyph_spans: list[tuple[str, float, float]] = field(default_factory=list)
    retrace_coverage: np.ndarray | None = None  # second pass over one glyph, same shape


def render_glyph_tile(text: str, font_id: str, em_px: float, width_scale: float,
                      rotation_degrees: float) -> tuple[np.ndarray, int, int]:
    """Coverage tile of `text` plus the tile position of its baseline-left origin (x, y)."""
    font = load_font(font_id, max(MIN_GLYPH_EM_PX, int(round(em_px))))
    x0, y0, x1, y1 = font.getbbox(text, anchor="ls")
    pad = int(em_px * TILE_PAD_PER_EM) + 2
    tile = Image.new("L", (x1 - x0 + 2 * pad, y1 - y0 + 2 * pad), 0)
    origin_x, origin_y = pad - x0, pad - y0
    ImageDraw.Draw(tile).text((origin_x, origin_y), text, font=font, fill=255, anchor="ls")
    if abs(width_scale - 1) > 1e-3:
        tile = tile.resize((max(1, int(round(tile.width * width_scale))), tile.height), Image.Resampling.BILINEAR)
        origin_x = int(round(origin_x * width_scale))
    if abs(rotation_degrees) > 1e-3:
        tile = tile.rotate(rotation_degrees, resample=Image.Resampling.BILINEAR, center=(origin_x, origin_y))
    return np.asarray(tile, np.float32) / FULL_INK, origin_x, origin_y


def stamp_tile(layer: np.ndarray, tile: np.ndarray, left: int, top: int) -> None:
    """Max-composite `tile` into `layer` at (left, top), clipping at the layer edge."""
    layer_height, layer_width = layer.shape
    x0, y0 = max(0, left), max(0, top)
    x1, y1 = min(layer_width, left + tile.shape[1]), min(layer_height, top + tile.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    region = layer[y0:y1, x0:x1]
    np.maximum(region, tile[y0 - top:y1 - top, x0 - left:x1 - left], out=region)


def split_into_units(text: str, is_cursive: bool) -> list[str]:
    """Drawing units: single characters for print hands; words and single spaces for cursive."""
    if not is_cursive:
        return list(text)
    units: list[str] = []
    for index, word in enumerate(text.split(" ")):
        if index:
            units.append(" ")
        if word:
            units.append(word)
    return units


def lay_out_line(text: str, font_id: str, em_px: float, x_height_px: float, is_cursive: bool,
                 habits: HandHabits, rng: np.random.Generator, retrace_character_index: int | None = None,
                 retrace_offset_px: tuple[float, float] = (0.0, 0.0)) -> LaidOutLine:
    """Place every glyph of `text` on a fresh coverage map (see module docstring)."""
    font = load_font(font_id, max(MIN_GLYPH_EM_PX, int(round(em_px))))
    ascent, descent = font.getmetrics()
    margin_x, margin_y = int(em_px * MARGIN_X_PER_EM), int(em_px * MARGIN_Y_PER_EM)
    baseline_y = float(margin_y + ascent)
    size_jitter = habits.glyph_size_jitter * (WORD_SIZE_JITTER_SHRINK if is_cursive else 1.0)

    plans = []  # (unit, em, width_scale, rotation, baseline offset, advance)
    for unit in split_into_units(text, is_cursive):
        unit_em = em_px * (1 + rng.normal(0, size_jitter))
        width_scale = 1 + rng.normal(0, habits.glyph_width_jitter)
        if unit == " ":
            plans.append((unit, unit_em, 1.0, 0.0, 0.0, font.getlength(" ") * habits.word_spacing_scale * rng.uniform(0.8, 1.25)))
            continue
        rotation = rng.normal(0, habits.glyph_rotation_jitter_degrees)
        baseline_offset = rng.normal(0, habits.glyph_baseline_jitter * x_height_px)
        advance = load_font(font_id, max(MIN_GLYPH_EM_PX, int(round(unit_em)))).getlength(unit) * width_scale
        if not is_cursive:
            advance *= habits.letter_spacing_scale * rng.uniform(0.96, 1.05)
        plans.append((unit, unit_em, width_scale, rotation, baseline_offset, advance))

    width = int(sum(plan[5] for plan in plans) + 2 * margin_x)
    height = int(baseline_y + descent + margin_y)
    coverage = np.zeros((height, width), np.float32)
    retrace_coverage = np.zeros_like(coverage) if retrace_character_index is not None else None
    glyph_spans: list[tuple[str, float, float]] = []
    cursor_x = float(margin_x)
    for unit, unit_em, width_scale, rotation, baseline_offset, advance in plans:
        if unit != " ":
            tile, tile_origin_x, tile_origin_y = render_glyph_tile(unit, font_id, unit_em, width_scale, rotation)
            left = int(round(cursor_x - tile_origin_x))
            top = int(round(baseline_y + baseline_offset - tile_origin_y))
            stamp_tile(coverage, tile, left, top)
            unit_font = load_font(font_id, max(MIN_GLYPH_EM_PX, int(round(unit_em))))
            for offset, character in enumerate(unit):
                character_x0 = cursor_x + unit_font.getlength(unit[:offset]) * width_scale
                character_x1 = cursor_x + unit_font.getlength(unit[:offset + 1]) * width_scale
                if retrace_coverage is not None and len(glyph_spans) == retrace_character_index:
                    retrace_tile, retrace_x, retrace_y = render_glyph_tile(character, font_id, unit_em, width_scale,
                                                                           rotation + rng.normal(0, 2.0))
                    stamp_tile(retrace_coverage, retrace_tile,
                               int(round(character_x0 + retrace_offset_px[0] - retrace_x)),
                               int(round(baseline_y + baseline_offset + retrace_offset_px[1] - retrace_y)))
                glyph_spans.append((character, character_x0, character_x1))
        else:
            glyph_spans.append((" ", cursor_x, cursor_x + advance))
        cursor_x += advance
    return LaidOutLine(coverage, baseline_y, float(margin_x), x_height_px, glyph_spans, retrace_coverage)
