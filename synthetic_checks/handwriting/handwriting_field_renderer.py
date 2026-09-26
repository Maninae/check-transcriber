"""Ink one handwritten field onto a check: layout -> warp -> centreline -> ballpoint -> composite.

Steps:
1. Size: the font em is chosen so its x-height hits the wanted physical x-height (fonts
   disagree about em size); a line too long for its field is squeezed, then shrunk.
2. Layout the glyph coverage (`handwriting_line_layout`), warp it (`handwriting_line_warp`).
3. Threshold, thin to a one-pixel centreline and prune spurs (`pen_centreline`); add the retrace pass and
   any signature flourish at centreline level.
4. Re-ink with the ballpoint model (`pen_ballpoint`), composite onto the canvas.
5. The label box is measured from the alpha actually composited onto the canvas, after
   every transform and clipped to the canvas, so it is the tight box of the laid-down ink.
"""

from dataclasses import dataclass, replace

import numpy as np
from PIL import Image

from synthetic_checks.fonts.font_registry import FONT_SPECS_BY_ID, load_font
from synthetic_checks.handwriting.handwriting_font_metrics import em_px_for_x_height, x_height_per_em
from synthetic_checks.handwriting.handwriting_habits import HandHabits
from synthetic_checks.handwriting.handwriting_line_layout import lay_out_line
from synthetic_checks.handwriting.handwriting_line_warp import apply_line_warp, build_line_warp_maps
from synthetic_checks.handwriting.handwriting_quirks import FieldQuirks, draw_signature_flourish
from synthetic_checks.handwriting.pen_ballpoint import BallpointPen, ink_centreline
from synthetic_checks.handwriting.pen_centreline import prune_centreline_spurs, thin_to_centreline
from synthetic_checks.printed_text.text_drawing import ALPHA_INK_THRESHOLD, paste_ink_layer

COVERAGE_THRESHOLD = 0.4          # warped glyph coverage above this is "inside the stroke"
FIT_SAFETY = 0.97                 # leave a little room for warp and slant at the line end
MIN_X_HEIGHT_PX = 6.0
MAX_SPACING_SQUEEZE = 0.10       # a crowded line first loses up to 10% of its spacing, then shrinks
SPUR_LENGTH_PER_X_HEIGHT = 0.12
FULL_ALPHA = 255


@dataclass(frozen=True)
class FieldStyle:
    """Everything that decides how one field looks, resolved from the Writer."""

    font_id: str
    x_height_px: float
    hand: HandHabits
    pen: BallpointPen
    ink_rgb: tuple[int, int, int]
    quirks: FieldQuirks
    add_flourish: bool


def fit_line_to_width(text: str, style: FieldStyle, max_width_px: float) -> tuple[float, float, HandHabits]:
    """(em_px, x_height_px, hand) so `text` fits `max_width_px`: squeeze spacing first, then shrink.

    Shrinking alone clogs long lines (the pen keeps its width while letters get smaller); real
    writers crowd their letters before they write smaller.
    """
    x_height = style.x_height_px
    em_px = em_px_for_x_height(style.font_id, x_height)
    hand = style.hand
    natural_width = load_font(style.font_id, em_px).getlength(text) * hand.letter_spacing_scale
    allowed_width = max_width_px * FIT_SAFETY * style.quirks.width_allowance
    if natural_width <= allowed_width:
        return float(em_px), float(x_height), hand
    squeeze = max(1 - MAX_SPACING_SQUEEZE, allowed_width / natural_width)
    hand = replace(hand, letter_spacing_scale=hand.letter_spacing_scale * squeeze,
                   word_spacing_scale=hand.word_spacing_scale * squeeze)
    remaining_ratio = allowed_width / (natural_width * squeeze)
    if remaining_ratio < 1:
        x_height = max(MIN_X_HEIGHT_PX, x_height * remaining_ratio)
        em_px = x_height / x_height_per_em(style.font_id)
    return float(em_px), float(x_height), hand


def render_field_alpha(text: str, style: FieldStyle, max_width_px: float,
                       rng: np.random.Generator) -> tuple[np.ndarray, float, float]:
    """Float alpha of the inked field plus its baseline-left anchor (x, y) in alpha pixels."""
    em_px, x_height, hand = fit_line_to_width(text, style, max_width_px)
    is_cursive = FONT_SPECS_BY_ID[style.font_id].is_cursive
    line = lay_out_line(text, style.font_id, em_px, x_height, is_cursive, hand, rng,
                        style.quirks.retrace_character_index, style.quirks.retrace_offset_px)
    maps = build_line_warp_maps(line.coverage.shape, line.baseline_y, line.origin_x, x_height, hand, rng)
    spur_length = max(1, int(round(SPUR_LENGTH_PER_X_HEIGHT * x_height)))
    skeleton = prune_centreline_spurs(thin_to_centreline(apply_line_warp(line.coverage, maps) > COVERAGE_THRESHOLD),
                                      spur_length)
    if line.retrace_coverage is not None:
        skeleton |= thin_to_centreline(apply_line_warp(line.retrace_coverage, maps) > COVERAGE_THRESHOLD)
    if style.add_flourish:
        draw_signature_flourish(skeleton, line.baseline_y, x_height, rng)
    return ink_centreline(skeleton, style.pen, rng), line.origin_x, line.baseline_y


def composite_field(canvas: Image.Image, alpha: np.ndarray, ink_rgb: tuple[int, int, int],
                    offset_x: int, offset_y: int) -> tuple[int, int, int, int] | None:
    """Paste ink with `alpha` at an offset; return the tight box of the ink that landed on the canvas."""
    alpha_u8 = np.round(alpha * FULL_ALPHA).astype(np.uint8)
    visible_x0, visible_y0 = max(0, -offset_x), max(0, -offset_y)
    visible_x1 = min(alpha_u8.shape[1], canvas.width - offset_x)
    visible_y1 = min(alpha_u8.shape[0], canvas.height - offset_y)
    if visible_x1 <= visible_x0 or visible_y1 <= visible_y0:
        return None
    visible_alpha = alpha_u8[visible_y0:visible_y1, visible_x0:visible_x1]
    inked = visible_alpha > ALPHA_INK_THRESHOLD
    rows = np.flatnonzero(inked.any(axis=1))
    if len(rows) == 0:
        return None
    columns = np.flatnonzero(inked.any(axis=0))
    ink_layer = Image.new("RGBA", (alpha_u8.shape[1], alpha_u8.shape[0]), ink_rgb + (0,))
    ink_layer.putalpha(Image.fromarray(alpha_u8, "L"))
    paste_ink_layer(canvas, ink_layer, offset_x, offset_y)
    left, top = offset_x + visible_x0, offset_y + visible_y0
    return (int(left + columns[0]), int(top + rows[0]), int(left + columns[-1] + 1), int(top + rows[-1] + 1))


def draw_field(canvas: Image.Image, text: str, style: FieldStyle, baseline_left: tuple[float, float],
               max_width_px: float, rng: np.random.Generator) -> tuple[int, int, int, int] | None:
    """Ink `text` in `style` with its baseline starting at `baseline_left`; return the ink box or None."""
    alpha, anchor_x, anchor_y = render_field_alpha(text, style, max_width_px, rng)
    baseline_drop = style.quirks.baseline_drop_x_heights * style.x_height_px
    offset_x = int(round(baseline_left[0] - anchor_x))
    offset_y = int(round(baseline_left[1] + baseline_drop - anchor_y))
    return composite_field(canvas, alpha, style.ink_rgb, offset_x, offset_y)
