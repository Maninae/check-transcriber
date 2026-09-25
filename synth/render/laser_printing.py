"""Laser (toner) printing of per-check text: payer block, number, bank, printed fill-ins, MICR.

Each string is rasterized alone on a small coverage layer, passed through the toner model
(`print_model.laser_toner`) when textured, multiplied into the image, and its label box is
measured from the toner that actually landed (coverage above `INK_BOX_THRESHOLD`), so boxes
stay tight even though toner spreads and throws satellites.
"""

import numpy as np
from PIL import Image, ImageDraw

from synth.render.field_slots import TextSlot
from synth.render.fonts import load_font
from synth.render.print_model import laser_toner, multiply_ink
from synth.render.text_drawing import fit_font_size

INK_BOX_THRESHOLD = 0.2
LAYER_PAD_PX = 5
MICR_REFERENCE_EM = 200


class LaserPrinter:
    """Prints strings onto one check image (float32 RGB reflectance), clean or with toner texture."""

    def __init__(self, image: np.ndarray, rng: np.random.Generator, textured: bool):
        self.image = image
        self.rng = rng
        self.textured = textured

    def print_text(self, text: str, font_id: str, em_px: int, position: tuple[float, float], ink_rgb: tuple[int, int, int],
                   anchor: str = "ls", max_width: float | None = None) -> tuple[int, int, int, int] | None:
        """Print `text` at `position` (Pillow anchor); return its tight ink box in image pixels, or None if nothing landed."""
        if not text:
            return None
        size = fit_font_size(font_id, text, em_px, max_width) if max_width else em_px
        font = load_font(font_id, size)
        glyph_x0, glyph_y0, glyph_x1, glyph_y1 = font.getbbox(text, anchor=anchor)
        layer_x0 = int(np.floor(position[0] + glyph_x0)) - LAYER_PAD_PX
        layer_y0 = int(np.floor(position[1] + glyph_y0)) - LAYER_PAD_PX
        layer_size = (int(glyph_x1 - glyph_x0) + 2 * LAYER_PAD_PX + 2, int(glyph_y1 - glyph_y0) + 2 * LAYER_PAD_PX + 2)
        layer = Image.new("L", layer_size, 0)
        ImageDraw.Draw(layer).text((position[0] - layer_x0, position[1] - layer_y0), text, font=font, fill=255, anchor=anchor)
        coverage = np.asarray(layer, np.float32) / 255.0
        if self.textured:
            coverage = laser_toner(coverage, self.rng)
        multiply_ink(self.image, coverage, ink_rgb, layer_x0, layer_y0)
        return self.inked_box(coverage, layer_x0, layer_y0)

    def print_in_slot(self, text: str, font_id: str, slot: TextSlot, ink_rgb: tuple[int, int, int],
                      em_px: int | None = None) -> tuple[int, int, int, int] | None:
        """Print `text` in a slot (its anchor, width limit, and em unless overridden)."""
        return self.print_text(text, font_id, em_px or slot.em_px, (slot.x, slot.baseline_y), ink_rgb, slot.anchor, slot.max_width)

    def inked_box(self, coverage: np.ndarray, layer_x0: int, layer_y0: int) -> tuple[int, int, int, int] | None:
        """Tight box of coverage above threshold, shifted to image pixels and clipped to the image."""
        inked = coverage > INK_BOX_THRESHOLD
        rows = np.flatnonzero(inked.any(axis=1))
        columns = np.flatnonzero(inked.any(axis=0))
        if len(rows) == 0:
            return None
        height, width = self.image.shape[:2]
        x0 = min(width, max(0, layer_x0 + int(columns[0])))
        y0 = min(height, max(0, layer_y0 + int(rows[0])))
        x1 = min(width, max(0, layer_x0 + int(columns[-1]) + 1))
        y1 = min(height, max(0, layer_y0 + int(rows[-1]) + 1))
        return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def micr_em_px_for_digit_height(digit_height_px: int) -> int:
    """GnuMICR em size whose digit '0' is `digit_height_px` tall."""
    x0, y0, x1, y1 = load_font("gnu_micr", MICR_REFERENCE_EM).getbbox("0")
    return max(8, int(round(MICR_REFERENCE_EM * digit_height_px / (y1 - y0))))
