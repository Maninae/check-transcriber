"""The person filling in a check: one `Writer` per check, shared by every handwritten field.

This is the contract between the check renderer (which decides WHAT is written WHERE) and
the handwriting engine (which decides how a pen puts it on paper):
- `sample_writer(rng, ...)` picks a person: fonts, pen, slant, size, pressure habits.
- `draw_handwritten_field(canvas, text, writer, ...)` inks one field onto an RGBA layer and
  returns the tight box of the ink it laid down, in canvas pixels, or None if nothing inked.

Font pools are passed in (not read from the registry) so dataset splits can hold
handwriting fonts out: eval checks are written by fonts train never saw.
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image

from synth.render.fonts import FontRole, font_ids_with_role
from synth.render.text_drawing import draw_handwritten_text

FILL_IN_MAX_SLANT_DEGREES = 2.5
SIGNATURE_MAX_SLANT_DEGREES = 6.0


@dataclass(frozen=True)
class Writer:
    """One person's handwriting habits and pen, fixed for every field on one check."""

    font_id: str                 # handwriting font for date, payee, amounts, memo
    signature_font_id: str
    ink_rgb: tuple[int, int, int]
    pen_width_px: float          # ballpoint trace width at the render dpi
    slant_degrees: float         # writer's habitual slant; per-field jitter is added on top
    size_scale: float            # writes bigger or smaller than the nominal em
    pressure_variability: float  # 0 = even pressure, 1 = very uneven


def sample_writer(rng: np.random.Generator, ink_rgb: tuple[int, int, int],
                  handwriting_font_ids: list[str] | None = None,
                  signature_font_ids: list[str] | None = None) -> Writer:
    """Pick a writer; font pools default to every registered font of the role."""
    handwriting_pool = handwriting_font_ids or font_ids_with_role(FontRole.HANDWRITING)
    signature_pool = signature_font_ids or font_ids_with_role(FontRole.SIGNATURE)
    return Writer(
        font_id=str(rng.choice(handwriting_pool)),
        signature_font_id=str(rng.choice(signature_pool)),
        ink_rgb=ink_rgb,
        pen_width_px=float(rng.choice([1, 3, 3, 5])),
        slant_degrees=float(rng.uniform(-1.0, 1.0)),
        size_scale=1.0,
        pressure_variability=0.0,
    )


def draw_handwritten_field(canvas: Image.Image, text: str, writer: Writer, em_px: int,
                           baseline_left: tuple[float, float], max_width_px: float, rng: np.random.Generator,
                           is_signature: bool = False) -> tuple[int, int, int, int] | None:
    """Ink `text` onto RGBA `canvas` in this writer's hand; return the tight ink box or None."""
    return draw_handwritten_text(
        canvas, text, writer.signature_font_id if is_signature else writer.font_id, int(em_px * writer.size_scale),
        baseline_left, max_width_px, writer.ink_rgb, rng,
        max_slant_degrees=SIGNATURE_MAX_SLANT_DEGREES if is_signature else FILL_IN_MAX_SLANT_DEGREES,
        pen_width_px=int(writer.pen_width_px),
    )
