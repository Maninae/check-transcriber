"""The person filling in a check: one `Writer` per check, shared by every handwritten field.

This is the contract between the check renderer (which decides WHAT is written WHERE) and
the handwriting engine (which decides how a pen puts it on paper):
- `sample_writer(rng, ...)` picks a person: fonts, pen, slant, size, spacing and quirks.
- `draw_handwritten_field(canvas, text, writer, ...)` inks one field onto an RGBA canvas and
  returns the tight box of the ink it laid down, in canvas pixels, or None if nothing inked.

Font pools are passed in (not read from the registry) so dataset splits can hold
handwriting fonts out: eval checks are written by fonts train never saw.

- `em_px` from the renderer is a nominal size; the engine converts it to a physical x-height
  (`FILL_IN_X_HEIGHT_PER_EM`; signatures: ascender height) and picks each font's em to hit it.
- Signatures reuse the same pen with a lighter, faster, more slanted hand and may get a flourish.
"""

from dataclasses import dataclass, replace

import numpy as np
from PIL import Image

from synth.render.fonts import FontRole, font_ids_with_role
from synth.render.handwriting_field_renderer import FieldStyle, draw_field
from synth.render.handwriting_font_metrics import x_height_for_ascender_height
from synth.render.handwriting_habits import HandHabits
from synth.render.handwriting_quirks import sample_field_quirks
from synth.render.pen_ballpoint import BallpointPen

FILL_IN_X_HEIGHT_PER_EM = 0.46      # 0.18 in nominal em -> ~0.083 in x-height, as in real filled checks
SIGNATURE_ASCENDER_PER_EM = 0.62    # 0.30 in nominal em -> ~0.19 in tall capitals and ascenders
BALLPOINT_WIDTH_RANGE_PX = (3.2, 4.6)   # ~0.27-0.39 mm trace at 300 dpi (dz photo: ~4 px)
GEL_WIDTH_RANGE_PX = (3.8, 4.8)     # 0.5 mm gel lays a slightly wider, denser trace
BLUE_INK_MIN_BLUE_EXCESS = 40           # ink is "blue" when blue exceeds red by this much
GEL_PEN_PROBABILITY_FOR_BLACK = 0.4
RETRACE_PROBABILITY_RANGE = (0.0, 0.12)
OVERRUN_PROBABILITY_RANGE = (0.0, 0.10)
FLOURISH_PROBABILITY_RANGE = (0.3, 0.8)
SIGNATURE_EXTRA_SLANT_RANGE = (6.0, 18.0)
SIGNATURE_PEN_RADIUS_SCALE = 0.85       # a fast signature stroke lays a slightly thinner trace


@dataclass(frozen=True)
class Writer:
    """One person's handwriting habits and pen, fixed for every field on one check."""

    font_id: str                 # handwriting font for date, payee, amounts, memo
    signature_font_id: str
    ink_rgb: tuple[int, int, int]
    pen_width_px: float          # ballpoint trace width at the render dpi
    slant_degrees: float         # writer's habitual slant (a shear); per-glyph jitter is added on top
    size_scale: float            # writes bigger or smaller than the nominal x-height
    pressure_variability: float  # 0 = even pressure, 1 = very uneven
    hand: HandHabits
    signature_hand: HandHabits
    pen: BallpointPen
    signature_pen: BallpointPen
    retrace_probability: float   # per field with digits
    overrun_probability: float   # per field
    flourish_probability: float  # per signature


def sample_hand(rng: np.random.Generator, slant_degrees: float, messiness: float) -> HandHabits:
    """Geometry habits for one hand; `messiness` in [0, 1] scales every jitter."""
    return HandHabits(
        slant_degrees=slant_degrees,
        letter_spacing_scale=float(rng.uniform(0.92, 1.12)),
        word_spacing_scale=float(rng.uniform(0.9, 1.6)),
        glyph_size_jitter=0.03 + 0.035 * messiness,
        glyph_width_jitter=0.02 + 0.05 * messiness,
        glyph_rotation_jitter_degrees=1.0 + 2.5 * messiness,
        glyph_baseline_jitter=0.02 + 0.035 * messiness,
        baseline_wander=0.05 + 0.15 * messiness,
        baseline_drift_slope=float(rng.normal(0, 0.008)),
        elastic_warp=0.03 + 0.045 * messiness,
    )


def sample_pen(rng: np.random.Generator, ink_rgb: tuple[int, int, int], pressure_variability: float) -> tuple[BallpointPen, float]:
    """A pen matched to the ink colour, plus its trace width in px."""
    is_blue = ink_rgb[2] - ink_rgb[0] >= BLUE_INK_MIN_BLUE_EXCESS
    is_gel = not is_blue and rng.random() < GEL_PEN_PROBABILITY_FOR_BLACK
    width_px = float(rng.uniform(*(GEL_WIDTH_RANGE_PX if is_gel else BALLPOINT_WIDTH_RANGE_PX)))
    pen = BallpointPen(
        radius_px=width_px / 2,
        opacity=float(rng.uniform(0.94, 0.99) if is_gel else rng.uniform(0.78, 0.92)),
        width_swing=0.06 + 0.18 * pressure_variability,
        density_swing=(0.04 if is_gel else 0.08) + 0.14 * pressure_variability,
        grain_std=float(rng.uniform(0.02, 0.05) if is_gel else rng.uniform(0.06, 0.14)),
        blob_probability=float(rng.uniform(0.05, 0.25) if is_gel else rng.uniform(0.15, 0.5)),
        skip_rate=0.0 if is_gel else float(rng.uniform(0.0, 0.6)),
    )
    return pen, width_px


def sample_writer(rng: np.random.Generator, ink_rgb: tuple[int, int, int],
                  handwriting_font_ids: list[str] | None = None,
                  signature_font_ids: list[str] | None = None) -> Writer:
    """Pick a writer; font pools default to every registered font of the role."""
    handwriting_pool = handwriting_font_ids or font_ids_with_role(FontRole.HANDWRITING)
    signature_pool = signature_font_ids or font_ids_with_role(FontRole.SIGNATURE)
    messiness = float(rng.beta(2, 3))
    pressure_variability = float(rng.uniform(0.1, 0.9))
    slant_degrees = float(np.clip(rng.normal(4, 6), -8, 18))
    pen, width_px = sample_pen(rng, ink_rgb, pressure_variability)
    signature_slant = float(np.clip(slant_degrees + rng.uniform(*SIGNATURE_EXTRA_SLANT_RANGE), 0, 30))
    return Writer(
        font_id=str(rng.choice(handwriting_pool)),
        signature_font_id=str(rng.choice(signature_pool)),
        ink_rgb=ink_rgb,
        pen_width_px=width_px,
        slant_degrees=slant_degrees,
        size_scale=float(rng.uniform(0.85, 1.2)),
        pressure_variability=pressure_variability,
        hand=sample_hand(rng, slant_degrees, messiness),
        signature_hand=sample_hand(rng, signature_slant, min(1.0, messiness + 0.3)),
        pen=pen,
        signature_pen=replace(pen, radius_px=pen.radius_px * SIGNATURE_PEN_RADIUS_SCALE,
                              width_swing=pen.width_swing * 1.5, density_swing=pen.density_swing * 1.3),
        retrace_probability=float(rng.uniform(*RETRACE_PROBABILITY_RANGE)),
        overrun_probability=float(rng.uniform(*OVERRUN_PROBABILITY_RANGE)),
        flourish_probability=float(rng.uniform(*FLOURISH_PROBABILITY_RANGE)),
    )


def draw_handwritten_field(canvas: Image.Image, text: str, writer: Writer, em_px: int,
                           baseline_left: tuple[float, float], max_width_px: float, rng: np.random.Generator,
                           is_signature: bool = False) -> tuple[int, int, int, int] | None:
    """Ink `text` onto RGBA `canvas` in this writer's hand; return the tight ink box or None."""
    if not text.strip():
        return None
    pen = writer.signature_pen if is_signature else writer.pen
    quirks = sample_field_quirks(text, 0.0 if is_signature else writer.retrace_probability,
                                 0.0 if is_signature else writer.overrun_probability, pen.radius_px, rng)
    font_id = writer.signature_font_id if is_signature else writer.font_id
    if is_signature:
        x_height_px = x_height_for_ascender_height(font_id, em_px * writer.size_scale * SIGNATURE_ASCENDER_PER_EM)
    else:
        x_height_px = em_px * writer.size_scale * FILL_IN_X_HEIGHT_PER_EM
    style = FieldStyle(
        font_id=font_id,
        x_height_px=x_height_px,
        hand=writer.signature_hand if is_signature else writer.hand,
        pen=pen,
        ink_rgb=writer.ink_rgb,
        quirks=quirks,
        add_flourish=is_signature and rng.random() < writer.flourish_probability,
    )
    return draw_field(canvas, text, style, baseline_left, max_width_px, rng)
