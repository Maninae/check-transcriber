"""Blank check stock for one template: paper, then offset pre-print, cached per (template, dpi, texture).

Rendering order, every ink multiplying what is under it (see print_model.py):
1. paper (tint; textured: fibre, formation, specks, security fibres);
2. scenic wash (PERSONAL_NAME_ONLY stock that has one);
3. colour plate: security pattern + guilloche band + the family's colour elements, shifted by the
   template's plate misregistration (textured only);
4. dark plate: labels, rules, boxes, icons, microprint.

Each stage draws from its own rng stream seeded by crc32(template_id), so the family geometry, and
therefore every slot and label box, is identical for the clean and textured versions.
Cache: stock is uint8 RGB (~4-8 MB per template at 300 dpi); `STOCK_CACHE_SIZE` bounds memory.
"""

import functools
import zlib
from dataclasses import dataclass

import numpy as np

from synthetic_checks.check_fields import FieldLabel
from synthetic_checks.check_layout import RENDER_DPI
from synthetic_checks.check_templates import BorderKind, ScenicKind, TemplateDesign
from synthetic_checks.families import FAMILY_DRAWERS
from synthetic_checks.field_slots import FieldSlots
from synthetic_checks.printed_text.print_model import multiply_ink, multiply_transmittance, offset_ink, shift_coverage
from synthetic_checks.stock.paper_texture import make_paper
from synthetic_checks.stock.scenic_background import make_scenic_transmittance
from synthetic_checks.stock.security_pattern import make_security_pattern
from synthetic_checks.stock.stock_canvas import StockCanvas, guilloche_band_coverage

STOCK_CACHE_SIZE = 48
GUILLOCHE_BAND_STRENGTH = 0.8
CLEAN_PATTERN_WARP_PX = 9.0  # the pattern design itself is warped either way; only the print texture differs



LAYOUT_STREAM, PATTERN_STREAM, PAPER_STREAM, OFFSET_STREAM, SCENIC_STREAM = range(5)


@dataclass(frozen=True)
class BlankStock:
    """A rendered blank stock: pixels, the pre-printed text boxes, and where each fill-in goes."""

    rgb: np.ndarray                          # uint8 (height, width, 3); treat as read-only (cached)
    preprinted: tuple[FieldLabel, ...]
    slots: FieldSlots


def template_stream(template: TemplateDesign, stream: int) -> np.random.Generator:
    """Deterministic rng for one stage of one template (crc32, because str hash() is salted per process)."""
    return np.random.default_rng([zlib.crc32(template.template_id.encode()), stream])


def color_plate_coverage(template: TemplateDesign, canvas: StockCanvas) -> np.ndarray:
    """Everything the colour plate prints, as one coverage map."""
    pattern = make_security_pattern(template.pattern_kind, canvas.width, canvas.height, template_stream(template, PATTERN_STREAM),
                                    CLEAN_PATTERN_WARP_PX, template.has_pantograph) * template.pattern_strength
    if template.border_kind == BorderKind.GUILLOCHE_BAND:
        pattern = np.maximum(pattern, guilloche_band_coverage(canvas.width, canvas.height, canvas.dpi) * GUILLOCHE_BAND_STRENGTH)
    return np.maximum(pattern, np.asarray(canvas.color_plate, np.float32) / 255.0)


@functools.lru_cache(maxsize=STOCK_CACHE_SIZE)
def render_blank_template(template: TemplateDesign, dpi: int = RENDER_DPI, textured: bool = True) -> BlankStock:
    """Blank stock for `template` (see module docstring). `textured=False` is the clean print-sheet version."""
    canvas = StockCanvas(template, dpi)
    slots = FAMILY_DRAWERS[template.layout_family](canvas, template_stream(template, LAYOUT_STREAM))

    image = make_paper(canvas.width, canvas.height, template.paper_tint_rgb, dpi, template_stream(template, PAPER_STREAM),
                       textured, template.has_security_fibres)
    if template.scenic_kind != ScenicKind.NONE:
        multiply_transmittance(image, make_scenic_transmittance(template.scenic_kind, canvas.width, canvas.height,
                                                                template_stream(template, SCENIC_STREAM)))
    offset_rng = template_stream(template, OFFSET_STREAM)
    color_coverage = offset_ink(color_plate_coverage(template, canvas), offset_rng, textured)
    if textured:
        color_coverage = shift_coverage(color_coverage, template.plate_misregistration_px)
    multiply_ink(image, color_coverage, template.pattern_rgb)
    dark_coverage = offset_ink(np.asarray(canvas.dark_plate, np.float32) / 255.0, offset_rng, textured)
    multiply_ink(image, dark_coverage, template.dark_ink_rgb)

    rgb = np.clip(image * 255.0 + 0.5, 0, 255).astype(np.uint8)
    rgb.setflags(write=False)
    return BlankStock(rgb, tuple(canvas.preprinted), slots)
