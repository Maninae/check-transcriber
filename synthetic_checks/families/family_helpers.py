"""Building blocks shared by the layout families: each draws pre-print on a StockCanvas and/or returns a TextSlot.

Conventions (all families):
- `yf` values are the baseline of the writing, as a fraction of height; the printed rule sits
  `RULE_DROP_INCHES` below it and handwriting/laser fills sit on `FILL_DROP_INCHES`, so text rests on the rule.
- Geometry jitter comes from the template-seeded rng, so a template's layout is fixed across runs.
"""

import numpy as np

from synthetic_checks.check_layout import (
    BANK_CITY_EM_INCHES,
    BANK_NAME_EM_INCHES,
    CHECK_NUMBER_EM_INCHES,
    HANDWRITING_EM_INCHES,
    LABEL_EM_INCHES,
    MICR_BASELINE_FROM_BOTTOM_INCHES,
    PAYER_ADDRESS_EM_INCHES,
    PAYER_NAME_EM_INCHES,
    SERIAL_EM_INCHES,
    SIGNATURE_EM_INCHES,
    SMALL_LABEL_EM_INCHES,
)
from synthetic_checks.field_slots import TextSlot
from synthetic_checks.stock.stock_canvas import StockCanvas

RULE_DROP_INCHES = 0.02
FILL_DROP_INCHES = 0.012
FILL_START_PAD_INCHES = 0.04
DEFAULT_JITTER_FRACTION = 0.006
MICR_MAX_WIDTH_FRACTION = 0.86


def jitter(rng: np.random.Generator, scale: float = DEFAULT_JITTER_FRACTION) -> float:
    """A small symmetric random shift (fraction of width/height)."""
    return float(rng.uniform(-scale, scale))


def ruled_fill(canvas: StockCanvas, x0f: float, x1f: float, yf: float, handwriting: bool = True,
               draw_rule: bool = True, em_scale: float = 1.0) -> TextSlot:
    """Draw a writing rule from x0f to x1f at baseline yf (optional) and return the fill slot above it."""
    x0, x1, baseline = canvas.fx(x0f), canvas.fx(x1f), canvas.fy(yf)
    if draw_rule:
        canvas.line(x0, x1, baseline + canvas.px(RULE_DROP_INCHES))
    pad = canvas.px(FILL_START_PAD_INCHES)
    em = canvas.px(HANDWRITING_EM_INCHES * em_scale) if handwriting else canvas.px(0.12 * em_scale)
    return TextSlot(x0 + pad, baseline + canvas.px(FILL_DROP_INCHES), x1 - x0 - pad, em)


def payer_slots(canvas: StockCanvas, xf: float, yf: float, anchor: str = "ls", with_address: bool = True,
                name_scale: float = 1.0, max_width_f: float = 0.47) -> tuple[TextSlot, TextSlot | None]:
    """Name slot (baseline yf) and, optionally, the first address-line slot under it."""
    name_em = canvas.px(PAYER_NAME_EM_INCHES * name_scale)
    name = TextSlot(canvas.fx(xf), canvas.fy(yf), canvas.fx(max_width_f), name_em, anchor)
    if not with_address:
        return name, None
    address_em = canvas.px(PAYER_ADDRESS_EM_INCHES)
    address = TextSlot(canvas.fx(xf), canvas.fy(yf) + address_em * 1.45, canvas.fx(max_width_f), address_em, anchor)
    return name, address


def number_slots(canvas: StockCanvas, right_xf: float, yf: float, em_scale: float = 1.0) -> tuple[TextSlot, TextSlot]:
    """Right-aligned check-number slot and the fractional routing slot just below it."""
    number_em = canvas.px(CHECK_NUMBER_EM_INCHES * em_scale)
    number = TextSlot(canvas.fx(right_xf), canvas.fy(yf), canvas.fx(0.25), number_em, "rs")
    fraction_em = canvas.px(PAYER_ADDRESS_EM_INCHES * 0.8)
    fraction = TextSlot(canvas.fx(right_xf), canvas.fy(yf) + fraction_em * 1.6, canvas.fx(0.25), fraction_em, "rs")
    return number, fraction


def bank_block(canvas: StockCanvas, xf: float, yf: float, max_width_f: float = 0.42,
               logo_plate: str = "dark") -> tuple[TextSlot, TextSlot]:
    """Draw the template's bank logo at (xf, yf top) and return (bank name slot, bank city slot)."""
    name_em = canvas.px(BANK_NAME_EM_INCHES)
    logo_size = canvas.px(BANK_NAME_EM_INCHES * 1.5)
    text_x = canvas.bank_logo(canvas.fx(xf), canvas.fy(yf), logo_size, logo_plate)
    name = TextSlot(text_x, canvas.fy(yf) + name_em * 0.85, canvas.fx(max_width_f), name_em)
    city_em = canvas.px(BANK_CITY_EM_INCHES)
    city = TextSlot(text_x, name.baseline_y + city_em * 1.35, canvas.fx(max_width_f), city_em)
    return name, city


def amount_box(canvas: StockCanvas, x0f: float, x1f: float, y0f: float, y1f: float, handwriting: bool = True,
               dollar_inside: bool = False) -> TextSlot:
    """Courtesy amount box with its '$'; outlined or underlined per template. Returns the fill slot."""
    x0, x1, y0, y1 = canvas.fx(x0f), canvas.fx(x1f), canvas.fy(y0f), canvas.fy(y1f)
    box_height = y1 - y0
    dollar_em = canvas.px(LABEL_EM_INCHES * 1.8)
    if dollar_inside:
        canvas.label("label_dollar_sign", "$", x0 + box_height * 0.15, y1 - box_height * 0.22, em_px=dollar_em, apply_case=False)
        fill_x0 = x0 + box_height * 0.15 + dollar_em * 0.8
    else:
        canvas.label("label_dollar_sign", "$", x0 - canvas.px(0.03), y1 - box_height * 0.22, anchor="rs", em_px=dollar_em, apply_case=False)
        fill_x0 = x0 + box_height * 0.25
    if canvas.template.amount_box_outlined:
        canvas.rectangle((x0, y0, x1, y1))
    else:
        canvas.line(x0, x1, y1)
    em = canvas.px(HANDWRITING_EM_INCHES) if handwriting else canvas.px(0.12)
    return TextSlot(fill_x0, y1 - box_height * 0.22, x1 - fill_x0 - box_height * 0.15, em)


def security_note(canvas: StockCanvas, xf: float, yf: float, max_width_f: float = 0.14) -> None:
    """Padlock icon plus 'Security features included. Details on back.' in small type."""
    small_em = canvas.px(SMALL_LABEL_EM_INCHES)
    canvas.padlock_icon(canvas.fx(xf), canvas.fy(yf) - small_em * 1.4, int(small_em * 1.3))
    canvas.label("label_security_note", "Security features included. Details on back.", canvas.fx(xf) + small_em * 1.7,
                 canvas.fy(yf), em_px=int(small_em * 0.8), max_width=canvas.fx(max_width_f), apply_case=False)


def signature_slot(canvas: StockCanvas, x0f: float, x1f: float, yf: float, caption: str | None = None) -> TextSlot:
    """Signature rule (plain or microprint) with an optional small caption under it; returns the signing slot."""
    x0, x1, baseline = canvas.fx(x0f), canvas.fx(x1f), canvas.fy(yf)
    canvas.signature_line(x0, x1, baseline + canvas.px(RULE_DROP_INCHES))
    if caption:
        small_em = canvas.px(SMALL_LABEL_EM_INCHES)
        canvas.label("label_signature_caption", caption, x1, baseline + canvas.px(RULE_DROP_INCHES) + small_em * 1.5,
                     anchor="rs", em_px=small_em)
    return TextSlot(x0 + canvas.fx(0.03), baseline + canvas.px(0.03), x1 - x0 - canvas.fx(0.05), canvas.px(SIGNATURE_EM_INCHES))


def micr_slot(canvas: StockCanvas, start_xf: float) -> TextSlot:
    """MICR line slot in the clear band (the em is derived from the E-13B digit height by the filler)."""
    return TextSlot(canvas.fx(start_xf), canvas.height - canvas.px(MICR_BASELINE_FROM_BOTTOM_INCHES),
                    canvas.fx(MICR_MAX_WIDTH_FRACTION), 0)


def serial_slot(canvas: StockCanvas, xf: float, yf: float) -> TextSlot:
    """Small printed serial (print-sheet matching key)."""
    return TextSlot(canvas.fx(xf), canvas.fy(yf), canvas.fx(0.2), canvas.px(SERIAL_EM_INCHES))
