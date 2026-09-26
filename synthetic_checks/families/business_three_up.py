"""BUSINESS_THREE_UP: 8.5 x 3.5 in laser check from "three-to-a-page" business stock.

Company top-left, bank top-centre, a LARGE check number top-right, then DATE and AMOUNT header
boxes (tinted header strip, value below), "PAY TO THE ORDER OF" line, legal line ending in
DOLLARS, memo, and one authorized-signature line. Clean guillotine cut.
"""

import numpy as np

from synthetic_checks.check_layout import SMALL_LABEL_EM_INCHES
from synthetic_checks.families.family_helpers import (
    bank_block, jitter, micr_slot, number_slots, payer_slots, ruled_fill, security_note, serial_slot, signature_slot,
)
from synthetic_checks.field_slots import FieldSlots, TextSlot
from synthetic_checks.stock.stock_canvas import StockCanvas


def header_box(canvas: StockCanvas, name: str, caption: str, x0f: float, x1f: float, y0f: float, y1f: float) -> TextSlot:
    """A box with a tinted caption strip on top; return the value slot below the strip."""
    x0, x1, y0, y1 = canvas.fx(x0f), canvas.fx(x1f), canvas.fy(y0f), canvas.fy(y1f)
    strip_bottom = y0 + (y1 - y0) * 0.36
    canvas.rectangle((x0, y0, x1, strip_bottom), plate="color", fill=110)
    canvas.rectangle((x0, y0, x1, y1))
    small_em = canvas.px(SMALL_LABEL_EM_INCHES * 1.2)
    canvas.label(name, caption, (x0 + x1) / 2, strip_bottom - (strip_bottom - y0) * 0.26, anchor="ms", em_px=small_em)
    pad = canvas.px(0.05)
    return TextSlot(x0 + pad, y1 - (y1 - strip_bottom) * 0.25, x1 - x0 - 2 * pad, canvas.px(0.15))


def draw_business_three_up(canvas: StockCanvas, rng: np.random.Generator) -> FieldSlots:
    """Draw the three-to-a-page laser business stock; return its fill slots."""
    variant = canvas.template.variant
    canvas.border()
    payer_name, payer_address = payer_slots(canvas, 0.035 + jitter(rng), 0.11 + jitter(rng), name_scale=1.1, max_width_f=0.33)
    bank_name, bank_city = bank_block(canvas, 0.40 + jitter(rng), 0.06 + jitter(rng), max_width_f=0.25)
    number, fraction = number_slots(canvas, 0.965, 0.14 + jitter(rng), em_scale=1.6)

    boxes_top, boxes_bottom = 0.22 + jitter(rng), 0.36
    date = header_box(canvas, "label_date", "DATE", 0.60 + jitter(rng), 0.77, boxes_top, boxes_bottom)
    amount = header_box(canvas, "label_amount", "AMOUNT", 0.79, 0.965, boxes_top, boxes_bottom)

    payee_y = 0.48 + jitter(rng)
    canvas.label("label_pay_to", "PAY TO THE ORDER OF", canvas.fx(0.035), canvas.fy(payee_y), max_width=canvas.fx(0.17))
    payee = ruled_fill(canvas, 0.215, 0.965, payee_y, handwriting=False)
    payee_address = None
    if variant == 2:
        address_em = canvas.px(0.085)
        payee_address = TextSlot(payee.x, payee.baseline_y + address_em * 1.6, canvas.fx(0.4), address_em)

    legal_y = 0.63 + jitter(rng) if variant != 2 else 0.70
    words = ruled_fill(canvas, 0.035, 0.84, legal_y, handwriting=False)
    canvas.label("label_dollars", "DOLLARS", canvas.fx(0.85), canvas.fy(legal_y), max_width=canvas.fx(0.1))
    if variant == 1:
        security_note(canvas, 0.85, legal_y + 0.06)

    memo_y = 0.77
    canvas.label("label_memo", "MEMO", canvas.fx(0.035), canvas.fy(memo_y), max_width=canvas.fx(0.07))
    memo = ruled_fill(canvas, 0.1, 0.45, memo_y)
    signature = signature_slot(canvas, 0.58, 0.965, memo_y - 0.01, caption="AUTHORIZED SIGNATURE")
    return FieldSlots(
        payer_name=payer_name, payer_address=payer_address, check_number=number, fractional_routing=fraction,
        bank_name=bank_name, bank_city=bank_city, date=date, payee=payee, payee_address=payee_address,
        amount_numeric=amount, amount_words=words, memo=memo, signature=signature,
        micr=micr_slot(canvas, 0.22 + jitter(rng)), serial=serial_slot(canvas, 0.035, 0.85),
    )
