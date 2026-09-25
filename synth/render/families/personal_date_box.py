"""PERSONAL_DATE_BOX: personal check with the date in a small box under the check number.

Payer top-left, bank name top-centre, number top-right with a boxed DATE field under it,
two-line "PAY TO THE / ORDER OF" label, amount box, legal line, memo and signature.
"""

import numpy as np

from synth.render.check_layout import SMALL_LABEL_EM_INCHES
from synth.render.field_slots import FieldSlots, TextSlot
from synth.render.families.family_helpers import (
    amount_box, bank_block, jitter, micr_slot, number_slots, payer_slots, ruled_fill, security_note, serial_slot,
    signature_slot,
)
from synth.render.stock_canvas import StockCanvas


def draw_date_box(canvas: StockCanvas, x0f: float, x1f: float, y0f: float, y1f: float, label_inside: bool) -> TextSlot:
    """A small outlined date box, labelled inside (top-left, tiny) or to its left; return the fill slot."""
    x0, x1, y0, y1 = canvas.fx(x0f), canvas.fx(x1f), canvas.fy(y0f), canvas.fy(y1f)
    canvas.rectangle((x0, y0, x1, y1))
    small_em = canvas.px(SMALL_LABEL_EM_INCHES)
    if label_inside:
        canvas.label("label_date", "Date", x0 + small_em * 0.4, y0 + small_em * 1.2, em_px=small_em)
    else:
        canvas.label("label_date", "Date", x0 - canvas.px(0.04), y1 - (y1 - y0) * 0.25, anchor="rs")
    pad = canvas.px(0.03)
    return TextSlot(x0 + pad, y1 - (y1 - y0) * 0.2, x1 - x0 - 2 * pad, canvas.px(0.16))


def draw_personal_date_box(canvas: StockCanvas, rng: np.random.Generator) -> FieldSlots:
    """Draw the date-box personal stock; return its fill slots."""
    variant = canvas.template.variant
    canvas.border()
    payer_name, payer_address = payer_slots(canvas, 0.045 + jitter(rng), 0.12 + jitter(rng), max_width_f=0.36)
    bank_name, bank_city = bank_block(canvas, 0.42 + jitter(rng), 0.06 + jitter(rng), max_width_f=0.3)
    number, fraction = number_slots(canvas, 0.955, 0.12 + jitter(rng))
    date = draw_date_box(canvas, 0.70 + jitter(rng), 0.955, 0.25, 0.36 + jitter(rng, 0.004), label_inside=variant != 1)

    payee_y = 0.50 + jitter(rng)
    label_em = canvas.px(0.07)
    canvas.label("label_pay_to", "Pay to the", canvas.fx(0.045), canvas.fy(payee_y) - label_em * 1.15, em_px=label_em)
    canvas.label("label_order_of", "Order of", canvas.fx(0.045), canvas.fy(payee_y), em_px=label_em)
    payee = ruled_fill(canvas, 0.155, 0.74, payee_y)
    amount = amount_box(canvas, 0.775, 0.955, 0.415 + jitter(rng), 0.525, dollar_inside=variant == 2)

    legal_y = 0.64 + jitter(rng)
    words = ruled_fill(canvas, 0.045, 0.82, legal_y)
    canvas.label("label_dollars", "Dollars", canvas.fx(0.832), canvas.fy(legal_y), max_width=canvas.fx(0.13))
    security_note(canvas, 0.832, legal_y + 0.065)

    memo_y = 0.80 + jitter(rng, 0.004)
    canvas.label("label_memo", "Memo", canvas.fx(0.045), canvas.fy(memo_y), max_width=canvas.fx(0.07))
    memo = ruled_fill(canvas, 0.12, 0.49, memo_y)
    signature = signature_slot(canvas, 0.54, 0.955, memo_y)
    return FieldSlots(
        payer_name=payer_name, payer_address=payer_address, check_number=number, fractional_routing=fraction,
        bank_name=bank_name, bank_city=bank_city, date=date, payee=payee, payee_address=None, amount_numeric=amount,
        amount_words=words, memo=memo, signature=signature, micr=micr_slot(canvas, 0.1 + jitter(rng)),
        serial=serial_slot(canvas, 0.045, memo_y + 0.095),
    )
