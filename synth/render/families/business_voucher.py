"""BUSINESS_VOUCHER: 8.5 x 3.5 in business (wallet/voucher) check.

Company block top-left, bank block top-centre, number top-right; the amount in words comes
FIRST on a "PAY" line; then "TO THE ORDER OF" with the payee and a printed mailing block; a
boxed DATE | AMOUNT table on the right; "VOID AFTER 90 DAYS"; two signature lines. Clean cut.
"""

import numpy as np

from synth.render.check_layout import SMALL_LABEL_EM_INCHES
from synth.render.field_slots import FieldSlots, TextSlot
from synth.render.families.family_helpers import (
    bank_block, jitter, micr_slot, number_slots, payer_slots, ruled_fill, serial_slot, signature_slot,
)
from synth.render.stock_canvas import StockCanvas

VOID_NOTES = ["VOID AFTER 90 DAYS", "VOID AFTER 90 DAYS", "VOID IF NOT CASHED WITHIN 90 DAYS"]


def draw_date_amount_table(canvas: StockCanvas, x0f: float, x1f: float, y0f: float, y1f: float,
                           split_f: float) -> tuple[TextSlot, TextSlot]:
    """Two-column boxed table with DATE and AMOUNT headers; return (date slot, amount slot)."""
    x0, x1, y0, y1, split = canvas.fx(x0f), canvas.fx(x1f), canvas.fy(y0f), canvas.fy(y1f), canvas.fx(split_f)
    header_bottom = y0 + (y1 - y0) * 0.38
    canvas.rectangle((x0, y0, x1, header_bottom), plate="color", fill=90)  # tinted header strip
    canvas.rectangle((x0, y0, x1, y1))
    canvas.line(x0, x1, header_bottom)
    canvas.dark_draw.line((split, y0, split, y1), fill=255, width=canvas.line_width_px)
    small_em = canvas.px(SMALL_LABEL_EM_INCHES * 1.2)
    header_baseline = header_bottom - (header_bottom - y0) * 0.28
    canvas.label("label_date", "DATE", (x0 + split) / 2, header_baseline, anchor="ms", em_px=small_em)
    canvas.label("label_amount", "AMOUNT", (split + x1) / 2, header_baseline, anchor="ms", em_px=small_em)
    value_baseline = y1 - (y1 - header_bottom) * 0.25
    pad = canvas.px(0.04)
    value_em = canvas.px(0.15)
    return (TextSlot(x0 + pad, value_baseline, split - x0 - 2 * pad, value_em),
            TextSlot(split + pad, value_baseline, x1 - split - 2 * pad, value_em))


def draw_business_voucher(canvas: StockCanvas, rng: np.random.Generator) -> FieldSlots:
    """Draw the voucher business stock; return its fill slots."""
    variant = canvas.template.variant
    canvas.border()
    payer_name, payer_address = payer_slots(canvas, 0.04 + jitter(rng), 0.11 + jitter(rng), name_scale=1.1, max_width_f=0.34)
    bank_name, bank_city = bank_block(canvas, 0.42 + jitter(rng), 0.06 + jitter(rng), max_width_f=0.28)
    number, fraction = number_slots(canvas, 0.96, 0.12 + jitter(rng), em_scale=1.2)

    pay_y = 0.40 + jitter(rng)
    canvas.label("label_pay", "PAY", canvas.fx(0.04), canvas.fy(pay_y))
    words = ruled_fill(canvas, 0.085, 0.66, pay_y, handwriting=False, draw_rule=variant != 0)
    date, amount = draw_date_amount_table(canvas, 0.69 + jitter(rng), 0.96, 0.30 + jitter(rng), 0.46, 0.80 + jitter(rng))
    canvas.label("label_void_note", VOID_NOTES[variant], canvas.fx(0.96), canvas.fy(0.52),
                 anchor="rs", em_px=canvas.px(SMALL_LABEL_EM_INCHES), apply_case=False)

    order_y = 0.56 + jitter(rng)
    label_em = canvas.px(0.065)
    canvas.label("label_to_the", "TO THE", canvas.fx(0.04), canvas.fy(order_y) - label_em * 1.15, em_px=label_em)
    canvas.label("label_order_of", "ORDER OF", canvas.fx(0.04), canvas.fy(order_y), em_px=label_em)
    payee = ruled_fill(canvas, 0.12, 0.55, order_y, handwriting=False, draw_rule=False)
    address_em = canvas.px(0.09)
    payee_address = TextSlot(payee.x, payee.baseline_y + address_em * 1.45, payee.max_width, address_em)

    first_signature_y = 0.62 + jitter(rng, 0.004)
    canvas.signature_line(canvas.fx(0.6), canvas.fx(0.96), canvas.fy(first_signature_y) + canvas.px(0.02))
    signature = signature_slot(canvas, 0.6, 0.96, 0.74, caption="AUTHORIZED SIGNATURE")
    memo_y = 0.79
    canvas.label("label_memo", "MEMO", canvas.fx(0.04), canvas.fy(memo_y), max_width=canvas.fx(0.07))
    memo = ruled_fill(canvas, 0.11, 0.45, memo_y)
    return FieldSlots(
        payer_name=payer_name, payer_address=payer_address, check_number=number, fractional_routing=fraction,
        bank_name=bank_name, bank_city=bank_city, date=date, payee=payee, payee_address=payee_address,
        amount_numeric=amount, amount_words=words, memo=memo, signature=signature,
        micr=micr_slot(canvas, 0.2 + jitter(rng)), serial=serial_slot(canvas, 0.04, 0.86),
    )
