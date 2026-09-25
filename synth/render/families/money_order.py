"""MONEY_ORDER: a bank or retail money order, a common way to pay rent.

The issuer prints the name, serial number, date and amount (figures and words) at purchase;
the purchaser writes the payee, their own name and address, a memo, and signs. One short edge
is torn from the pad. `FieldSlots.payer_is_handwritten` tells the filler to write the payer block.
"""

import numpy as np

from synth.render.check_layout import BANK_NAME_EM_INCHES, SMALL_LABEL_EM_INCHES
from synth.render.field_slots import FieldSlots, TextSlot
from synth.render.families.family_helpers import jitter, micr_slot, ruled_fill, serial_slot, signature_slot
from synth.render.stock_canvas import StockCanvas

TITLES = ["MONEY ORDER", "PERSONAL MONEY ORDER", "MONEY ORDER"]
LIMIT_NOTES = ["NOT GOOD FOR MORE THAN ONE THOUSAND DOLLARS", "NOT VALID OVER $1,000.00", "NEGOTIABLE ONLY IN THE U.S."]


def draw_money_order(canvas: StockCanvas, rng: np.random.Generator) -> FieldSlots:
    """Draw the money-order stock; return its fill slots."""
    variant = canvas.template.variant
    canvas.border()
    issuer_em = canvas.px(BANK_NAME_EM_INCHES * 1.25)
    logo_x = canvas.bank_logo(canvas.fx(0.04), canvas.fy(0.07), int(issuer_em * 1.4), "color")
    bank_name = TextSlot(logo_x, canvas.fy(0.07) + issuer_em * 1.05, canvas.fx(0.4), issuer_em)
    canvas.label("label_title", TITLES[variant], canvas.fx(0.5 + jitter(rng)), canvas.fy(0.2), anchor="ms",
                 em_px=canvas.px(0.17), font_id="oswald", apply_case=False)
    small_em = canvas.px(SMALL_LABEL_EM_INCHES)
    canvas.label("label_serial_number", "SERIAL NUMBER", canvas.fx(0.955), canvas.fy(0.07), anchor="rs", em_px=small_em)
    number = TextSlot(canvas.fx(0.955), canvas.fy(0.15), canvas.fx(0.3), canvas.px(0.13), "rs")

    row_top, row_bottom = 0.25 + jitter(rng), 0.37
    canvas.rectangle((canvas.fx(0.04), canvas.fy(row_top), canvas.fx(0.955), canvas.fy(row_bottom)))
    canvas.dark_draw.line((canvas.fx(0.34), canvas.fy(row_top), canvas.fx(0.34), canvas.fy(row_bottom)), fill=255, width=canvas.line_width_px)
    caption_y = canvas.fy(row_top) + small_em * 1.3
    canvas.label("label_date", "DATE", canvas.fx(0.05), caption_y, em_px=small_em)
    canvas.label("label_amount", "AMOUNT", canvas.fx(0.35), caption_y, em_px=small_em)
    value_y = canvas.fy(row_bottom) - canvas.px(0.05)
    date = TextSlot(canvas.fx(0.05), value_y, canvas.fx(0.27), canvas.px(0.12))
    amount = TextSlot(canvas.fx(0.35), value_y, canvas.fx(0.58), canvas.px(0.14))
    words = TextSlot(canvas.fx(0.05), canvas.fy(0.47), canvas.fx(0.9), canvas.px(0.13))

    payee_y = 0.58 + jitter(rng)
    canvas.label("label_pay_to", "PAY TO THE ORDER OF", canvas.fx(0.04), canvas.fy(payee_y), max_width=canvas.fx(0.12))
    payee = ruled_fill(canvas, 0.17, 0.58, payee_y)
    signature = signature_slot(canvas, 0.62, 0.955, payee_y, caption="PURCHASER, SIGNER FOR DRAWER")

    from_y = 0.68 + jitter(rng, 0.004)
    canvas.label("label_from", "FROM", canvas.fx(0.04), canvas.fy(from_y), max_width=canvas.fx(0.1))
    payer_name = ruled_fill(canvas, 0.17, 0.58, from_y, em_scale=0.85)
    address_y = from_y + 0.095
    canvas.label("label_address", "ADDRESS", canvas.fx(0.04), canvas.fy(address_y), max_width=canvas.fx(0.12))
    payer_address = ruled_fill(canvas, 0.17, 0.58, address_y, em_scale=0.75)
    canvas.label("label_memo", "MEMO", canvas.fx(0.62), canvas.fy(from_y + 0.045), max_width=canvas.fx(0.07))
    memo = ruled_fill(canvas, 0.69, 0.955, from_y + 0.045, em_scale=0.85)
    canvas.label("label_limit_note", LIMIT_NOTES[variant], canvas.fx(0.04), canvas.fy(0.075) + issuer_em * 1.9,
                 em_px=small_em, max_width=canvas.fx(0.3), apply_case=False)
    return FieldSlots(
        payer_name=payer_name, payer_address=payer_address, check_number=number, fractional_routing=None,
        bank_name=bank_name, bank_city=None, date=date, payee=payee, payee_address=None, amount_numeric=amount,
        amount_words=words, memo=memo, signature=signature, micr=micr_slot(canvas, 0.1 + jitter(rng)),
        serial=serial_slot(canvas, 0.62, 0.79), payer_is_handwritten=True,
    )
