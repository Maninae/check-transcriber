"""PERSONAL_NAME_ONLY: personal check printed with the name only (no address), often on scenic stock.

The name sits top-left or top-centre in a larger face; the amount box is often underline-only;
bank block lower-left; "For" memo. The faded illustrated background is added by `stock_render`.
"""

import numpy as np

from synth.render.field_slots import FieldSlots
from synth.render.families.family_helpers import (
    amount_box, bank_block, jitter, micr_slot, number_slots, payer_slots, ruled_fill, serial_slot, signature_slot,
)
from synth.render.stock_canvas import StockCanvas


def draw_personal_name_only(canvas: StockCanvas, rng: np.random.Generator) -> FieldSlots:
    """Draw the name-only personal stock; return its fill slots."""
    variant = canvas.template.variant
    canvas.border()
    if variant == 1:
        payer_name, _ = payer_slots(canvas, 0.5, 0.15 + jitter(rng), anchor="ms", with_address=False, name_scale=1.35,
                                    max_width_f=0.5)
    else:
        payer_name, _ = payer_slots(canvas, 0.05 + jitter(rng), 0.16 + jitter(rng), with_address=False, name_scale=1.3)
    number, fraction = number_slots(canvas, 0.95, 0.14 + jitter(rng))

    date_y = 0.30 + jitter(rng)
    canvas.label("label_date", "Date", canvas.fx(0.60 + jitter(rng)), canvas.fy(date_y))
    date = ruled_fill(canvas, 0.665, 0.95, date_y)

    payee_y = 0.47 + jitter(rng)
    canvas.label("label_pay_to", "Pay to the Order of", canvas.fx(0.05), canvas.fy(payee_y), max_width=canvas.fx(0.15))
    payee = ruled_fill(canvas, 0.21, 0.73, payee_y)
    amount = amount_box(canvas, 0.78, 0.95, 0.38 + jitter(rng), 0.49, dollar_inside=variant == 2)

    legal_y = 0.61 + jitter(rng)
    words = ruled_fill(canvas, 0.05, 0.85, legal_y)
    canvas.label("label_dollars", "Dollars", canvas.fx(0.86), canvas.fy(legal_y), max_width=canvas.fx(0.11))

    bank_name, bank_city = bank_block(canvas, 0.05, 0.655 + jitter(rng), max_width_f=0.36)
    memo_y = 0.82 + jitter(rng, 0.004)
    canvas.label("label_memo", "For", canvas.fx(0.05), canvas.fy(memo_y), max_width=canvas.fx(0.05))
    memo = ruled_fill(canvas, 0.1, 0.46, memo_y)
    signature = signature_slot(canvas, 0.53, 0.95, memo_y)
    return FieldSlots(
        payer_name=payer_name, payer_address=None, check_number=number, fractional_routing=fraction,
        bank_name=bank_name, bank_city=bank_city, date=date, payee=payee, payee_address=None, amount_numeric=amount,
        amount_words=words, memo=memo, signature=signature, micr=micr_slot(canvas, 0.11 + jitter(rng)),
        serial=serial_slot(canvas, 0.05, memo_y + 0.095),
    )
