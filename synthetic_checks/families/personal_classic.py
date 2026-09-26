"""PERSONAL_CLASSIC: the standard US personal check.

Payer name and address top-left, number top-right, "Date ____" right of centre, "Pay to the
Order of ____ $[   ]", legal line ending in "Dollars", bank block, memo bottom-left, signature
bottom-right. One short edge is a torn perforation (checkbook).
"""

import numpy as np

from synthetic_checks.families.family_helpers import (
    amount_box, bank_block, jitter, micr_slot, number_slots, payer_slots, ruled_fill, security_note, serial_slot,
    signature_slot,
)
from synthetic_checks.field_slots import FieldSlots
from synthetic_checks.stock.stock_canvas import StockCanvas

MEMO_LABELS = ["Memo", "For", "Memo"]


def draw_personal_classic(canvas: StockCanvas, rng: np.random.Generator) -> FieldSlots:
    """Draw the classic personal stock; return its fill slots."""
    variant = canvas.template.variant
    canvas.border()
    payer_name, payer_address = payer_slots(canvas, 0.045 + jitter(rng), 0.13 + jitter(rng))
    number, fraction = number_slots(canvas, 0.955 + jitter(rng), 0.13 + jitter(rng))

    date_y = 0.315 + jitter(rng)
    canvas.label("label_date", "Date", canvas.fx(0.575 + jitter(rng)), canvas.fy(date_y))
    date = ruled_fill(canvas, 0.64, 0.88 + jitter(rng), date_y)

    payee_y = 0.465 + jitter(rng)
    canvas.label("label_pay_to", "Pay to the Order of", canvas.fx(0.045), canvas.fy(payee_y), max_width=canvas.fx(0.15))
    payee = ruled_fill(canvas, 0.205, 0.725, payee_y)
    amount = amount_box(canvas, 0.77, 0.955, 0.375 + jitter(rng), 0.49)

    legal_y = 0.605 + jitter(rng)
    words = ruled_fill(canvas, 0.045, 0.83, legal_y)
    canvas.label("label_dollars", "Dollars", canvas.fx(0.842), canvas.fy(legal_y), max_width=canvas.fx(0.13))
    security_note(canvas, 0.842, legal_y + 0.07)

    bank_name, bank_city = bank_block(canvas, 0.045, 0.645 + jitter(rng))
    memo_y = 0.815 + jitter(rng, 0.004)
    canvas.label("label_memo", MEMO_LABELS[variant], canvas.fx(0.045), canvas.fy(memo_y), max_width=canvas.fx(0.065))
    memo = ruled_fill(canvas, 0.115, 0.47, memo_y)
    signature = signature_slot(canvas, 0.55, 0.955, memo_y, caption="MP" if canvas.template.microprint_signature_line else None)
    return FieldSlots(
        payer_name=payer_name, payer_address=payer_address, check_number=number, fractional_routing=fraction,
        bank_name=bank_name, bank_city=bank_city, date=date, payee=payee, payee_address=None, amount_numeric=amount,
        amount_words=words, memo=memo, signature=signature, micr=micr_slot(canvas, 0.12 + jitter(rng)),
        serial=serial_slot(canvas, 0.045, memo_y + 0.1),
    )
