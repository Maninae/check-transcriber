"""Where each per-check value goes on a given stock: the hand-off from a layout family to the filler.

A layout family draws the pre-printed stock and returns a `FieldSlots`; `render_check` then
writes (laser or pen) each value into its slot. All numbers are check pixels at the stock's DPI.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TextSlot:
    """One place to put text: an anchor point, the room available, and the type size.

    - `anchor` is a Pillow anchor for printed text ("ls" left-baseline, "rs" right-baseline, "ms" centered).
    - Handwriting always starts at (x, baseline_y) and runs right, so hand-fillable slots use "ls".
    - `max_width` bounds printed and handwritten text alike; text shrinks to fit.
    """

    x: float
    baseline_y: float
    max_width: float
    em_px: int
    anchor: str = "ls"


@dataclass(frozen=True)
class FieldSlots:
    """Every fill-in position on one stock. Optional slots are None when the family has no such field."""

    payer_name: TextSlot
    payer_address: TextSlot | None       # first address line; later lines step down by 1.25 em
    check_number: TextSlot
    fractional_routing: TextSlot | None
    bank_name: TextSlot
    bank_city: TextSlot | None
    date: TextSlot
    payee: TextSlot
    payee_address: TextSlot | None       # printed mailing block under the payee (business stock)
    amount_numeric: TextSlot
    amount_words: TextSlot
    memo: TextSlot | None
    signature: TextSlot
    micr: TextSlot
    serial: TextSlot
    payer_is_handwritten: bool = False   # money orders: the purchaser writes name and address
