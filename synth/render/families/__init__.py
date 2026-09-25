"""Layout families: one module per real-world check design, each `draw_<family>(canvas, rng) -> FieldSlots`.

To add a family: add it to `LayoutFamily` (+ its size in `FAMILY_SIZE_KIND`), write its module,
register it below, and add it to the per-family test in `synth/tests/test_layout_families.py`.
"""

from synth.render.check_layout import LayoutFamily
from synth.render.families.business_three_up import draw_business_three_up
from synth.render.families.business_voucher import draw_business_voucher
from synth.render.families.money_order import draw_money_order
from synth.render.families.personal_classic import draw_personal_classic
from synth.render.families.personal_date_box import draw_personal_date_box
from synth.render.families.personal_name_only import draw_personal_name_only

FAMILY_DRAWERS = {
    LayoutFamily.PERSONAL_CLASSIC: draw_personal_classic,
    LayoutFamily.PERSONAL_DATE_BOX: draw_personal_date_box,
    LayoutFamily.PERSONAL_NAME_ONLY: draw_personal_name_only,
    LayoutFamily.BUSINESS_VOUCHER: draw_business_voucher,
    LayoutFamily.BUSINESS_THREE_UP: draw_business_three_up,
    LayoutFamily.MONEY_ORDER: draw_money_order,
}

# Families whose checks are torn from a pad along one short edge (the rest are clean-cut).
PERFORATED_FAMILIES = {
    LayoutFamily.PERSONAL_CLASSIC,
    LayoutFamily.PERSONAL_DATE_BOX,
    LayoutFamily.PERSONAL_NAME_ONLY,
    LayoutFamily.MONEY_ORDER,
}
