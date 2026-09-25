"""Physical sizes, layout families and physical type sizes for rendered checks.

Positions inside a check are decided by the layout family modules (`synth/render/families/`);
this leaf module holds only what every family shares: the check sizes, the DPI, the family
enum, and text sizes in inches so type stays physically plausible at any DPI.
"""

from enum import Enum

RENDER_DPI = 300


class CheckSizeKind(str, Enum):
    """The physical check sizes we render."""

    PERSONAL = "personal"
    BUSINESS = "business"
    MONEY_ORDER = "money_order"


CHECK_SIZE_INCHES: dict[CheckSizeKind, tuple[float, float]] = {
    CheckSizeKind.PERSONAL: (6.0, 2.75),
    CheckSizeKind.BUSINESS: (8.5, 3.5),
    CheckSizeKind.MONEY_ORDER: (7.0, 3.125),
}


class LayoutFamily(str, Enum):
    """A coherent real-world check design; every family carries the same labelled field set."""

    PERSONAL_CLASSIC = "personal_classic"          # payer top-left, date mid-right on a line
    PERSONAL_DATE_BOX = "personal_date_box"        # date in a small box under the number, top right
    PERSONAL_NAME_ONLY = "personal_name_only"      # name only (no address), scenic or illustrated background
    BUSINESS_VOUCHER = "business_voucher"          # PAY line first, payee address block, date/amount table, two signatures
    BUSINESS_THREE_UP = "business_three_up"        # laser "three-to-a-page" stock: big number, DATE / AMOUNT header boxes
    MONEY_ORDER = "money_order"                    # issuer-printed amount and number, purchaser writes the rest


FAMILY_SIZE_KIND: dict[LayoutFamily, CheckSizeKind] = {
    LayoutFamily.PERSONAL_CLASSIC: CheckSizeKind.PERSONAL,
    LayoutFamily.PERSONAL_DATE_BOX: CheckSizeKind.PERSONAL,
    LayoutFamily.PERSONAL_NAME_ONLY: CheckSizeKind.PERSONAL,
    LayoutFamily.BUSINESS_VOUCHER: CheckSizeKind.BUSINESS,
    LayoutFamily.BUSINESS_THREE_UP: CheckSizeKind.BUSINESS,
    LayoutFamily.MONEY_ORDER: CheckSizeKind.MONEY_ORDER,
}


def check_size_pixels(size_kind: CheckSizeKind, dpi: int = RENDER_DPI) -> tuple[int, int]:
    """Width and height in pixels of a check of this kind at `dpi`."""
    width_inches, height_inches = CHECK_SIZE_INCHES[size_kind]
    return round(width_inches * dpi), round(height_inches * dpi)


def inches_to_px(inches: float, dpi: int) -> int:
    """Convert a physical length to whole pixels (at least 1)."""
    return max(1, int(round(inches * dpi)))


# Physical text sizes (inches of font em size unless noted).
PAYER_NAME_EM_INCHES = 0.135
PAYER_ADDRESS_EM_INCHES = 0.095
CHECK_NUMBER_EM_INCHES = 0.13
LABEL_EM_INCHES = 0.08
SMALL_LABEL_EM_INCHES = 0.055
BANK_NAME_EM_INCHES = 0.11
BANK_CITY_EM_INCHES = 0.075
PRINTED_FILL_EM_INCHES = 0.12
HANDWRITING_EM_INCHES = 0.18
SIGNATURE_EM_INCHES = 0.30
SERIAL_EM_INCHES = 0.075
MICROPRINT_EM_INCHES = 0.018           # real microprint is ~0.01 in; one pixel short of illegible at 300 dpi
MICR_DIGIT_HEIGHT_INCHES = 0.117       # E-13B character height (ANSI X9 / ISO 1004)
MICR_BASELINE_FROM_BOTTOM_INCHES = 0.19  # inside the 5/8 in clear band
LINE_WIDTH_INCHES = 0.006
MICR_CLEAR_BAND_INCHES = 0.625         # bottom 5/8 inch of every check stays free of pre-print
