"""Physical sizes and the standard US check layout, as constants.

Positions are fractions of the check's width (x) and height (y) so one layout serves both
the personal and business sizes; text sizes are in inches so type stays physically
plausible on both. Values follow the common US personal-check layout (spec section 5,
step 5): payer top-left, number top-right, date right of center, payee line, courtesy
amount box, legal line, bank block, memo bottom-left, signature bottom-right, MICR band.
"""

from dataclasses import dataclass
from enum import Enum

RENDER_DPI = 300


class CheckSizeKind(str, Enum):
    """The two physical check sizes we render."""

    PERSONAL = "personal"
    BUSINESS = "business"


CHECK_SIZE_INCHES: dict[CheckSizeKind, tuple[float, float]] = {
    CheckSizeKind.PERSONAL: (6.0, 2.75),
    CheckSizeKind.BUSINESS: (8.5, 3.5),
}


def check_size_pixels(size_kind: CheckSizeKind, dpi: int = RENDER_DPI) -> tuple[int, int]:
    """Width and height in pixels of a check of this kind at `dpi`."""
    width_inches, height_inches = CHECK_SIZE_INCHES[size_kind]
    return round(width_inches * dpi), round(height_inches * dpi)


@dataclass(frozen=True)
class CheckLayout:
    """Normalized anchor positions for every field (x as fraction of width, y of height)."""

    payer_block_x: float = 0.045
    payer_block_y: float = 0.075
    check_number_right_x: float = 0.955
    check_number_y: float = 0.075
    date_label_x: float = 0.575
    date_line_x0: float = 0.64
    date_line_x1: float = 0.88
    date_baseline_y: float = 0.315
    pay_label_x: float = 0.045
    payee_line_x0: float = 0.205
    payee_line_x1: float = 0.725
    payee_baseline_y: float = 0.465
    dollar_sign_x: float = 0.745
    amount_box_x0: float = 0.77
    amount_box_x1: float = 0.955
    amount_box_y0: float = 0.375
    amount_box_y1: float = 0.49
    legal_line_x0: float = 0.045
    legal_line_x1: float = 0.83
    legal_baseline_y: float = 0.605
    dollars_label_x: float = 0.842
    bank_block_x: float = 0.045
    bank_block_y: float = 0.645
    memo_label_x: float = 0.045
    memo_line_x0: float = 0.115
    memo_line_x1: float = 0.47
    memo_baseline_y: float = 0.815
    signature_line_x0: float = 0.55
    signature_line_x1: float = 0.955
    signature_baseline_y: float = 0.815
    micr_start_x: float = 0.12


# Physical text sizes (inches of font em size unless noted).
PAYER_NAME_EM_INCHES = 0.135
PAYER_ADDRESS_EM_INCHES = 0.095
CHECK_NUMBER_EM_INCHES = 0.13
LABEL_EM_INCHES = 0.08
BANK_NAME_EM_INCHES = 0.11
BANK_CITY_EM_INCHES = 0.075
PRINTED_FILL_EM_INCHES = 0.12
HANDWRITING_EM_INCHES = 0.18
SIGNATURE_EM_INCHES = 0.30
SERIAL_EM_INCHES = 0.075
MICR_DIGIT_HEIGHT_INCHES = 0.117       # E-13B character height (ANSI X9 / ISO 1004)
MICR_BASELINE_FROM_BOTTOM_INCHES = 0.19  # inside the 5/8 in clear band
LINE_WIDTH_INCHES = 0.006

# The MICR clear band: bottom 5/8 inch of every check.
MICR_CLEAR_BAND_INCHES = 0.625
