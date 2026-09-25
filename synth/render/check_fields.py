"""Data records for one rendered check: what was written on it, and where.

`CheckContent` is the fake data that goes onto the check before rendering.
`CheckLabel` is what the renderer returns: every field's text, its tight pixel box in
check coordinates, and whether it was drawn in a handwriting font.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum


class FieldName(str, Enum):
    """Every labeled field on a check. Values are the JSON keys."""

    PAYER_NAME = "payer_name"
    PAYER_ADDRESS = "payer_address"
    CHECK_NUMBER = "check_number"
    DATE = "date"
    PAYEE = "payee"
    AMOUNT_NUMERIC = "amount_numeric"
    AMOUNT_WORDS = "amount_words"
    BANK_NAME = "bank_name"
    MEMO = "memo"
    SIGNATURE = "signature"
    MICR = "micr"
    SERIAL = "serial"


# Fields that a person may fill in by hand; everything else is always pre-printed.
HANDWRITABLE_FIELDS: tuple[FieldName, ...] = (
    FieldName.DATE,
    FieldName.PAYEE,
    FieldName.AMOUNT_NUMERIC,
    FieldName.AMOUNT_WORDS,
    FieldName.MEMO,
)


@dataclass
class CheckContent:
    """The fake values to write on one check (all invented; see fake_data.py)."""

    payer_name: str
    payer_address_lines: list[str]
    check_number: str
    date_text: str
    date_iso: str
    payee_text: str
    payee_canonical: str
    amount_cents: int
    amount_numeric_text: str
    amount_words_text: str
    bank_name: str
    bank_city_line: str
    memo_text: str
    signature_text: str
    routing_number: str
    account_number: str
    micr_font_text: str      # characters as typed in GnuMICR (A/B/C/D are the symbols)
    micr_readable_text: str  # same line with Unicode E-13B symbols, for humans and OCR targets
    handwritten_fields: list[str]
    ink_rgb: tuple[int, int, int]
    serial: str | None = None


@dataclass
class FieldLabel:
    """One field's text and tight box, in check pixel coordinates (x0, y0, x1, y1)."""

    field_name: str
    text: str
    box: tuple[int, int, int, int]
    handwritten: bool


@dataclass
class CheckLabel:
    """Everything the renderer knows about one rendered check."""

    template_id: str
    size_kind: str
    width_px: int
    height_px: int
    dpi: int
    fields: list[FieldLabel]
    preprinted_text: list[FieldLabel] = field(default_factory=list)
    canonical: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Plain-JSON form."""
        return asdict(self)

    def field_by_name(self, field_name: FieldName) -> FieldLabel | None:
        """Return the labeled field with this name, or None when the check has none (e.g. empty memo)."""
        for field_label in self.fields:
            if field_label.field_name == field_name.value:
                return field_label
        return None
