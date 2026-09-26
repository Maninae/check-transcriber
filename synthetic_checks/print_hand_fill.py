"""Hand-fill print mode: the prompt strip under each check, writer ids, and the check-plus-strip tile.

In hand-fill mode every field a person writes is left blank on the printed check (see
`render_check(..., leave_handwriting_blank=True)`) and a strip printed just under the check, outside
the cut line, says exactly what to write in each blank. The label for a hand-filled field is the
strip's text for it, character for character, so a real writer's photo keeps an exact answer key.

- Strip items run in reading order (`HAND_FILL_PROMPT_ORDER`) as `Label: value`, packed onto lines
  between ` | ` separators, never split inside an item.
- The strip has a fixed height (`PROMPT_STRIP_HEIGHT_PX`, sized so three personal checks still fit
  a page); text shrinks from `PROMPT_STRIP_EM_PX` towards `PROMPT_STRIP_MIN_EM_PX` until it fits.
- Writers are letters (A, B, C...) appended to the serial: `S-0007-B`.
"""

import string
from dataclasses import dataclass

from PIL import Image, ImageDraw

from synthetic_checks.check_fields import CheckLabel, FieldName
from synthetic_checks.check_layout import RENDER_DPI
from synthetic_checks.fonts.font_registry import load_font

PROMPT_STRIP_HEIGHT_PX = int(0.37 * RENDER_DPI)
PROMPT_STRIP_TOP_PAD_PX = int(0.07 * RENDER_DPI)     # clear of the dashed cut line
PROMPT_STRIP_SIDE_INSET_PX = int(0.12 * RENDER_DPI)  # clear of the corner cut marks at the check's edges
PROMPT_STRIP_EM_PX = int(0.105 * RENDER_DPI)          # ~7.5 pt
PROMPT_STRIP_MIN_EM_PX = int(0.08 * RENDER_DPI)       # ~5.8 pt, the smallest we will print
PROMPT_STRIP_LINE_PITCH_EM = 1.3
PROMPT_LABEL_FONT_ID = "libre_franklin_bold"
PROMPT_VALUE_FONT_ID = "libre_franklin"
PROMPT_TEXT_RGB = (20, 20, 20)
PROMPT_SEPARATOR_RGB = (150, 150, 150)
PROMPT_ITEM_SEPARATOR = "  |  "
MAX_WRITERS = len(string.ascii_uppercase)

# Reading order of the blanks, and the word printed before each value on the strip.
HAND_FILL_PROMPT_ORDER: tuple[FieldName, ...] = (
    FieldName.DATE,
    FieldName.PAYEE,
    FieldName.AMOUNT_NUMERIC,
    FieldName.AMOUNT_WORDS,
    FieldName.PAYER_NAME,
    FieldName.PAYER_ADDRESS,
    FieldName.MEMO,
    FieldName.SIGNATURE,
)
PROMPT_LABEL_BY_FIELD: dict[FieldName, str] = {
    FieldName.DATE: "Date",
    FieldName.PAYEE: "Pay to",
    FieldName.AMOUNT_NUMERIC: "Amount",
    FieldName.AMOUNT_WORDS: "In words",
    FieldName.PAYER_NAME: "From",
    FieldName.PAYER_ADDRESS: "Address",
    FieldName.MEMO: "Memo",
    FieldName.SIGNATURE: "Sign",
}


@dataclass(frozen=True)
class PromptItem:
    """One blank to fill: which field, the word shown before it, and the exact text to write."""

    field_name: FieldName
    label: str
    value: str


def writer_letter(writer_index: int) -> str:
    """0 -> 'A', 1 -> 'B', ..."""
    if not 0 <= writer_index < MAX_WRITERS:
        raise ValueError(f"writer index {writer_index} outside 0..{MAX_WRITERS - 1}")
    return string.ascii_uppercase[writer_index]


def hand_fill_serial(serial_number: int, letter: str) -> str:
    """The printed serial of one writer's copy of a check, e.g. S-0007-B."""
    return f"S-{serial_number:04d}-{letter}"


def hand_fill_prompt_items(label: CheckLabel) -> list[PromptItem]:
    """Every field the writer fills on this check, in reading order, with its exact text."""
    blank_fields = {field.field_name: field.text for field in label.fields if field.handwritten}
    return [PromptItem(field_name, PROMPT_LABEL_BY_FIELD[field_name], blank_fields[field_name.value])
            for field_name in HAND_FILL_PROMPT_ORDER if field_name.value in blank_fields]


def pack_items_into_lines(items: list[PromptItem], em_px: int, line_width_px: int) -> list[list[PromptItem]]:
    """Greedy line fill at this type size; an item never breaks across lines."""
    label_font, value_font = load_font(PROMPT_LABEL_FONT_ID, em_px), load_font(PROMPT_VALUE_FONT_ID, em_px)
    separator_width = value_font.getlength(PROMPT_ITEM_SEPARATOR)
    lines, current_line, current_width = [], [], 0.0
    for item in items:
        item_width = label_font.getlength(f"{item.label}: ") + value_font.getlength(item.value)
        needed = item_width + (separator_width if current_line else 0.0)
        if current_line and current_width + needed > line_width_px:
            lines.append(current_line)
            current_line, current_width = [], 0.0
            needed = item_width
        current_line.append(item)
        current_width += needed
    if current_line:
        lines.append(current_line)
    return lines


def prompt_strip_text(lines: list[list[PromptItem]]) -> str:
    """The strip exactly as printed: items joined by the separator, one text line per printed line."""
    return "\n".join(PROMPT_ITEM_SEPARATOR.join(f"{item.label}: {item.value}" for item in line) for line in lines)


def fit_prompt_lines(items: list[PromptItem], strip_width_px: int) -> tuple[int, list[list[PromptItem]]]:
    """Largest type size (down to the floor) whose packed lines fit the strip height."""
    line_width = strip_width_px - 2 * PROMPT_STRIP_SIDE_INSET_PX
    for em_px in range(PROMPT_STRIP_EM_PX, PROMPT_STRIP_MIN_EM_PX - 1, -1):
        lines = pack_items_into_lines(items, em_px, line_width)
        widest = max(line_pixel_width(line, em_px) for line in lines)
        if len(lines) * round(em_px * PROMPT_STRIP_LINE_PITCH_EM) <= PROMPT_STRIP_HEIGHT_PX - PROMPT_STRIP_TOP_PAD_PX \
                and widest <= line_width:
            return em_px, lines
    raise ValueError(f"prompt strip does not fit {strip_width_px} px even at {PROMPT_STRIP_MIN_EM_PX} px: {items}")


def line_pixel_width(line: list[PromptItem], em_px: int) -> float:
    """Printed width of one strip line."""
    label_font, value_font = load_font(PROMPT_LABEL_FONT_ID, em_px), load_font(PROMPT_VALUE_FONT_ID, em_px)
    return sum(label_font.getlength(f"{item.label}: ") + value_font.getlength(item.value) for item in line) \
        + value_font.getlength(PROMPT_ITEM_SEPARATOR) * (len(line) - 1)


def render_prompt_strip(items: list[PromptItem], strip_width_px: int) -> tuple[Image.Image, str]:
    """White strip image (check width x strip height) and its text exactly as printed."""
    em_px, lines = fit_prompt_lines(items, strip_width_px)
    strip = Image.new("RGB", (strip_width_px, PROMPT_STRIP_HEIGHT_PX), "white")
    draw = ImageDraw.Draw(strip)
    label_font, value_font = load_font(PROMPT_LABEL_FONT_ID, em_px), load_font(PROMPT_VALUE_FONT_ID, em_px)
    pitch = round(em_px * PROMPT_STRIP_LINE_PITCH_EM)
    for line_index, line in enumerate(lines):
        x, baseline = float(PROMPT_STRIP_SIDE_INSET_PX), PROMPT_STRIP_TOP_PAD_PX + em_px + line_index * pitch
        for item_index, item in enumerate(line):
            if item_index:
                draw.text((x, baseline), PROMPT_ITEM_SEPARATOR, fill=PROMPT_SEPARATOR_RGB, font=value_font, anchor="ls")
                x += value_font.getlength(PROMPT_ITEM_SEPARATOR)
            draw.text((x, baseline), f"{item.label}: ", fill=PROMPT_TEXT_RGB, font=label_font, anchor="ls")
            x += label_font.getlength(f"{item.label}: ")
            draw.text((x, baseline), item.value, fill=PROMPT_TEXT_RGB, font=value_font, anchor="ls")
            x += value_font.getlength(item.value)
    return strip, prompt_strip_text(lines)


def check_with_prompt_strip(check_image: Image.Image, strip: Image.Image) -> Image.Image:
    """The tile placed on the page: check on top, its strip directly under it (check frame, before any rotation)."""
    tile = Image.new("RGB", (check_image.width, check_image.height + strip.height), "white")
    tile.paste(check_image, (0, 0))
    tile.paste(strip, (0, check_image.height))
    return tile
