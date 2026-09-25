"""Stage 1: render one flat, top-down fake US check plus its field labels.

`render_check(template, content, rng)` returns an RGB image at print resolution and a
`CheckLabel` holding every field's text and tight box in check pixel coordinates.

- The blank stock (paper tint, security pattern, border, pre-printed labels and lines)
  depends only on the template, so it is rendered once per process and cached.
- Fill-ins (payer, number, bank, date, payee, amounts, memo, signature, MICR) are drawn
  per check; handwritten fields use one handwriting font per check (one writer).
- MICR uses GnuMICR (E-13B) with made-up digits; see fake_data.py for the invalid-checksum rule.
"""

import functools

import numpy as np
from PIL import Image, ImageDraw

from synth.render.check_fields import CheckContent, CheckLabel, FieldLabel, FieldName
from synth.render.check_layout import (
    BANK_CITY_EM_INCHES,
    BANK_NAME_EM_INCHES,
    CHECK_NUMBER_EM_INCHES,
    HANDWRITING_EM_INCHES,
    LABEL_EM_INCHES,
    LINE_WIDTH_INCHES,
    MICR_BASELINE_FROM_BOTTOM_INCHES,
    MICR_DIGIT_HEIGHT_INCHES,
    PAYER_ADDRESS_EM_INCHES,
    PAYER_NAME_EM_INCHES,
    PRINTED_FILL_EM_INCHES,
    RENDER_DPI,
    SERIAL_EM_INCHES,
    SIGNATURE_EM_INCHES,
    CheckSizeKind,
    check_size_pixels,
)
from synth.render.check_templates import BorderKind, TemplateDesign, darker_shade
from synth.render.fake_data import PRINTED_INK_RGB
from synth.render.fonts import FontRole, font_ids_with_role, load_font
from synth.render.security_pattern import make_security_pattern
from synth.render.text_drawing import draw_handwritten_text, draw_printed_text

PAPER_GRAIN_STD = 2.5
MICR_INK_RGB = (18, 18, 20)
BORDER_INSET_INCHES = 0.06
GUILLOCHE_BAND_INCHES = 0.11


def inches_to_px(inches: float, dpi: int) -> int:
    """Convert a physical length to pixels."""
    return max(1, int(round(inches * dpi)))


def preprinted_ink_rgb(template: TemplateDesign) -> tuple[int, int, int]:
    """Dark ink tied to the stock color, used for labels and lines."""
    return darker_shade(template.pattern_rgb, 0.4)


def draw_border(draw: ImageDraw.ImageDraw, template: TemplateDesign, width: int, height: int, dpi: int) -> None:
    """Frame the check face according to `template.border_kind`."""
    inset = inches_to_px(BORDER_INSET_INCHES, dpi)
    ink = preprinted_ink_rgb(template)
    box = (inset, inset, width - inset, height - inset)
    if template.border_kind == BorderKind.THIN:
        draw.rectangle(box, outline=ink, width=3)
    elif template.border_kind == BorderKind.DOUBLE:
        draw.rectangle(box, outline=ink, width=4)
        draw.rectangle((inset + 10, inset + 10, width - inset - 10, height - inset - 10), outline=ink, width=2)
    elif template.border_kind == BorderKind.DASHED:
        dash_length = 18
        for x in range(inset, width - inset, dash_length * 2):
            draw.line((x, inset, x + dash_length, inset), fill=ink, width=3)
            draw.line((x, height - inset, x + dash_length, height - inset), fill=ink, width=3)
        for y in range(inset, height - inset, dash_length * 2):
            draw.line((inset, y, inset, y + dash_length), fill=ink, width=3)
            draw.line((width - inset, y, width - inset, y + dash_length), fill=ink, width=3)


def guilloche_band_mask(width: int, height: int, dpi: int) -> np.ndarray:
    """Coverage map for a decorative wavy band along all four edges."""
    band = inches_to_px(GUILLOCHE_BAND_INCHES, dpi)
    inset = inches_to_px(BORDER_INSET_INCHES, dpi)
    y_grid, x_grid = np.mgrid[0:height, 0:width].astype(np.float32)
    waves = 0.5 + 0.5 * np.sin(2 * np.pi * (x_grid + y_grid) / 14 + 3 * np.sin(2 * np.pi * (x_grid - y_grid) / 90))
    distance_to_edge = np.minimum.reduce([x_grid - inset, y_grid - inset, width - inset - x_grid, height - inset - y_grid])
    in_band = ((distance_to_edge >= 0) & (distance_to_edge < band)).astype(np.float32)
    return (waves > 0.6).astype(np.float32) * in_band


@functools.lru_cache(maxsize=64)
def render_blank_template(template: TemplateDesign, dpi: int = RENDER_DPI) -> tuple[np.ndarray, tuple[FieldLabel, ...]]:
    """Blank check stock for `template` as an RGB uint8 array, plus the pre-printed label boxes."""
    width, height = check_size_pixels(template.size_kind, dpi)
    rng = np.random.default_rng(abs(hash(template.template_id)) % (2**32))
    coverage = make_security_pattern(template.pattern_kind, width, height, rng) * template.pattern_strength
    if template.border_kind == BorderKind.GUILLOCHE_BAND:
        coverage = np.maximum(coverage, guilloche_band_mask(width, height, dpi) * 0.8)
    tint = np.array(template.paper_tint_rgb, np.float32)
    pattern_color = np.array(template.pattern_rgb, np.float32)
    paper = tint * (1 - coverage[..., None]) + pattern_color * coverage[..., None]
    paper += rng.normal(0, PAPER_GRAIN_STD, (height, width, 1)).astype(np.float32)
    image = Image.fromarray(np.clip(paper, 0, 255).astype(np.uint8), "RGB")

    draw = ImageDraw.Draw(image)
    draw_border(draw, template, width, height, dpi)
    layout = template.layout
    ink = preprinted_ink_rgb(template)
    line_width = inches_to_px(LINE_WIDTH_INCHES, dpi) + 1
    label_px = inches_to_px(LABEL_EM_INCHES, dpi)

    def label(text: str) -> str:
        return text.upper() if template.labels_uppercase else text

    preprinted = []

    def add_label(field_name: str, text: str, x: float, y: float, max_width: float | None = None,
                  em_px: int = label_px, anchor: str = "ls") -> None:
        box = draw_printed_text(draw, text, template.label_font_id, em_px, (x, y), ink, anchor, max_width)
        preprinted.append(FieldLabel(field_name, text, box, False))

    def add_line(x0: float, x1: float, y: float) -> None:
        draw.line((x0 * width, y * height + 6, x1 * width, y * height + 6), fill=ink, width=line_width)

    add_label("label_date", label("Date"), layout.date_label_x * width, layout.date_baseline_y * height)
    add_line(layout.date_line_x0, layout.date_line_x1, layout.date_baseline_y)
    add_label("label_pay_to", label("Pay to the Order of"), layout.pay_label_x * width, layout.payee_baseline_y * height,
              max_width=(layout.payee_line_x0 - layout.pay_label_x) * width - 12)
    add_line(layout.payee_line_x0, layout.payee_line_x1, layout.payee_baseline_y)
    box_x0, box_x1 = layout.amount_box_x0 * width, layout.amount_box_x1 * width
    box_y0, box_y1 = layout.amount_box_y0 * height, layout.amount_box_y1 * height
    add_label("label_dollar_sign", "$", layout.dollar_sign_x * width, box_y1 - (box_y1 - box_y0) * 0.22,
              em_px=inches_to_px(LABEL_EM_INCHES * 1.8, dpi))
    if template.amount_box_outlined:
        draw.rectangle((box_x0, box_y0, box_x1, box_y1), outline=ink, width=line_width)
    else:
        draw.line((box_x0, box_y1, box_x1, box_y1), fill=ink, width=line_width)
    add_line(layout.legal_line_x0, layout.legal_line_x1, layout.legal_baseline_y)
    add_label("label_dollars", label("Dollars"), layout.dollars_label_x * width, layout.legal_baseline_y * height,
              max_width=(0.985 - layout.dollars_label_x) * width)
    add_label("label_security_note", "Security features included. Details on back.", layout.dollars_label_x * width,
              layout.legal_baseline_y * height + label_px * 1.1, max_width=(0.985 - layout.dollars_label_x) * width,
              em_px=int(label_px * 0.55))
    add_label("label_memo", label("Memo" if template.labels_uppercase or layout.memo_line_x0 > 0.11 else "For"),
              layout.memo_label_x * width, layout.memo_baseline_y * height,
              max_width=(layout.memo_line_x0 - layout.memo_label_x) * width - 8)
    add_line(layout.memo_line_x0, layout.memo_line_x1, layout.memo_baseline_y)
    add_line(layout.signature_line_x0, layout.signature_line_x1, layout.signature_baseline_y)
    if template.size_kind == CheckSizeKind.BUSINESS:
        add_label("label_authorized_signature", "AUTHORIZED SIGNATURE", layout.signature_line_x1 * width,
                  layout.signature_baseline_y * height + label_px * 1.3, em_px=int(label_px * 0.7), anchor="rs")
    return np.asarray(image).copy(), tuple(preprinted)


def micr_em_px_for_digit_height(digit_height_px: int) -> int:
    """GnuMICR em size whose digit '0' is `digit_height_px` tall."""
    reference_em = 200
    x0, y0, x1, y1 = load_font("gnu_micr", reference_em).getbbox("0")
    return max(8, int(round(reference_em * digit_height_px / (y1 - y0))))


def draw_bank_logo(draw: ImageDraw.ImageDraw, shape: str, x: float, y: float, size: int, ink: tuple[int, int, int]) -> float:
    """Draw a simple bank mark at (x, y) top-left; return the x where the bank name should start."""
    if shape == "none":
        return x
    box = (x, y, x + size, y + size)
    if shape == "circle":
        draw.ellipse(box, outline=ink, width=max(2, size // 10))
        draw.ellipse((x + size * 0.3, y + size * 0.3, x + size * 0.7, y + size * 0.7), fill=ink)
    elif shape == "square":
        draw.rectangle(box, fill=ink)
        draw.rectangle((x + size * 0.25, y + size * 0.25, x + size * 0.75, y + size * 0.75), fill=(245, 245, 245))
    else:
        center_x, center_y = x + size / 2, y + size / 2
        draw.polygon([(center_x, y), (x + size, center_y), (center_x, y + size), (x, center_y)], fill=ink)
    return x + size * 1.3


def fractional_routing_text(routing_number: str) -> str:
    """The small 'NN-NNNN/NNNN' fraction printed under the check number, derived from the fake routing digits."""
    return f"{int(routing_number[4:6]) + 10}-{routing_number[6:9]}{routing_number[1]}/{routing_number[:4]}"


def render_check(
    template: TemplateDesign,
    content: CheckContent,
    rng: np.random.Generator,
    dpi: int = RENDER_DPI,
) -> tuple[Image.Image, CheckLabel]:
    """Render `content` onto `template`. Returns (RGB image, label with tight field boxes)."""
    blank_pixels, preprinted = render_blank_template(template, dpi)
    height, width = blank_pixels.shape[:2]
    canvas = Image.fromarray(blank_pixels, "RGB").convert("RGBA")
    draw = ImageDraw.Draw(canvas)
    layout = template.layout
    printed_ink = PRINTED_INK_RGB
    label_ink = preprinted_ink_rgb(template)
    handwriting_font_id = rng.choice(font_ids_with_role(FontRole.HANDWRITING))
    signature_font_id = rng.choice(font_ids_with_role(FontRole.SIGNATURE))
    handwriting_em = inches_to_px(HANDWRITING_EM_INCHES, dpi)
    printed_fill_em = inches_to_px(PRINTED_FILL_EM_INCHES, dpi)
    pen_width_px = int(rng.choice([1, 3, 3, 5]))  # one pen per writer
    fields: list[FieldLabel] = []

    def add_field(field_name: FieldName, text: str, box, handwritten: bool) -> None:
        if box is not None and box[2] > box[0] and box[3] > box[1]:
            fields.append(FieldLabel(field_name.value, text, tuple(int(v) for v in box), handwritten))

    # Pre-printed personalization: payer block, check number, fraction, bank block.
    payer_x, payer_y = layout.payer_block_x * width, layout.payer_block_y * height
    name_box = draw_printed_text(draw, content.payer_name, template.header_font_id, inches_to_px(PAYER_NAME_EM_INCHES, dpi),
                                 (payer_x, payer_y), label_ink, "la", max_width_px=0.47 * width)
    add_field(FieldName.PAYER_NAME, content.payer_name, name_box, False)
    address_em = inches_to_px(PAYER_ADDRESS_EM_INCHES, dpi)
    line_top = name_box[3] + address_em * 0.25
    address_boxes = []
    for address_line in content.payer_address_lines:
        address_boxes.append(draw_printed_text(draw, address_line, template.body_font_id, address_em, (payer_x, line_top),
                                               label_ink, "la", max_width_px=0.47 * width))
        line_top = address_boxes[-1][3] + address_em * 0.2
    add_field(FieldName.PAYER_ADDRESS, "\n".join(content.payer_address_lines),
              (min(b[0] for b in address_boxes), address_boxes[0][1], max(b[2] for b in address_boxes), address_boxes[-1][3]), False)

    number_box = draw_printed_text(draw, content.check_number, template.header_font_id, inches_to_px(CHECK_NUMBER_EM_INCHES, dpi),
                                   (layout.check_number_right_x * width, layout.check_number_y * height), label_ink, "ra")
    add_field(FieldName.CHECK_NUMBER, content.check_number, number_box, False)
    fraction = fractional_routing_text(content.routing_number)
    fraction_box = draw_printed_text(draw, fraction, template.body_font_id, int(address_em * 0.8),
                                     (layout.check_number_right_x * width, number_box[3] + address_em * 0.4), label_ink, "ra")
    preprinted = preprinted + (FieldLabel("fractional_routing", fraction, fraction_box, False),)

    bank_x, bank_y = layout.bank_block_x * width, layout.bank_block_y * height
    logo_size = inches_to_px(BANK_NAME_EM_INCHES * 1.5, dpi)
    bank_text_x = draw_bank_logo(draw, template.bank_logo_shape, bank_x, bank_y, logo_size, label_ink)
    bank_box = draw_printed_text(draw, content.bank_name, template.header_font_id, inches_to_px(BANK_NAME_EM_INCHES, dpi),
                                 (bank_text_x, bank_y), label_ink, "la", max_width_px=0.42 * width)
    add_field(FieldName.BANK_NAME, content.bank_name, bank_box, False)
    draw_printed_text(draw, content.bank_city_line, template.body_font_id, inches_to_px(BANK_CITY_EM_INCHES, dpi),
                      (bank_text_x, bank_box[3] + 6), label_ink, "la", max_width_px=0.42 * width)

    # Fill-ins: handwritten or printed, one writer (font + ink) per check.
    def fill_in(field_name: FieldName, text: str, x0: float, x1: float, baseline_y: float) -> None:
        if not text:
            return
        if field_name.value in content.handwritten_fields:
            box = draw_handwritten_text(canvas, text, handwriting_font_id, handwriting_em, (x0, baseline_y),
                                        x1 - x0, content.ink_rgb, rng, pen_width_px=pen_width_px)
            add_field(field_name, text, box, True)
        else:
            box = draw_printed_text(draw, text, template.printed_fill_font_id, printed_fill_em, (x0, baseline_y),
                                    printed_ink, "ls", max_width_px=x1 - x0)
            add_field(field_name, text, box, False)

    lift = inches_to_px(0.03, dpi)
    fill_in(FieldName.DATE, content.date_text, (layout.date_line_x0 + 0.01) * width, layout.date_line_x1 * width,
            layout.date_baseline_y * height + lift * 0.5)
    fill_in(FieldName.PAYEE, content.payee_text, (layout.payee_line_x0 + 0.015) * width, layout.payee_line_x1 * width,
            layout.payee_baseline_y * height + lift * 0.5)
    box_height = (layout.amount_box_y1 - layout.amount_box_y0) * height
    fill_in(FieldName.AMOUNT_NUMERIC, content.amount_numeric_text, layout.amount_box_x0 * width + box_height * 0.25,
            layout.amount_box_x1 * width - box_height * 0.15, layout.amount_box_y1 * height - box_height * 0.22)
    fill_in(FieldName.AMOUNT_WORDS, content.amount_words_text, (layout.legal_line_x0 + 0.01) * width,
            layout.legal_line_x1 * width, layout.legal_baseline_y * height + lift * 0.5)
    fill_in(FieldName.MEMO, content.memo_text, (layout.memo_line_x0 + 0.01) * width, layout.memo_line_x1 * width,
            layout.memo_baseline_y * height + lift * 0.5)

    signature_box = draw_handwritten_text(canvas, content.signature_text, signature_font_id, inches_to_px(SIGNATURE_EM_INCHES, dpi),
                                          ((layout.signature_line_x0 + 0.03) * width, layout.signature_baseline_y * height + lift),
                                          (layout.signature_line_x1 - layout.signature_line_x0 - 0.05) * width,
                                          content.ink_rgb, rng, max_slant_degrees=6, pen_width_px=pen_width_px)
    add_field(FieldName.SIGNATURE, content.signature_text, signature_box, True)

    micr_em = micr_em_px_for_digit_height(inches_to_px(MICR_DIGIT_HEIGHT_INCHES, dpi))
    micr_baseline = height - inches_to_px(MICR_BASELINE_FROM_BOTTOM_INCHES, dpi)
    micr_box = draw_printed_text(draw, content.micr_font_text, "gnu_micr", micr_em, (layout.micr_start_x * width, micr_baseline),
                                 MICR_INK_RGB, "ls", max_width_px=0.86 * width)
    add_field(FieldName.MICR, content.micr_readable_text, micr_box, False)

    if content.serial:
        serial_box = draw_printed_text(draw, content.serial, "courier_prime", inches_to_px(SERIAL_EM_INCHES, dpi),
                                       (layout.memo_label_x * width, layout.memo_baseline_y * height + inches_to_px(0.1, dpi)),
                                       printed_ink, "ls")
        add_field(FieldName.SERIAL, content.serial, serial_box, False)

    label = CheckLabel(
        template_id=template.template_id,
        size_kind=template.size_kind.value,
        width_px=width,
        height_px=height,
        dpi=dpi,
        fields=fields,
        preprinted_text=list(preprinted),
        canonical={
            "amount_cents": content.amount_cents,
            "date_iso": content.date_iso,
            "payee_canonical": content.payee_canonical,
            "check_number": content.check_number,
            "routing_number_fake_invalid_checksum": content.routing_number,
            "account_number_fake": content.account_number,
            "handwriting_font_id": str(handwriting_font_id),
            "signature_font_id": str(signature_font_id),
            "pen_width_px": pen_width_px,
            "serial": content.serial,
        },
    )
    return canvas.convert("RGB"), label
