"""Stage 1: render one flat, top-down fake US check plus its field labels (contract C2).

`render_check(template, content, rng, ...)` returns an RGBA image at print resolution, alpha =
paper coverage (perforation nubs, worn corners), and a `CheckLabel` with every field's text and
tight box in check pixels. Layers, each multiplied into the one below (print_model.py):
1. blank stock: paper + offset pre-print, cached per template (`stock_render.render_blank_template`);
2. laser: payer block, number, bank, printed fill-ins, MICR, serial, in toner (`laser_printing`);
3. pen: handwritten fields via the Writer contract (C1), drawn on a transparent layer, then multiplied.
`simulate_print_texture=False` gives the clean version for print sheets (a real printer and paper
add their own toner and fibre): flat paper, crisp text, no misregistration, full rectangle alpha.
"""

import dataclasses

import numpy as np
from PIL import Image

from synth.render.check_fields import CheckContent, CheckLabel, FieldLabel, FieldName
from synth.render.check_layout import MICR_DIGIT_HEIGHT_INCHES, PRINTED_FILL_EM_INCHES, RENDER_DPI, inches_to_px
from synth.render.check_templates import TemplateDesign
from synth.render.fake_data import PRINTED_INK_RGB
from synth.render.fake_payees_and_banks import PAYEE_BY_NAME
from synth.render.families import PERFORATED_FAMILIES
from synth.render.field_slots import FieldSlots, TextSlot
from synth.render.fonts import load_font
from synth.render.handwriting_writer import Writer, draw_handwritten_field, sample_writer
from synth.render.laser_printing import LaserPrinter, micr_em_px_for_digit_height
from synth.render.paper_edges import make_paper_alpha
from synth.render.print_model import TONER_RGB, multiply_rgba_layer
from synth.render.stock_render import render_blank_template

PRINTED_FILL_MAX_EM_INCHES = 0.15
HANDWRITING_EM_RANGE_INCHES = (0.13, 0.18)
LEGAL_DASH_MIN_GAP_INCHES = 0.35       # draw the trailing dash only when this much line is left
LEGAL_DASH_PROBABILITY = 0.7
LEFT_PERFORATION_PROBABILITY = 0.85


class CheckFiller:
    """Writes one check's content onto a copy of its stock: laser text into `image`, pen strokes into `pen_layer`."""

    def __init__(self, template: TemplateDesign, content: CheckContent, image: np.ndarray, writer: Writer,
                 rng: np.random.Generator, dpi: int, textured: bool):
        self.template, self.content, self.writer, self.rng, self.dpi = template, content, writer, rng, dpi
        self.laser = LaserPrinter(image, rng, textured)
        self.pen_layer = Image.new("RGBA", (image.shape[1], image.shape[0]), (0, 0, 0, 0))
        self.fields: list[FieldLabel] = []
        self.extra_printed: list[FieldLabel] = []

    def add_field(self, field_name: FieldName, text: str, box, handwritten: bool) -> None:
        """Record a labelled field when something was actually inked."""
        if box is not None and box[2] > box[0] and box[3] > box[1]:
            self.fields.append(FieldLabel(field_name.value, text, tuple(int(v) for v in box), handwritten))

    def handwrite(self, text: str, slot: TextSlot, is_signature: bool = False):
        """Write `text` in the slot with this check's writer (C1); return the ink box or None."""
        low, high = (inches_to_px(bound, self.dpi) for bound in HANDWRITING_EM_RANGE_INCHES)
        em = slot.em_px if is_signature else int(np.clip(slot.em_px, low, high))
        return draw_handwritten_field(self.pen_layer, text, self.writer, em, (slot.x, slot.baseline_y), slot.max_width,
                                      self.rng, is_signature=is_signature)

    def print_fill(self, text: str, slot: TextSlot):
        """Laser-print a computer-filled value in the template's fill font."""
        em = min(slot.em_px, inches_to_px(PRINTED_FILL_MAX_EM_INCHES, self.dpi))
        return self.laser.print_in_slot(text, self.template.printed_fill_font_id, slot, PRINTED_INK_RGB, em)

    def fill(self, field_name: FieldName, text: str, slot: TextSlot | None) -> tuple[int, int, int, int] | None:
        """Handwrite or print one fill-in field per `content.handwritten_fields`."""
        if not text or slot is None:
            return None
        handwritten = field_name.value in self.content.handwritten_fields
        box = self.handwrite(text, slot) if handwritten else self.print_fill(text, slot)
        self.add_field(field_name, text, box, handwritten)
        return box

    def personalize(self, slots: FieldSlots) -> None:
        """Payer block, number, fractional routing and bank block (laser, or pen on money orders)."""
        content, template = self.content, self.template
        if slots.payer_is_handwritten:
            self.add_field(FieldName.PAYER_NAME, content.payer_name, self.handwrite(content.payer_name, slots.payer_name), True)
            address_text = ", ".join(content.payer_address_lines)
            self.add_field(FieldName.PAYER_ADDRESS, address_text, self.handwrite(address_text, slots.payer_address), True)
        else:
            ink = template.dark_ink_rgb
            name_box = self.laser.print_in_slot(content.payer_name, template.header_font_id, slots.payer_name, ink)
            self.add_field(FieldName.PAYER_NAME, content.payer_name, name_box, False)
            if slots.payer_address is not None:
                self.add_field(FieldName.PAYER_ADDRESS, "\n".join(content.payer_address_lines),
                               self.print_lines(content.payer_address_lines, template.body_font_id, slots.payer_address, ink), False)
        number_box = self.laser.print_in_slot(content.check_number, template.header_font_id, slots.check_number, template.dark_ink_rgb)
        self.add_field(FieldName.CHECK_NUMBER, content.check_number, number_box, False)
        if slots.fractional_routing is not None:
            fraction = fractional_routing_text(content.routing_number)
            self.record_printed("fractional_routing", fraction,
                                self.laser.print_in_slot(fraction, template.body_font_id, slots.fractional_routing, template.dark_ink_rgb))
        bank_box = self.laser.print_in_slot(content.bank_name, template.header_font_id, slots.bank_name, template.dark_ink_rgb)
        self.add_field(FieldName.BANK_NAME, content.bank_name, bank_box, False)
        if slots.bank_city is not None:
            self.record_printed("bank_city", content.bank_city_line,
                                self.laser.print_in_slot(content.bank_city_line, template.body_font_id, slots.bank_city, template.dark_ink_rgb))

    def print_lines(self, lines: list[str], font_id: str, first_line: TextSlot, ink) -> tuple[int, int, int, int] | None:
        """Print stacked lines from `first_line` down (1.3 em pitch); return the union box."""
        boxes = []
        for index, line_text in enumerate(lines):
            line_slot = TextSlot(first_line.x, first_line.baseline_y + index * first_line.em_px * 1.3, first_line.max_width,
                                 first_line.em_px, first_line.anchor)
            box = self.laser.print_in_slot(line_text, font_id, line_slot, ink)
            if box is not None:
                boxes.append(box)
        if not boxes:
            return None
        return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))

    def record_printed(self, name: str, text: str, box) -> None:
        """Keep unlabelled-but-printed text (fraction, bank city, payee address) in `preprinted_text`."""
        if box is not None:
            self.extra_printed.append(FieldLabel(name, text, box, False))

    def legal_line_dash(self, words_box, slot: TextSlot) -> None:
        """The long hand-drawn dash people add after the amount in words (not labelled)."""
        remaining = slot.x + slot.max_width - words_box[2]
        if remaining < inches_to_px(LEGAL_DASH_MIN_GAP_INCHES, self.dpi) or self.rng.random() > LEGAL_DASH_PROBABILITY:
            return
        em = int(np.clip(slot.em_px, *(inches_to_px(b, self.dpi) for b in HANDWRITING_EM_RANGE_INCHES)))
        dash_width = max(1.0, load_font(self.writer.font_id, em).getlength("-"))
        dash_count = max(2, int(remaining * 0.8 / dash_width))
        start = (words_box[2] + em * 0.3, slot.baseline_y - em * 0.18)
        draw_handwritten_field(self.pen_layer, "-" * dash_count, self.writer, em, start, remaining - em * 0.4, self.rng)

    def fill_all(self, slots: FieldSlots) -> None:
        """Every per-check element, in print order (laser first, then the pen)."""
        content = self.content
        self.personalize(slots)
        if slots.payee_address is not None:
            address = list(PAYEE_BY_NAME[content.payee_canonical].mailing_address_lines)
            self.record_printed("payee_address", "\n".join(address),
                                self.print_lines(address, self.template.printed_fill_font_id, slots.payee_address, PRINTED_INK_RGB))
        self.fill(FieldName.DATE, content.date_text, slots.date)
        self.fill(FieldName.PAYEE, content.payee_text, slots.payee)
        self.fill(FieldName.AMOUNT_NUMERIC, content.amount_numeric_text, slots.amount_numeric)
        words_box = self.fill(FieldName.AMOUNT_WORDS, content.amount_words_text, slots.amount_words)
        if words_box is not None and "amount_words" in content.handwritten_fields:
            self.legal_line_dash(words_box, slots.amount_words)
        self.fill(FieldName.MEMO, content.memo_text, slots.memo)
        self.add_field(FieldName.SIGNATURE, content.signature_text, self.handwrite(content.signature_text, slots.signature, True), True)

        micr_em = micr_em_px_for_digit_height(inches_to_px(MICR_DIGIT_HEIGHT_INCHES, self.dpi))
        self.add_field(FieldName.MICR, content.micr_readable_text,
                       self.laser.print_in_slot(content.micr_font_text, "gnu_micr", slots.micr, TONER_RGB, micr_em), False)
        if content.serial:
            self.add_field(FieldName.SERIAL, content.serial,
                           self.laser.print_in_slot(content.serial, "courier_prime", slots.serial, PRINTED_INK_RGB), False)

    def multiply_pen_layer(self) -> None:
        """Multiply the pen strokes into the image (only the region that has ink)."""
        ink_region = self.pen_layer.getchannel("A").getbbox()
        if ink_region is None:
            return
        x0, y0, x1, y1 = ink_region
        multiply_rgba_layer(self.laser.image, np.asarray(self.pen_layer.crop(ink_region)), x0, y0)


def fractional_routing_text(routing_number: str) -> str:
    """The small 'NN-NNNN/NNNN' fraction printed under the check number, derived from the fake routing digits."""
    return f"{int(routing_number[4:6]) + 10}-{routing_number[6:9]}{routing_number[1]}/{routing_number[:4]}"


def render_check(
    template: TemplateDesign,
    content: CheckContent,
    rng: np.random.Generator,
    dpi: int = RENDER_DPI,
    handwriting_font_ids: list[str] | None = None,
    signature_font_ids: list[str] | None = None,
    simulate_print_texture: bool = True,
) -> tuple[Image.Image, CheckLabel]:
    """Render `content` onto `template`. Returns (RGBA image, label with tight field boxes); see module docstring.

    Font pools restrict which handwriting and signature fonts the writer may use (split holdout);
    None means every registered font of that role.
    """
    stock = render_blank_template(template, dpi, simulate_print_texture)
    height, width = stock.rgb.shape[:2]
    image = stock.rgb.astype(np.float32) * (1.0 / 255.0)
    writer = sample_writer(rng, content.ink_rgb, handwriting_font_ids, signature_font_ids)
    filler = CheckFiller(template, content, image, writer, rng, dpi, simulate_print_texture)
    filler.fill_all(stock.slots)
    filler.multiply_pen_layer()

    perforated_side = None
    if simulate_print_texture and template.layout_family in PERFORATED_FAMILIES:
        perforated_side = "left" if rng.random() < LEFT_PERFORATION_PROBABILITY else "right"
    alpha = make_paper_alpha(width, height, dpi, perforated_side, rng) if simulate_print_texture \
        else np.full((height, width), 255, np.uint8)
    rgba = np.dstack([np.clip(image * 255.0 + 0.5, 0, 255).astype(np.uint8), alpha])

    label = CheckLabel(
        template_id=template.template_id,
        size_kind=template.size_kind.value,
        width_px=width,
        height_px=height,
        dpi=dpi,
        fields=filler.fields,
        preprinted_text=list(stock.preprinted) + filler.extra_printed,
        canonical={
            "layout_family": template.layout_family.value,
            "amount_cents": content.amount_cents,
            "date_iso": content.date_iso,
            "payee_canonical": content.payee_canonical,
            "check_number": content.check_number,
            "routing_number_fake_invalid_checksum": content.routing_number,
            "account_number_fake": content.account_number,
            "handwriting_font_id": writer.font_id,
            "signature_font_id": writer.signature_font_id,
            "pen_width_px": writer.pen_width_px,
            "writer": dataclasses.asdict(writer),  # pen type and hand habits (C1 Writer), JSON-safe floats
            "serial": content.serial,
            "print_texture": simulate_print_texture,
            "perforated_side": perforated_side,
        },
    )
    return Image.fromarray(rgba, "RGBA"), label
