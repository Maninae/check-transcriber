"""Drawing surface for pre-printed check stock: two offset plates plus the record of printed labels.

- `dark_plate` (key ink: labels, lines, borders, icons) and `color_plate` (decorative elements in
  the pattern colour) are coverage images ("L", 0..255). `stock_render` inks them onto the paper.
- Every pre-printed text is recorded in `preprinted` with its tight box. Labels always go on the
  dark plate, so plate misregistration (which moves only the colour plate) never moves a box.
- Family modules place things with fractions of width/height via `fx`/`fy` and physical sizes via `px`.
"""

import numpy as np
from PIL import Image, ImageDraw

from synth.render.check_fields import FieldLabel
from synth.render.check_layout import LABEL_EM_INCHES, LINE_WIDTH_INCHES, MICROPRINT_EM_INCHES, check_size_pixels, inches_to_px
from synth.render.check_templates import BorderKind, TemplateDesign
from synth.render.fonts import load_font
from synth.render.text_drawing import draw_printed_text

BORDER_INSET_INCHES = 0.06
GUILLOCHE_BAND_INCHES = 0.11
MICROPRINT_PHRASES = ["AUTHORIZED SIGNATURE", "ORIGINAL DOCUMENT", "SECURITY FEATURES INCLUDED"]
FULL_INK = 255


class StockCanvas:
    """The two plates and label list for one template's stock; see module docstring."""

    def __init__(self, template: TemplateDesign, dpi: int):
        self.template = template
        self.dpi = dpi
        self.width, self.height = check_size_pixels(template.size_kind, dpi)
        self.dark_plate = Image.new("L", (self.width, self.height), 0)
        self.color_plate = Image.new("L", (self.width, self.height), 0)
        self.dark_draw = ImageDraw.Draw(self.dark_plate)
        self.color_draw = ImageDraw.Draw(self.color_plate)
        self.preprinted: list[FieldLabel] = []
        self.line_width_px = inches_to_px(LINE_WIDTH_INCHES, dpi) + 1

    def px(self, inches: float) -> int:
        """Physical length to pixels at this stock's DPI."""
        return inches_to_px(inches, self.dpi)

    def fx(self, fraction: float) -> float:
        """x pixel of a fraction of the check width."""
        return fraction * self.width

    def fy(self, fraction: float) -> float:
        """y pixel of a fraction of the check height."""
        return fraction * self.height

    def wording(self, text: str) -> str:
        """Label text in the template's case convention."""
        return text.upper() if self.template.labels_uppercase else text

    def draw_for(self, plate: str) -> ImageDraw.ImageDraw:
        """The ImageDraw of the 'dark' or 'color' plate."""
        return self.dark_draw if plate == "dark" else self.color_draw

    def label(self, name: str, text: str, x: float, y: float, anchor: str = "ls", em_px: int | None = None,
              max_width: float | None = None, font_id: str | None = None, apply_case: bool = True) -> tuple[int, int, int, int]:
        """Print a dark-plate label, record it in `preprinted`, return its tight box."""
        shown = self.wording(text) if apply_case else text
        box = draw_printed_text(self.dark_draw, shown, font_id or self.template.label_font_id,
                                em_px or self.px(LABEL_EM_INCHES), (x, y), FULL_INK, anchor, max_width)
        self.preprinted.append(FieldLabel(name, shown, box, False))
        return box

    def line(self, x0: float, x1: float, y: float, plate: str = "dark", width_px: int | None = None) -> None:
        """A horizontal rule from x0 to x1 at y."""
        self.draw_for(plate).line((x0, y, x1, y), fill=FULL_INK, width=width_px or self.line_width_px)

    def rectangle(self, box: tuple[float, float, float, float], plate: str = "dark", width_px: int | None = None,
                  fill: int | None = None) -> None:
        """An outlined (or filled, with `fill` coverage 0..255) rectangle."""
        self.draw_for(plate).rectangle(box, outline=FULL_INK, width=width_px or self.line_width_px, fill=fill)

    def microprint_line(self, x0: float, x1: float, y: float, phrase: str) -> None:
        """A rule made of tiny repeated text; reads as a line at arm's length, as text under a loupe."""
        font = load_font("pt_sans_bold", max(5, self.px(MICROPRINT_EM_INCHES * 1.2)))
        repeated = (phrase + " ") * (int((x1 - x0) / max(1.0, font.getlength(phrase + " "))) + 1)
        strip = Image.new("L", (int(x1 - x0) + 1, font.size * 2), 0)
        ImageDraw.Draw(strip).text((0, font.size * 1.5), repeated, font=font, fill=FULL_INK, anchor="ls")
        self.dark_plate.paste(FULL_INK, (int(x0), int(y - font.size * 1.5)), mask=strip)

    def signature_line(self, x0: float, x1: float, y: float) -> None:
        """Signature rule: microprinted on stock that has it, else a plain line."""
        if self.template.microprint_signature_line:
            self.microprint_line(x0, x1, y, MICROPRINT_PHRASES[0])
        else:
            self.line(x0, x1, y)

    def padlock_icon(self, x: float, y: float, size: int) -> None:
        """The check-industry padlock mark ('security features, details on back') with top-left at (x, y)."""
        body_top = y + size * 0.42
        shackle_width = max(2, size // 9)
        self.dark_draw.arc((x + size * 0.18, y, x + size * 0.82, y + size * 0.8), 180, 360, fill=FULL_INK, width=shackle_width)
        self.dark_draw.line((x + size * 0.18 + shackle_width / 2, y + size * 0.4, x + size * 0.18 + shackle_width / 2, body_top), fill=FULL_INK, width=shackle_width)
        self.dark_draw.line((x + size * 0.82 - shackle_width / 2, y + size * 0.4, x + size * 0.82 - shackle_width / 2, body_top), fill=FULL_INK, width=shackle_width)
        self.dark_draw.rectangle((x, body_top, x + size, y + size * 1.1), fill=FULL_INK)
        self.dark_draw.ellipse((x + size * 0.42, body_top + size * 0.2, x + size * 0.58, body_top + size * 0.36), fill=0)

    def bank_logo(self, x: float, y: float, size: int, plate: str = "dark") -> float:
        """Draw the template's bank mark with top-left (x, y); return the x where the bank name should start."""
        shape = self.template.bank_logo_shape
        draw = self.draw_for(plate)
        if shape == "none":
            return x
        if shape == "circle":
            draw.ellipse((x, y, x + size, y + size), outline=FULL_INK, width=max(2, size // 10))
            draw.ellipse((x + size * 0.3, y + size * 0.3, x + size * 0.7, y + size * 0.7), fill=FULL_INK)
        elif shape == "square":
            draw.rectangle((x, y, x + size, y + size), fill=FULL_INK)
            draw.rectangle((x + size * 0.25, y + size * 0.25, x + size * 0.75, y + size * 0.75), fill=0)
        elif shape == "building":
            draw.polygon([(x, y + size * 0.3), (x + size / 2, y), (x + size, y + size * 0.3)], fill=FULL_INK)
            for column in range(4):
                column_x = x + size * (0.1 + column * 0.24)
                draw.rectangle((column_x, y + size * 0.38, column_x + size * 0.1, y + size * 0.85), fill=FULL_INK)
            draw.rectangle((x, y + size * 0.88, x + size, y + size), fill=FULL_INK)
        else:
            center_x, center_y = x + size / 2, y + size / 2
            draw.polygon([(center_x, y), (x + size, center_y), (center_x, y + size), (x, center_y)], fill=FULL_INK)
        return x + size * 1.3

    def border(self) -> None:
        """Frame the face per `template.border_kind` (the guilloche band is added by `stock_render`)."""
        inset = self.px(BORDER_INSET_INCHES)
        kind = self.template.border_kind
        right, bottom = self.width - inset, self.height - inset
        if kind == BorderKind.THIN:
            self.rectangle((inset, inset, right, bottom), width_px=3)
        elif kind == BorderKind.DOUBLE:
            self.rectangle((inset, inset, right, bottom), width_px=4)
            self.rectangle((inset + 10, inset + 10, right - 10, bottom - 10), width_px=2)
        elif kind == BorderKind.DASHED:
            dash = 18
            for x in range(inset, right, dash * 2):
                self.dark_draw.line((x, inset, x + dash, inset), fill=FULL_INK, width=3)
                self.dark_draw.line((x, bottom, x + dash, bottom), fill=FULL_INK, width=3)
            for y in range(inset, bottom, dash * 2):
                self.dark_draw.line((inset, y, inset, y + dash), fill=FULL_INK, width=3)
                self.dark_draw.line((right, y, right, y + dash), fill=FULL_INK, width=3)
        elif kind == BorderKind.MICROPRINT:
            self.microprint_line(inset, right, inset + 4, MICROPRINT_PHRASES[1])
            self.microprint_line(inset, right, bottom, MICROPRINT_PHRASES[1])


def guilloche_band_coverage(width: int, height: int, dpi: int) -> np.ndarray:
    """Coverage of a decorative wavy band along all four edges (colour plate)."""
    band = inches_to_px(GUILLOCHE_BAND_INCHES, dpi)
    inset = inches_to_px(BORDER_INSET_INCHES, dpi)
    y_grid, x_grid = np.mgrid[0:height, 0:width].astype(np.float32)
    waves = 0.5 + 0.5 * np.sin(2 * np.pi * (x_grid + y_grid) / 14 + 3 * np.sin(2 * np.pi * (x_grid - y_grid) / 90))
    distance_to_edge = np.minimum.reduce([x_grid - inset, y_grid - inset, width - inset - x_grid, height - inset - y_grid])
    in_band = ((distance_to_edge >= 0) & (distance_to_edge < band)).astype(np.float32)
    return np.clip((waves - 0.55) * 4, 0, 1) * in_band
