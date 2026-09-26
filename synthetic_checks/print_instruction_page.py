"""Hand-fill print mode: the one-page instructions that open each writer's stack.

Written for a non-technical person holding a pen. What the eye should land on first: the writer
letter (so stacks never get mixed up), then the three numbered steps in order (fill in, cut out,
photograph), with a before/after example check between step 1 and step 2.

- All wording lives in the constants below; nothing names a real person.
- The example pair is passed in (rendered by the caller from the same pools), so this module only lays out;
  the caller also draws the page footer.
"""

from PIL import Image, ImageDraw, ImageFont

from synthetic_checks.check_layout import RENDER_DPI
from synthetic_checks.fonts.font_registry import load_font
from synthetic_checks.print_page_layout import LETTER_SIZE_PX

PAGE_MARGIN_PX = int(0.75 * RENDER_DPI)
TITLE_EM_PX = int(0.30 * RENDER_DPI)
SUBTITLE_EM_PX = int(0.15 * RENDER_DPI)
STEP_HEADING_EM_PX = int(0.20 * RENDER_DPI)
BODY_EM_PX = int(0.155 * RENDER_DPI)          # ~11 pt
CAPTION_EM_PX = int(0.12 * RENDER_DPI)
BADGE_LETTER_EM_PX = int(0.62 * RENDER_DPI)
BADGE_SIZE_PX = int(1.0 * RENDER_DPI)
BULLET_INDENT_PX = int(0.30 * RENDER_DPI)
LINE_PITCH_EM = 1.35
PARAGRAPH_GAP_PX = int(0.07 * RENDER_DPI)
SECTION_GAP_PX = int(0.24 * RENDER_DPI)
EXAMPLE_GAP_PX = int(0.30 * RENDER_DPI)
REGULAR_FONT_ID = "libre_franklin"
BOLD_FONT_ID = "libre_franklin_bold"
TEXT_RGB = (20, 20, 20)
MUTED_RGB = (110, 110, 110)
RULE_RGB = (200, 200, 200)

TITLE = "How to fill in your mock checks"
FILL_IN_STEPS = (
    "Use a blue or black ballpoint pen.",
    "Under each check is a strip that tells you what to write in each blank. Copy it exactly: same spelling, "
    "numbers and punctuation, even when it looks unusual (like \"1,070.xx\" or \"July 25th 2026\").",
    "Write naturally, the way you would write a real check. Do not try to be extra neat.",
    "Fill every blank on every check. Keep your writing inside the check, on its lines and in its boxes. "
    "Do not write on the strip.",
    "Sign with the name after \"Sign:\" on the strip (it is made up), in your usual signing style.",
    "Leave the small printed code under the memo line (like S-0007-A) uncovered.",
    "Made a mistake? Do not cross it out. Put that check aside and leave it out of the photos.",
)
CUT_STEPS = (
    "When every check on every page is filled in, cut each one out along its dashed outline. "
    "The black corner marks show where the lines meet.",
    "Throw away the strips and the rest of the paper.",
)
PHOTO_STEPS = (
    "Lay 4 to 6 checks on a bedsheet or a table, not overlapping.",
    "Photograph them from above with a phone. Take a few photos of each batch: some straight down, some at an angle.",
    "Keep every check fully in the picture. Change the surface, room or lighting between batches.",
)
FAKE_DATA_NOTE = "Everything on these checks is made up: names, banks, amounts and account numbers. They are not real checks."
EXAMPLE_BEFORE_CAPTION = "Before: the blanks, with the strip underneath"
EXAMPLE_AFTER_CAPTION = "After: filled in by copying the strip"


def wrap_to_width(text: str, font: ImageFont.FreeTypeFont, max_width_px: int) -> list[str]:
    """Break `text` at spaces so no line is wider than `max_width_px`."""
    lines, current = [], ""
    for word in text.split(" "):
        candidate = f"{current} {word}" if current else word
        if current and font.getlength(candidate) > max_width_px:
            lines.append(current)
            current = word
        else:
            current = candidate
    return lines + [current] if current else lines


class InstructionPageWriter:
    """Lays text down a Letter page from the top, tracking the next free line."""

    def __init__(self):
        self.page = Image.new("RGB", LETTER_SIZE_PX, "white")
        self.draw = ImageDraw.Draw(self.page)
        self.y = PAGE_MARGIN_PX
        self.text_width = LETTER_SIZE_PX[0] - 2 * PAGE_MARGIN_PX

    def paragraph(self, text: str, font_id: str, em_px: int, rgb=TEXT_RGB, indent_px: int = 0, width_px: int | None = None) -> None:
        """Wrapped text starting at the cursor."""
        font = load_font(font_id, em_px)
        for line in wrap_to_width(text, font, (width_px or self.text_width) - indent_px):
            self.y += em_px
            self.draw.text((PAGE_MARGIN_PX + indent_px, self.y), line, fill=rgb, font=font, anchor="ls")
            self.y += round(em_px * (LINE_PITCH_EM - 1.0))
        self.y += PARAGRAPH_GAP_PX

    def numbered_step(self, number: int, heading: str, bullets: tuple[str, ...]) -> None:
        """'1  Heading' then its bullets with a hanging indent."""
        self.y += SECTION_GAP_PX
        self.paragraph(f"{number}   {heading}", BOLD_FONT_ID, STEP_HEADING_EM_PX)
        body_font = load_font(REGULAR_FONT_ID, BODY_EM_PX)
        for bullet in bullets:
            self.draw.text((PAGE_MARGIN_PX + BULLET_INDENT_PX // 3, self.y + BODY_EM_PX), "•", fill=TEXT_RGB,
                           font=body_font, anchor="ls")
            self.paragraph(bullet, REGULAR_FONT_ID, BODY_EM_PX, indent_px=BULLET_INDENT_PX)

    def writer_badge(self, letter: str) -> None:
        """Big boxed writer letter, top right, so stacks from different writers never mix."""
        x1, y0 = LETTER_SIZE_PX[0] - PAGE_MARGIN_PX, PAGE_MARGIN_PX
        x0, y1 = x1 - BADGE_SIZE_PX, y0 + BADGE_SIZE_PX
        self.draw.rounded_rectangle((x0, y0, x1, y1), radius=BADGE_SIZE_PX // 8, outline=TEXT_RGB, width=6)
        self.draw.text(((x0 + x1) // 2, y0 + int(0.2 * RENDER_DPI)), "WRITER", fill=MUTED_RGB,
                       font=load_font(BOLD_FONT_ID, CAPTION_EM_PX), anchor="ms")
        self.draw.text(((x0 + x1) // 2, y1 - int(0.14 * RENDER_DPI)), letter, fill=TEXT_RGB,
                       font=load_font(BOLD_FONT_ID, BADGE_LETTER_EM_PX), anchor="ms")

    def example_pair(self, before: Image.Image, after: Image.Image) -> None:
        """Before and after tiles side by side, scaled to the text width, captioned above."""
        gap = EXAMPLE_GAP_PX
        scale = (self.text_width - gap) / (before.width + after.width)
        caption_font = load_font(BOLD_FONT_ID, CAPTION_EM_PX)
        self.y += SECTION_GAP_PX
        x = PAGE_MARGIN_PX
        tile_top = self.y + round(CAPTION_EM_PX * 1.6)
        for tile, caption in ((before, EXAMPLE_BEFORE_CAPTION), (after, EXAMPLE_AFTER_CAPTION)):
            scaled = tile.resize((round(tile.width * scale), round(tile.height * scale)), Image.Resampling.LANCZOS)
            self.draw.text((x, self.y + CAPTION_EM_PX), caption, fill=MUTED_RGB, font=caption_font, anchor="ls")
            self.page.paste(scaled, (x, tile_top))
            self.draw.rectangle((x - 1, tile_top - 1, x + scaled.width, tile_top + scaled.height), outline=RULE_RGB, width=2)
            x += scaled.width + gap
        self.y = tile_top + round(max(before.height, after.height) * scale)


def render_instruction_page(letter: str, first_page: int, last_page: int, check_count: int,
                            example_before: Image.Image, example_after: Image.Image) -> Image.Image:
    """The instruction page that opens writer `letter`'s stack (Letter size, page pixels)."""
    writer = InstructionPageWriter()
    writer.writer_badge(letter)
    heading_width = writer.text_width - BADGE_SIZE_PX - int(0.2 * RENDER_DPI)
    writer.paragraph(TITLE, BOLD_FONT_ID, TITLE_EM_PX, width_px=heading_width)
    writer.paragraph(f"Your stack: pages {first_page} to {last_page}, {check_count} checks. "
                     f"Every check code in it ends in -{letter}.", REGULAR_FONT_ID, SUBTITLE_EM_PX, width_px=heading_width)
    writer.paragraph(FAKE_DATA_NOTE, REGULAR_FONT_ID, SUBTITLE_EM_PX, rgb=MUTED_RGB, width_px=heading_width)
    writer.numbered_step(1, "Fill in every check (before cutting anything)", FILL_IN_STEPS)
    writer.example_pair(example_before, example_after)
    writer.numbered_step(2, "Cut out the checks", CUT_STEPS)
    writer.numbered_step(3, "Photograph them", PHOTO_STEPS)
    if writer.y > LETTER_SIZE_PX[1] - PAGE_MARGIN_PX:
        raise ValueError(f"instruction page overflows: text ends at {writer.y} px of {LETTER_SIZE_PX[1]}")
    return writer.page
