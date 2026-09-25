"""Letter-page geometry for true-size printed checks: how many fit, where they go, cut guides.

- A check narrower than the printable width is placed upright, stacked top to bottom.
- A wider check (business, 8.5 in) is rotated 90 degrees and placed side by side.
- Cut guides: short corner marks plus a light dashed rectangle along every check edge, so
  scissors can follow the line and a slightly off cut still leaves the check whole.
"""

from dataclasses import dataclass

from PIL import ImageDraw

from synth.render.check_layout import RENDER_DPI

LETTER_SIZE_PX = (int(8.5 * RENDER_DPI), int(11 * RENDER_DPI))
PAGE_MARGIN_PX = int(0.4 * RENDER_DPI)       # inside most printers' unprintable border
FOOTER_HEIGHT_PX = int(0.3 * RENDER_DPI)
MIN_GAP_PX = int(0.25 * RENDER_DPI)          # room for cut marks between checks
CUT_MARK_LENGTH_PX = int(0.2 * RENDER_DPI)
CUT_MARK_GAP_PX = int(0.05 * RENDER_DPI)
CUT_MARK_WIDTH_PX = 2
CUT_MARK_RGB = (0, 0, 0)
DASH_LENGTH_PX = int(0.06 * RENDER_DPI)
DASH_RGB = (170, 170, 170)


@dataclass(frozen=True)
class PageArrangement:
    """How checks of one size sit on a Letter page."""

    rotated: bool                          # checks are turned 90 degrees clockwise on the page
    origins: tuple[tuple[int, int], ...]   # top-left of each slot, page pixels, after any rotation


def arrange_checks_on_page(check_width_px: int, check_height_px: int) -> PageArrangement:
    """Fit as many true-size checks as possible, evenly spaced."""
    usable_width = LETTER_SIZE_PX[0] - 2 * PAGE_MARGIN_PX
    usable_height = LETTER_SIZE_PX[1] - 2 * PAGE_MARGIN_PX - FOOTER_HEIGHT_PX
    rotated = check_width_px > usable_width
    placed_width, placed_height = (check_height_px, check_width_px) if rotated else (check_width_px, check_height_px)
    if rotated:
        count = max(1, (usable_width + MIN_GAP_PX) // (placed_width + MIN_GAP_PX))
        gap = (LETTER_SIZE_PX[0] - count * placed_width) // (count + 1)   # spread over the whole width: wider inner gap
        top = PAGE_MARGIN_PX + (usable_height - placed_height) // 2
        origins = tuple((gap + slot * (placed_width + gap), top) for slot in range(count))
    else:
        count = max(1, (usable_height + MIN_GAP_PX) // (placed_height + MIN_GAP_PX))
        gap = (usable_height - count * placed_height) // (count + 1)
        left = (LETTER_SIZE_PX[0] - placed_width) // 2
        origins = tuple((left, PAGE_MARGIN_PX + gap + slot * (placed_height + gap)) for slot in range(count))
    return PageArrangement(rotated, origins)


def draw_dashed_rectangle(draw: ImageDraw.ImageDraw, x0: int, y0: int, x1: int, y1: int) -> None:
    """Light dashed lines exactly on the check's edges."""
    for start in range(x0, x1, 2 * DASH_LENGTH_PX):
        end = min(start + DASH_LENGTH_PX, x1)
        draw.line((start, y0, end, y0), fill=DASH_RGB, width=1)
        draw.line((start, y1, end, y1), fill=DASH_RGB, width=1)
    for start in range(y0, y1, 2 * DASH_LENGTH_PX):
        end = min(start + DASH_LENGTH_PX, y1)
        draw.line((x0, start, x0, end), fill=DASH_RGB, width=1)
        draw.line((x1, start, x1, end), fill=DASH_RGB, width=1)


def draw_cut_guides(draw: ImageDraw.ImageDraw, x0: int, y0: int, x1: int, y1: int) -> None:
    """Corner hairlines just outside each corner, along both edges, plus the dashed outline."""
    draw_dashed_rectangle(draw, x0, y0, x1, y1)
    gap, length = CUT_MARK_GAP_PX, CUT_MARK_LENGTH_PX
    for corner_x, corner_y, direction_x, direction_y in ((x0, y0, -1, -1), (x1, y0, 1, -1), (x1, y1, 1, 1), (x0, y1, -1, 1)):
        draw.line((corner_x + direction_x * gap, corner_y, corner_x + direction_x * (gap + length), corner_y),
                  fill=CUT_MARK_RGB, width=CUT_MARK_WIDTH_PX)
        draw.line((corner_x, corner_y + direction_y * gap, corner_x, corner_y + direction_y * (gap + length)),
                  fill=CUT_MARK_RGB, width=CUT_MARK_WIDTH_PX)
