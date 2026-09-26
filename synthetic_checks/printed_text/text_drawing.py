"""Printed text on a check, plus the compositing helpers the handwriting engine shares.

- `draw_printed_text`: straight Pillow text, shrunk to fit `max_width_px`, box from `textbbox`.
- Handwriting lives in `handwriting_writer` (contract) and `handwriting_field_renderer`
  (pipeline); it reuses `paste_ink_layer` and `ALPHA_INK_THRESHOLD` from here.
"""

from PIL import Image, ImageDraw

from synthetic_checks.fonts.font_registry import load_font

ALPHA_INK_THRESHOLD = 40  # alpha (0-255) above which a pixel counts as ink for label boxes
MIN_FIT_FONT_PX = 10


def fit_font_size(font_id: str, text: str, em_px: int, max_width_px: float) -> int:
    """Largest em size <= `em_px` at which `text` fits in `max_width_px`."""
    size = em_px
    while size > MIN_FIT_FONT_PX and load_font(font_id, size).getlength(text) > max_width_px:
        size = int(size * 0.92)
    return size


def draw_printed_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font_id: str,
    em_px: int,
    position: tuple[float, float],
    fill_rgb: tuple[int, int, int],
    anchor: str = "ls",
    max_width_px: float | None = None,
) -> tuple[int, int, int, int]:
    """Draw straight text; return its (x0, y0, x1, y1) box."""
    size = fit_font_size(font_id, text, em_px, max_width_px) if max_width_px else em_px
    font = load_font(font_id, size)
    draw.text(position, text, font=font, fill=fill_rgb, anchor=anchor)
    return tuple(int(round(v)) for v in draw.textbbox(position, text, font=font, anchor=anchor))


def paste_ink_layer(canvas: Image.Image, ink_layer: Image.Image, offset_x: int, offset_y: int) -> None:
    """Alpha-composite `ink_layer` onto `canvas` at an offset that may be partly off-canvas."""
    source_x0, source_y0 = max(0, -offset_x), max(0, -offset_y)
    dest_x0, dest_y0 = max(0, offset_x), max(0, offset_y)
    width = min(ink_layer.width - source_x0, canvas.width - dest_x0)
    height = min(ink_layer.height - source_y0, canvas.height - dest_y0)
    if width <= 0 or height <= 0:
        return
    cropped_layer = ink_layer.crop((source_x0, source_y0, source_x0 + width, source_y0 + height))
    canvas.alpha_composite(cropped_layer, dest=(dest_x0, dest_y0))


def clip_box_to_size(box: tuple[float, float, float, float], size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Clamp an (x0, y0, x1, y1) box to [0, width] x [0, height] and round to ints."""
    width, height = size
    x0, y0, x1, y1 = box
    return (int(max(0, min(width, x0))), int(max(0, min(height, y0))), int(max(0, min(width, x1))), int(max(0, min(height, y1))))
