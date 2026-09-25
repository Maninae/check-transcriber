"""Draw a string onto a check and report the tight pixel box of the ink it laid down.

Two styles:
- printed: straight Pillow text, box from `textbbox`.
- handwritten: glyph-by-glyph onto a transparent layer with per-glyph size, baseline and
  spacing jitter, a small overall slant and baseline drift, variable ink density; the box
  is measured from the layer's alpha AFTER rotation, so it is tight around the real ink.

Both shrink the font until the text fits `max_width_px`.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from synth.render.fonts import load_font

ALPHA_INK_THRESHOLD = 40
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


def draw_handwritten_text(
    canvas: Image.Image,
    text: str,
    font_id: str,
    em_px: int,
    baseline_left: tuple[float, float],
    max_width_px: float,
    ink_rgb: tuple[int, int, int],
    rng: np.random.Generator,
    max_slant_degrees: float = 2.5,
    pen_width_px: int = 1,
) -> tuple[int, int, int, int] | None:
    """Composite pen-like text onto RGBA `canvas`; return its tight box, or None if nothing was inked."""
    size = fit_font_size(font_id, text, em_px, max_width_px * 0.97)
    per_glyph_sizes = [max(MIN_FIT_FONT_PX, int(size * rng.uniform(0.93, 1.07))) for _ in text]
    advances = [load_font(font_id, glyph_size).getlength(character) * rng.uniform(0.96, 1.06)
                for character, glyph_size in zip(text, per_glyph_sizes)]
    layer_width = int(sum(advances) + size * 2)
    layer_height = int(size * 3)
    layer = Image.new("L", (layer_width, layer_height), 0)
    layer_draw = ImageDraw.Draw(layer)
    baseline_y = size * 1.8
    baseline_drift_per_px = rng.uniform(-0.006, 0.006)
    cursor_x = float(size)
    for character, glyph_size, advance in zip(text, per_glyph_sizes, advances):
        glyph_baseline = baseline_y + rng.normal(0, size * 0.025) + (cursor_x - size) * baseline_drift_per_px
        ink_density = int(255 * rng.uniform(0.8, 1.0))
        layer_draw.text((cursor_x, glyph_baseline), character, font=load_font(font_id, glyph_size), fill=ink_density, anchor="ls")
        cursor_x += advance

    slant_degrees = rng.uniform(-max_slant_degrees, max_slant_degrees)
    pivot = (size, baseline_y)
    if pen_width_px > 1:
        # Handwriting fonts draw thinner strokes than a ballpoint; grow them to the pen width.
        layer = layer.filter(ImageFilter.MaxFilter(pen_width_px | 1))
    layer = layer.rotate(slant_degrees, resample=Image.Resampling.BICUBIC, center=pivot)
    offset_x = int(round(baseline_left[0] - pivot[0]))
    offset_y = int(round(baseline_left[1] - pivot[1]))

    alpha = np.asarray(layer)
    rows = np.where((alpha > ALPHA_INK_THRESHOLD).any(axis=1))[0]
    columns = np.where((alpha > ALPHA_INK_THRESHOLD).any(axis=0))[0]
    if len(rows) == 0:
        return None
    ink_layer = Image.new("RGBA", layer.size, ink_rgb + (0,))
    ink_layer.putalpha(layer)
    paste_ink_layer(canvas, ink_layer, offset_x, offset_y)
    box = (columns[0] + offset_x, rows[0] + offset_y, columns[-1] + 1 + offset_x, rows[-1] + 1 + offset_y)
    return clip_box_to_size(box, canvas.size)


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
