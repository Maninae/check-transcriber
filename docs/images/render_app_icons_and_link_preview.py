"""Render the app's icons and link-preview images from their sources, into app/.

    python3 docs/images/render_app_icons_and_link_preview.py

Sources:
    app/favicon.svg                      the icon (a check on the lavender tile)
    docs/images/link_preview_card.html   the og:image card layout
    docs/images/count-step.png           the photo shown in the card (only the photo is cropped)

Outputs (all rasterized by headless Chromium, so the PNGs match what a browser draws):
    app/favicon.ico                  16 + 32 px
    app/icons/icon-192.png, icon-512.png            rounded tile, transparent corners
    app/icons/apple-touch-icon.png   180 px, full-bleed opaque tile (iOS rounds it)
    app/icons/icon-maskable-512.png  full-bleed tile, glyph inside the 80% safe circle
    app/preview.png                  1200x630 og:image
    app/preview-square.png           1080x1080

- Re-run after changing any source, then `python3 app/tests/sync_service_worker_shell_list.py`
  and bump SW_VERSION in app/sw.js.
"""

import base64
import io
import re
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
APP_DIRECTORY = REPOSITORY_ROOT / "app"
ICON_DIRECTORY = APP_DIRECTORY / "icons"
FAVICON_SVG_PATH = APP_DIRECTORY / "favicon.svg"
PREVIEW_CARD_HTML_PATH = REPOSITORY_ROOT / "docs" / "images" / "link_preview_card.html"
COUNT_STEP_SCREENSHOT_PATH = REPOSITORY_ROOT / "docs" / "images" / "count-step.png"

# The checks inside count-step.png's photo (the photo itself spans x 108-1063, y 159-875;
# everything outside it is the pre-theme page chrome, which must not appear in the card).
COUNT_STEP_CHECKS_CROP_BOX = (150, 180, 1050, 800)
# Glyph centre in favicon.svg user units, and the scales that fit it in each full-bleed variant.
GLYPH_CENTER = (32.0, 34.5)
APPLE_TOUCH_GLYPH_SCALE = 0.9
MASKABLE_GLYPH_SCALE = 0.66  # glyph half-diagonal ~36 units * 0.66 < the 25.6-unit safe radius
PREVIEW_MAX_BYTES = 300_000
# A photo is ~420 KB as truecolour PNG; a dithered palette shows no banding at 2x zoom.
# Largest palette that fits the budget wins.
PREVIEW_PALETTE_COLOR_STEPS = (256, 192, 160, 128)


def build_full_bleed_icon_svg(favicon_svg: str, glyph_scale: float) -> str:
    """The favicon with square corners and the glyph re-centred at `glyph_scale`."""
    square_tile_svg = favicon_svg.replace(' rx="14"', "", 1)
    center_x, center_y = GLYPH_CENTER
    transform = f"translate(32 32) scale({glyph_scale}) translate({-center_x} {-center_y})"
    return square_tile_svg.replace('<g id="glyph">', f'<g id="glyph" transform="{transform}">', 1)


def rasterize_svg(browser, svg_text: str, pixel_size: int) -> Image.Image:
    """Draw `svg_text` at pixel_size x pixel_size, keeping transparency."""
    page = browser.new_page(viewport={"width": pixel_size, "height": pixel_size}, device_scale_factor=1)
    svg_data_uri = "data:image/svg+xml;base64," + base64.b64encode(svg_text.encode()).decode()
    page.set_content(
        f'<body style="margin:0"><img src="{svg_data_uri}" width="{pixel_size}" height="{pixel_size}" style="display:block"></body>'
    )
    png_bytes = page.screenshot(omit_background=True)
    page.close()
    return Image.open(io.BytesIO(png_bytes)).convert("RGBA")


def render_icons(browser) -> None:
    """Every icon file, from app/favicon.svg."""
    favicon_svg = FAVICON_SVG_PATH.read_text()
    ICON_DIRECTORY.mkdir(exist_ok=True)
    for pixel_size in (192, 512):
        rasterize_svg(browser, favicon_svg, pixel_size).save(ICON_DIRECTORY / f"icon-{pixel_size}.png", optimize=True)
    apple_touch_svg = build_full_bleed_icon_svg(favicon_svg, APPLE_TOUCH_GLYPH_SCALE)
    # iOS shows transparency as black, so flatten onto the lavender.
    apple_touch_icon = rasterize_svg(browser, apple_touch_svg, 180)
    opaque_background = Image.new("RGB", apple_touch_icon.size, "#edeef8")
    opaque_background.paste(apple_touch_icon, mask=apple_touch_icon.getchannel("A"))
    opaque_background.save(ICON_DIRECTORY / "apple-touch-icon.png", optimize=True)
    maskable_svg = build_full_bleed_icon_svg(favicon_svg, MASKABLE_GLYPH_SCALE)
    rasterize_svg(browser, maskable_svg, 512).convert("RGB").save(ICON_DIRECTORY / "icon-maskable-512.png", optimize=True)
    icon_32 = rasterize_svg(browser, favicon_svg, 32)
    icon_16 = rasterize_svg(browser, favicon_svg, 16)
    icon_32.save(APP_DIRECTORY / "favicon.ico", format="ICO", sizes=[(32, 32), (16, 16)], append_images=[icon_16])


def build_photo_data_uri() -> str:
    """The checks from the count-step screenshot as a PNG data URI."""
    photo = Image.open(COUNT_STEP_SCREENSHOT_PATH).convert("RGB").crop(COUNT_STEP_CHECKS_CROP_BOX)
    buffer = io.BytesIO()
    photo.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def render_preview(browser, layout_class: str, width: int, height: int, output_path: Path) -> None:
    """Render the card at exactly width x height and report overlaps/overflow."""
    card_html = PREVIEW_CARD_HTML_PATH.read_text().replace("PHOTO_DATA_URI", build_photo_data_uri())
    card_html = re.sub(r'<body class="[^"]*">', f'<body class="{layout_class}">', card_html)
    scratch_html_path = PREVIEW_CARD_HTML_PATH.with_name(".rendered_link_preview_card.html")
    scratch_html_path.write_text(card_html)
    try:
        page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=1)
        page.goto(scratch_html_path.as_uri())
        page.evaluate("document.fonts.ready")
        page.wait_for_function("document.querySelector('.photo img').complete")
        boxes = page.evaluate(
            """() => Object.fromEntries(['.title', '.line', '.photo'].map(s => {
                const r = document.querySelector(s).getBoundingClientRect();
                return [s, [Math.round(r.left), Math.round(r.top), Math.round(r.right), Math.round(r.bottom)]];
            }))"""
        )
        font_ok = page.evaluate("document.fonts.check('500 64px Poppins')")
        print(f"{output_path.name}: boxes {boxes}, Poppins loaded {font_ok}")
        for selector, (left, top, right, bottom) in boxes.items():
            if left < 0 or top < 0 or right > width or bottom > height:
                raise SystemExit(f"{selector} overflows the {width}x{height} frame: {boxes[selector]}")
        png_bytes = page.screenshot(clip={"x": 0, "y": 0, "width": width, "height": height})
        page.close()
    finally:
        scratch_html_path.unlink()
    rendered_card = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    for palette_colors in PREVIEW_PALETTE_COLOR_STEPS:
        rendered_card.quantize(
            colors=palette_colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG
        ).save(output_path, optimize=True)
        size_bytes = output_path.stat().st_size
        if size_bytes <= PREVIEW_MAX_BYTES:
            print(f"{output_path.name}: {size_bytes / 1000:.0f} KB at {palette_colors} colours")
            return
    raise SystemExit(f"{output_path} is over {PREVIEW_MAX_BYTES} bytes even at {palette_colors} colours")


def main() -> None:
    """Render everything."""
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        render_icons(browser)
        render_preview(browser, "wide", 1200, 630, APP_DIRECTORY / "preview.png")
        render_preview(browser, "square", 1080, 1080, APP_DIRECTORY / "preview-square.png")
        browser.close()


if __name__ == "__main__":
    main()
