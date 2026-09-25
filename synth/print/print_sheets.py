"""Print-ready PDF pages of fake checks at true size, for the real-photo gold eval set.

Print these on Letter paper at 100% scale, cut along the marks, lay the checks on a
bedsheet, and photograph them. Each check carries a small serial (e.g. S-0007) under
its memo line; `print_labels.csv` maps every serial to its ground truth so photos can be
matched back without hand transcription.

- Personal checks (6 x 2.75 in): 3 per page, stacked.
- Business checks (8.5 x 3.5 in): too wide for Letter with margins, so 2 per page, rotated.
"""

import csv
import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from synth.render.check_layout import RENDER_DPI, CheckSizeKind
from synth.render.check_templates import build_template_catalog
from synth.render.fake_data import sample_check_content
from synth.render.fonts import load_font
from synth.render.render_check import render_check

logger = logging.getLogger(__name__)

LETTER_SIZE_PX = (int(8.5 * RENDER_DPI), int(11 * RENDER_DPI))
CUT_MARK_LENGTH_PX = int(0.2 * RENDER_DPI)
CUT_MARK_GAP_PX = int(0.05 * RENDER_DPI)
CHECKS_PER_PAGE = {CheckSizeKind.PERSONAL: 3, CheckSizeKind.BUSINESS: 2}
PDF_JPEG_QUALITY = 95
CSV_COLUMNS = ["serial", "page", "slot", "template_id", "size_kind", "payer_name", "check_number", "date_text", "date_iso",
               "payee_text", "payee_canonical", "amount_cents", "amount_numeric_text", "amount_words_text", "memo_text",
               "bank_name", "handwritten_fields"]


def draw_cut_marks(draw: ImageDraw.ImageDraw, x0: int, y0: int, x1: int, y1: int) -> None:
    """Short hairlines just outside each corner, along both edges."""
    gap, length = CUT_MARK_GAP_PX, CUT_MARK_LENGTH_PX
    for corner_x, corner_y, direction_x, direction_y in ((x0, y0, -1, -1), (x1, y0, 1, -1), (x1, y1, 1, 1), (x0, y1, -1, 1)):
        draw.line((corner_x + direction_x * gap, corner_y, corner_x + direction_x * (gap + length), corner_y), fill=0, width=2)
        draw.line((corner_x, corner_y + direction_y * gap, corner_x, corner_y + direction_y * (gap + length)), fill=0, width=2)


def slot_origins(size_kind: CheckSizeKind, check_size: tuple[int, int]) -> list[tuple[int, int]]:
    """Top-left positions for each slot on a page (after any rotation of the check)."""
    page_width, page_height = LETTER_SIZE_PX
    check_width, check_height = check_size
    count = CHECKS_PER_PAGE[size_kind]
    if size_kind == CheckSizeKind.PERSONAL:
        gap = (page_height - count * check_height) // (count + 1)
        return [((page_width - check_width) // 2, gap + slot * (check_height + gap)) for slot in range(count)]
    gap = (page_width - count * check_width) // (count + 1)
    return [(gap + slot * (check_width + gap), (page_height - check_height) // 2) for slot in range(count)]


def write_print_sheets(output_directory: Path, page_count: int, seed: int, template_count: int) -> Path:
    """Write `print_sheets.pdf`, `print_labels.csv` and `print_labels.json` into `output_directory`."""
    output_directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng([seed, 9091])
    catalog = build_template_catalog(template_count)
    pages, csv_rows, json_records = [], [], []
    serial_number = 1
    for page_index in range(page_count):
        size_kind = CheckSizeKind.BUSINESS if page_index % 4 == 3 else CheckSizeKind.PERSONAL
        templates = [template for template in catalog if template.size_kind == size_kind]
        page = Image.new("RGB", LETTER_SIZE_PX, "white")
        draw = ImageDraw.Draw(page)
        rendered = []
        for _ in range(CHECKS_PER_PAGE[size_kind]):
            template = templates[int(rng.integers(len(templates)))]
            content = sample_check_content(template, rng, serial=f"S-{serial_number:04d}")
            serial_number += 1
            image, label = render_check(template, content, rng, simulate_print_texture=False)  # real printer + paper add texture
            rendered.append((template, content, image.rotate(90, expand=True) if size_kind == CheckSizeKind.BUSINESS else image, label))
        for slot, ((template, content, image, label), origin) in enumerate(zip(rendered, slot_origins(size_kind, rendered[0][2].size))):
            page.paste(image, origin)
            draw_cut_marks(draw, origin[0], origin[1], origin[0] + image.width, origin[1] + image.height)
            csv_rows.append({
                "serial": content.serial, "page": page_index + 1, "slot": slot + 1, "template_id": template.template_id,
                "size_kind": size_kind.value, "payer_name": content.payer_name, "check_number": content.check_number,
                "date_text": content.date_text, "date_iso": content.date_iso, "payee_text": content.payee_text,
                "payee_canonical": content.payee_canonical, "amount_cents": content.amount_cents,
                "amount_numeric_text": content.amount_numeric_text, "amount_words_text": content.amount_words_text,
                "memo_text": content.memo_text, "bank_name": content.bank_name,
                "handwritten_fields": ";".join(content.handwritten_fields),
            })
            json_records.append({"serial": content.serial, "page": page_index + 1, "slot": slot + 1,
                                 "rotated_on_page": size_kind == CheckSizeKind.BUSINESS, "check_label": label.to_dict()})
        footer = f"Check Transcriber mock checks, page {page_index + 1} of {page_count}. Fictional data; print at 100% scale."
        draw.text((LETTER_SIZE_PX[0] // 2, LETTER_SIZE_PX[1] - 40), footer, fill=90, font=load_font("pt_sans", 28), anchor="ms")
        pages.append(page)

    pdf_path = output_directory / "print_sheets.pdf"
    pages[0].save(pdf_path, save_all=True, append_images=pages[1:], resolution=RENDER_DPI, quality=PDF_JPEG_QUALITY, subsampling=0)
    with open(output_directory / "print_labels.csv", "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(csv_rows)
    (output_directory / "print_labels.json").write_text(json.dumps(json_records, indent=1))
    logger.info("wrote %d pages, %d checks to %s", page_count, len(csv_rows), output_directory)
    return pdf_path
