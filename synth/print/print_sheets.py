"""Print-ready Letter pages of fake checks at true size, for Owen's real-photo gold eval set.

Print `print_sheets.pdf` at 100% scale, cut along the guides, lay the checks on a bedsheet,
photograph them. Each check carries a small printed serial (S-0007) under its memo line;
`print_labels.csv` / `.json` map every serial to every field's text, so photos can be scored
without hand transcription. `print_sheet__page=NN.png` is each page as rendered, for viewing.

- Clean stock: a real printer and paper add their own toner and fibre, so checks are rendered
  without simulated print texture (`render_check(..., simulate_print_texture=False)`) by default.
- Coverage: each page alternates slots between handwritten and printed fill-ins, and cycles
  layout families, so every sheet has both kinds of fill-in and several designs.
- `held_out_split="eval"` restricts templates and fonts to that seed's eval pools, so a model
  trained on the same seed's train split has never seen the printed designs or hands.
"""

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from synth.dataset.splits import (
    DEFAULT_SPLIT_FRACTIONS,
    HANDWRITING_FONT_SALT,
    SIGNATURE_FONT_SALT,
    assign_ids_to_splits,
    plan_template_pools,
)
from synth.dataset.build_plan import template_family_by_id
from synth.print.print_page_layout import LETTER_SIZE_PX, arrange_checks_on_page, draw_cut_guides
from synth.render.check_fields import FieldName
from synth.render.check_layout import RENDER_DPI, CheckSizeKind, check_size_pixels
from synth.render.check_templates import TemplateDesign, build_template_catalog
from synth.render.fake_data import sample_check_content
from synth.render.fonts import FontRole, font_ids_with_role, load_font
from synth.render.render_check import render_check

logger = logging.getLogger(__name__)

PRINT_RNG_SALT = 9091
PAGE_SIZE_KIND_CYCLE = ("personal", "business", "personal", "money_order")
MIN_HANDWRITTEN_FIELDS_FOR_HANDWRITTEN_STYLE = 2
MAX_CONTENT_DRAWS = 200
PDF_JPEG_QUALITY = 95
FOOTER_FONT_ID = "pt_sans"
FOOTER_FONT_PX = 28
FOOTER_RGB = (90, 90, 90)
FIELD_TEXT_COLUMNS = [f"text__{field_name.value}" for field_name in FieldName]
CSV_COLUMNS = ["serial", "page", "slot", "template_id", "layout_family", "size_kind", "rotated_on_page", "fill_in_style",
               "handwritten_fields", "handwriting_font_id", "signature_font_id", "amount_cents", "date_iso",
               "payee_canonical", "clean_stock", *FIELD_TEXT_COLUMNS]


def render_check_for_print(template: TemplateDesign, content, rng: np.random.Generator, font_pools: dict,
                           clean_stock: bool) -> tuple[Image.Image, object]:
    """Render one check as RGB (clean stock has full alpha, so dropping it loses nothing)."""
    image, label = render_check(template, content, rng, handwriting_font_ids=font_pools.get("handwriting"),
                                signature_font_ids=font_pools.get("signature"), simulate_print_texture=not clean_stock)
    return image.convert("RGB"), label


def layout_family_of(template: TemplateDesign) -> str:
    """The template's layout family name."""
    return template.layout_family.value


def sample_content_with_style(template: TemplateDesign, rng: np.random.Generator, serial: str, want_handwritten: bool):
    """Draw fake content until its fill-in style matches the request (kept deterministic by the rng)."""
    content = None
    for _ in range(MAX_CONTENT_DRAWS):
        content = sample_check_content(template, rng, serial=serial)
        if (fill_in_style(content) == "handwritten") == want_handwritten:
            return content
    logger.warning("%s: could not draw %s fill-ins on %s", serial, "handwritten" if want_handwritten else "printed",
                   template.template_id)
    return content


def fill_in_style(content) -> str:
    """What was actually drawn: money orders are always hand-filled, so a printed request can fall back."""
    return "handwritten" if len(content.handwritten_fields) >= MIN_HANDWRITTEN_FIELDS_FOR_HANDWRITTEN_STYLE else "printed"


def print_pools(template_count: int, seed: int, held_out_split: str | None) -> tuple[list[TemplateDesign], dict]:
    """Templates and font pools to print from: everything, or one split's held-out pools."""
    catalog = build_template_catalog(template_count)
    if held_out_split is None:
        return catalog, {}
    template_ids = set(plan_template_pools(template_family_by_id(catalog), seed)[held_out_split])
    font_pools = {
        "handwriting": assign_ids_to_splits(font_ids_with_role(FontRole.HANDWRITING), DEFAULT_SPLIT_FRACTIONS, seed,
                                            HANDWRITING_FONT_SALT)[held_out_split],
        "signature": assign_ids_to_splits(font_ids_with_role(FontRole.SIGNATURE), DEFAULT_SPLIT_FRACTIONS, seed,
                                          SIGNATURE_FONT_SALT)[held_out_split],
    }
    return [template for template in catalog if template.template_id in template_ids], font_pools


def label_record(serial: str, page: int, slot: int, template: TemplateDesign, rotated: bool,
                 content, label, clean_stock: bool) -> dict:
    """One CSV row: identity, style, fonts and the text of every field on the check."""
    texts = {field.field_name: field.text for field in label.fields}
    return {
        "serial": serial, "page": page, "slot": slot, "template_id": template.template_id,
        "layout_family": layout_family_of(template), "size_kind": template.size_kind.value, "rotated_on_page": rotated,
        "fill_in_style": fill_in_style(content),
        "handwritten_fields": ";".join(content.handwritten_fields),
        "handwriting_font_id": label.canonical.get("handwriting_font_id"),
        "signature_font_id": label.canonical.get("signature_font_id"),
        "amount_cents": content.amount_cents, "date_iso": content.date_iso, "payee_canonical": content.payee_canonical,
        "clean_stock": clean_stock,
        **{f"text__{field_name.value}": texts.get(field_name.value, "") for field_name in FieldName},
    }


def write_print_sheets(output_directory: Path, page_count: int, seed: int, template_count: int,
                       held_out_split: str | None = None, clean_stock: bool = True) -> Path:
    """Write the PDF, one PNG per page, `print_labels.csv` and `print_labels.json`; returns the PDF path."""
    output_directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng([seed, PRINT_RNG_SALT])
    templates, font_pools = print_pools(template_count, seed, held_out_split)
    families_by_size = defaultdict(lambda: defaultdict(list))
    for template in templates:
        families_by_size[template.size_kind.value][layout_family_of(template)].append(template)
    size_cycle = [kind for kind in PAGE_SIZE_KIND_CYCLE if kind in families_by_size] or sorted(families_by_size)
    family_turn = defaultdict(int)
    pages, csv_rows, json_records = [], [], []
    serial_number = 1
    for page_index in range(page_count):
        size_kind = size_cycle[page_index % len(size_cycle)]
        families = sorted(families_by_size[size_kind])
        arrangement = arrange_checks_on_page(*check_size_pixels(CheckSizeKind(size_kind)))
        page = Image.new("RGB", LETTER_SIZE_PX, "white")
        draw = ImageDraw.Draw(page)
        for slot, (x, y) in enumerate(arrangement.origins):
            family = families[family_turn[size_kind] % len(families)]
            family_turn[size_kind] += 1
            candidates = families_by_size[size_kind][family]
            template = candidates[int(rng.integers(len(candidates)))]
            want_handwritten = slot % 2 == 0
            serial = f"S-{serial_number:04d}"
            serial_number += 1
            content = sample_content_with_style(template, rng, serial, want_handwritten)
            image, label = render_check_for_print(template, content, rng, font_pools, clean_stock)
            if arrangement.rotated:
                image = image.transpose(Image.Transpose.ROTATE_270)
            page.paste(image, (x, y))
            draw_cut_guides(draw, x, y, x + image.width, y + image.height)
            csv_rows.append(label_record(serial, page_index + 1, slot + 1, template, arrangement.rotated,
                                         content, label, clean_stock))
            json_records.append({**csv_rows[-1], "check_label": label.to_dict()})
        footer = f"Check Transcriber mock checks, page {page_index + 1} of {page_count}. Fictional data; print at 100% scale."
        draw.text((LETTER_SIZE_PX[0] // 2, LETTER_SIZE_PX[1] - 40), footer, fill=FOOTER_RGB,
                  font=load_font(FOOTER_FONT_ID, FOOTER_FONT_PX), anchor="ms")
        page.save(output_directory / f"print_sheet__page={page_index + 1:02d}.png")
        pages.append(page)

    pdf_path = output_directory / "print_sheets.pdf"
    pages[0].save(pdf_path, save_all=True, append_images=pages[1:], resolution=RENDER_DPI, quality=PDF_JPEG_QUALITY,
                  subsampling=0)
    with open(output_directory / "print_labels.csv", "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(csv_rows)
    (output_directory / "print_labels.json").write_text(json.dumps(json_records, indent=1))
    logger.info("wrote %d pages, %d checks to %s (clean stock: %s)", page_count, len(csv_rows), output_directory,
                clean_stock)
    return pdf_path
