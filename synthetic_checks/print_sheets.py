"""Print-ready Letter pages of fake checks at true size, for Owen's real-photo gold eval set.

Print `print_sheets.pdf` at 100% scale, cut along the guides, lay the checks on a bedsheet,
photograph them. Each check carries a small printed serial (S-0007) under its memo line;
`print_labels.csv` / `.json` map every serial to every field's text, so photos can be scored
without hand transcription. `print_sheet__page=NN.png` is each page as rendered, for viewing.

- Clean stock: a real printer and paper add their own toner and fibre, so checks are rendered
  without simulated print texture (`render_check(..., simulate_print_texture=False)`) by default.
- Coverage: each page alternates slots between handwritten and printed fill-ins, and cycles
  layout families, so every sheet has both kinds of fill-in and several designs.
- `held_out_split="eval"` restricts templates, fonts, payees and banks to that seed's eval pools,
  so a model trained on the same seed's train split has never seen the designs, hands or names.
- Hand-fill (`--hand-fill`): real people write the handwritten fields. Every check is a hand-filled
  one with those fields left blank and a prompt strip under it saying exactly what to write
  (`print_hand_fill.py`); each of `--writers N` gets the same checks with a writer letter in the
  serial (S-0007-B) behind an instruction page (`print_instruction_page.py`). Labels add `writer`,
  `hand_fill` and `prompt_strip_text`; `handwritten_fields` lists every blank, signature included.
  Default output is unchanged by these options (`test_default_output_is_unchanged`).

Usage:
    python -m synthetic_checks.print_sheets --output DIR --pages 4
    python -m synthetic_checks.print_sheets --output DIR --pages 12 --print-pool eval --seed 1
    python -m synthetic_checks.print_sheets --output DIR --pages 6 --print-pool eval --seed 1 --hand-fill --writers 3
"""

import argparse
import csv
import dataclasses
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from synthetic_checks.check_fields import FieldName
from synthetic_checks.check_layout import RENDER_DPI, CheckSizeKind, check_size_pixels
from synthetic_checks.check_templates import DEFAULT_TEMPLATE_COUNT, TemplateDesign, build_template_catalog, template_family_by_id
from synthetic_checks.content.fake_data import sample_check_content
from synthetic_checks.content.fake_payees_and_banks import BANK_NAMES, PAYEE_NAMES
from synthetic_checks.fonts.font_registry import load_font
from synthetic_checks.print_hand_fill import (PROMPT_STRIP_HEIGHT_PX, check_with_prompt_strip, hand_fill_prompt_items,
                                              hand_fill_serial, render_prompt_strip, writer_letter)
from synthetic_checks.print_instruction_page import render_instruction_page
from synthetic_checks.print_page_layout import LETTER_SIZE_PX, arrange_checks_on_page, draw_cut_guides
from synthetic_checks.render_check import render_check
from synthetic_checks.splits import plan_registry_pools, plan_template_pools

logger = logging.getLogger(__name__)

PRINT_RNG_SALT = 9091
EXAMPLE_RNG_SALT = 9092   # the instruction page's example check, drawn apart from the printed checks
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
HAND_FILL_CSV_COLUMNS = ["serial", "writer", *CSV_COLUMNS[1:], "hand_fill", "prompt_strip_text"]


def render_check_for_print(template: TemplateDesign, content, rng: np.random.Generator, held_out_pools: dict,
                           clean_stock: bool, hand_fill: bool = False) -> tuple[Image.Image, object]:
    """Render one check as RGB (clean stock has full alpha, so dropping it loses nothing)."""
    image, label = render_check(template, content, rng, handwriting_font_ids=held_out_pools.get("handwriting"),
                                signature_font_ids=held_out_pools.get("signature"), simulate_print_texture=not clean_stock,
                                leave_handwriting_blank=hand_fill)
    return image.convert("RGB"), label


def layout_family_of(template: TemplateDesign) -> str:
    """The template's layout family name."""
    return template.layout_family.value


def sample_content_with_style(template: TemplateDesign, rng: np.random.Generator, serial: str | None, want_handwritten: bool,
                              held_out_pools: dict, require_memo: bool = False):
    """Draw fake content until its fill-in style (and, if asked, a non-empty memo) matches (deterministic by the rng)."""
    content = None
    for _ in range(MAX_CONTENT_DRAWS):
        content = sample_check_content(template, rng, serial=serial, payee_names=held_out_pools.get("payee", PAYEE_NAMES),
                                       bank_names=held_out_pools.get("bank", BANK_NAMES))
        if (fill_in_style(content) == "handwritten") == want_handwritten and (content.memo_text or not require_memo):
            return content
    logger.warning("%s: could not draw %s fill-ins on %s", serial, "handwritten" if want_handwritten else "printed",
                   template.template_id)
    return content


def fill_in_style(content) -> str:
    """What was actually drawn: money orders are always hand-filled, so a printed request can fall back."""
    return "handwritten" if len(content.handwritten_fields) >= MIN_HANDWRITTEN_FIELDS_FOR_HANDWRITTEN_STYLE else "printed"


def print_pools(template_count: int, seed: int, held_out_split: str | None) -> tuple[list[TemplateDesign], dict]:
    """Templates and name pools (fonts, payees, banks) to print from: everything, or one split's held-out pools."""
    catalog = build_template_catalog(template_count)
    if held_out_split is None:
        return catalog, {}
    template_ids = set(plan_template_pools(template_family_by_id(catalog), seed)[held_out_split])
    registry_pools = plan_registry_pools(seed)
    held_out_pools = {
        "handwriting": registry_pools["handwriting_font_ids"][held_out_split],
        "signature": registry_pools["signature_font_ids"][held_out_split],
        "payee": registry_pools["payee_names"][held_out_split],
        "bank": registry_pools["bank_names"][held_out_split],
    }
    return [template for template in catalog if template.template_id in template_ids], held_out_pools


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


def hand_fill_label_columns(record: dict, letter: str, label) -> tuple[dict, list[str]]:
    """Add the hand-fill columns to a label row; return it and the strip's items for rendering."""
    items = hand_fill_prompt_items(label)
    record = {"serial": record["serial"], "writer": letter, **record,
              "handwritten_fields": ";".join(item.field_name.value for item in items), "hand_fill": True}
    return record, items


def draw_page_footer(page: Image.Image, text: str) -> None:
    """Small grey line at the bottom centre of a page."""
    ImageDraw.Draw(page).text((LETTER_SIZE_PX[0] // 2, LETTER_SIZE_PX[1] - 40), text, fill=FOOTER_RGB,
                              font=load_font(FOOTER_FONT_ID, FOOTER_FONT_PX), anchor="ms")


def pdf_page_number(writer_index: int, page_index: int, page_count: int, hand_fill: bool) -> int:
    """1-based page in the PDF: hand-fill stacks are [instructions, check pages] per writer, in writer order."""
    return writer_index * (page_count + 1) + page_index + 2 if hand_fill else page_index + 1


def render_example_pair(templates_by_size: dict, seed: int, held_out_pools: dict,
                        clean_stock: bool) -> tuple[Image.Image, Image.Image]:
    """Instruction-page example: one personal check blank with its strip, and the same check filled by the pen model."""
    rng = np.random.default_rng([seed, PRINT_RNG_SALT, EXAMPLE_RNG_SALT])
    by_family = templates_by_size.get("personal") or next(iter(templates_by_size.values()))
    template = by_family[sorted(by_family)[0]][0]
    content = sample_content_with_style(template, rng, None, True, held_out_pools, require_memo=True)
    blank_image, blank_label = render_check_for_print(template, content, rng, held_out_pools, clean_stock, hand_fill=True)
    strip, _ = render_prompt_strip(hand_fill_prompt_items(blank_label), blank_image.width)
    filled_image, _ = render_check_for_print(template, content, rng, held_out_pools, clean_stock)
    return check_with_prompt_strip(blank_image, strip), check_with_prompt_strip(filled_image, strip)


def write_print_sheets(output_directory: Path, page_count: int, seed: int, template_count: int,
                       held_out_split: str | None = None, clean_stock: bool = True, hand_fill: bool = False,
                       writer_count: int = 1) -> Path:
    """Write the PDF, one PNG per page, `print_labels.csv` and `print_labels.json`; returns the PDF path.

    `hand_fill` leaves handwritten fields blank with a prompt strip under each check; `writer_count`
    copies of every check page (hand-fill only) go to writers A, B, ... each behind an instruction page.
    """
    if writer_count != 1 and not hand_fill:
        raise ValueError("--writers needs --hand-fill: printed copies for several writers only differ when hand-filled")
    output_directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng([seed, PRINT_RNG_SALT])
    templates, held_out_pools = print_pools(template_count, seed, held_out_split)
    families_by_size = defaultdict(lambda: defaultdict(list))
    for template in templates:
        families_by_size[template.size_kind.value][layout_family_of(template)].append(template)
    size_cycle = [kind for kind in PAGE_SIZE_KIND_CYCLE if kind in families_by_size] or sorted(families_by_size)
    family_turn = defaultdict(int)
    letters = [writer_letter(index) for index in range(writer_count)] if hand_fill else [""]
    check_pages = {letter: [] for letter in letters}
    records = {letter: [] for letter in letters}
    serial_number = 1
    for page_index in range(page_count):
        size_kind = size_cycle[page_index % len(size_cycle)]
        families = sorted(families_by_size[size_kind])
        arrangement = arrange_checks_on_page(*check_size_pixels(CheckSizeKind(size_kind)),
                                             PROMPT_STRIP_HEIGHT_PX if hand_fill else 0)
        pages = {letter: Image.new("RGB", LETTER_SIZE_PX, "white") for letter in letters}
        for slot in range(len(arrangement.origins)):
            family = families[family_turn[size_kind] % len(families)]
            family_turn[size_kind] += 1
            candidates = families_by_size[size_kind][family]
            template = candidates[int(rng.integers(len(candidates)))]
            want_handwritten = hand_fill or slot % 2 == 0
            content = sample_content_with_style(template, rng, f"S-{serial_number:04d}", want_handwritten, held_out_pools,
                                                require_memo=hand_fill)
            for writer_index, letter in enumerate(letters):
                serial = hand_fill_serial(serial_number, letter) if hand_fill else content.serial
                image, label = render_check_for_print(template, dataclasses.replace(content, serial=serial), rng,
                                                      held_out_pools, clean_stock, hand_fill)
                check_width, check_height = image.size
                record = label_record(serial, pdf_page_number(writer_index, page_index, page_count, hand_fill), slot + 1,
                                      template, arrangement.rotated, content, label, clean_stock)
                if hand_fill:
                    record, items = hand_fill_label_columns(record, letter, label)
                    strip, record["prompt_strip_text"] = render_prompt_strip(items, check_width)
                    image = check_with_prompt_strip(image, strip)
                if arrangement.rotated:
                    image = image.transpose(Image.Transpose.ROTATE_270)
                    check_width, check_height = check_height, check_width
                pages[letter].paste(image, arrangement.origins[slot])
                x, y = arrangement.check_origin(slot)
                draw_cut_guides(ImageDraw.Draw(pages[letter]), x, y, x + check_width, y + check_height)
                records[letter].append({**record, "check_label": label.to_dict()})
            serial_number += 1
        for letter in letters:
            check_pages[letter].append(pages[letter])

    pdf_pages = []
    total_pages = len(letters) * (page_count + int(hand_fill))
    example_pair = render_example_pair(families_by_size, seed, held_out_pools, clean_stock) if hand_fill else None
    for writer_index, letter in enumerate(letters):
        if hand_fill:
            first_page = pdf_page_number(writer_index, 0, page_count, True) - 1
            instructions = render_instruction_page(letter, first_page, first_page + page_count, len(records[letter]),
                                                   *example_pair)
            check_pages[letter].insert(0, instructions)
        for page in check_pages[letter]:
            pdf_pages.append(page)
            page_number = len(pdf_pages)
            who = f"writer {letter}, " if hand_fill else ""
            draw_page_footer(page, f"Check Transcriber mock checks, {who}page {page_number} of {total_pages}. "
                                   "Fictional data; print at 100% scale.")
            page.save(output_directory / f"print_sheet__page={page_number:02d}.png")

    pdf_path = output_directory / "print_sheets.pdf"
    pdf_pages[0].save(pdf_path, save_all=True, append_images=pdf_pages[1:], resolution=RENDER_DPI,
                      quality=PDF_JPEG_QUALITY, subsampling=0)
    json_records = [record for letter in letters for record in records[letter]]
    with open(output_directory / "print_labels.csv", "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=HAND_FILL_CSV_COLUMNS if hand_fill else CSV_COLUMNS)
        writer.writeheader()
        writer.writerows({key: value for key, value in record.items() if key != "check_label"} for record in json_records)
    (output_directory / "print_labels.json").write_text(json.dumps(json_records, indent=1))
    logger.info("wrote %d pages, %d checks to %s (clean stock: %s, hand-fill: %s, writers: %d)", len(pdf_pages),
                len(json_records), output_directory, clean_stock, hand_fill, len(letters))
    return pdf_path


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    """Command-line interface (also reachable as `python -m dataset_builder.build_dataset --print-sheets`)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="output directory")
    parser.add_argument("--pages", type=int, required=True, help="Letter pages to write")
    parser.add_argument("--seed", type=int, default=0, help="content seed, and the split seed for --print-pool eval")
    parser.add_argument("--templates", type=int, default=DEFAULT_TEMPLATE_COUNT, help="size of the template catalog")
    parser.add_argument("--print-pool", choices=("all", "eval"), default="all",
                        help="print from every template and font, or only this seed's eval-split pools")
    parser.add_argument("--simulated-print-texture", action="store_true",
                        help="render simulated toner and fibre too (off by default: the real printer adds its own)")
    parser.add_argument("--hand-fill", action="store_true",
                        help="leave handwritten fields blank, with a strip under each check saying what to write")
    parser.add_argument("--writers", type=int, default=1,
                        help="with --hand-fill: copies of every check page, one per writer (serials end -A, -B, ...)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for `python -m synthetic_checks.print_sheets`."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    arguments = parse_arguments(sys.argv[1:] if argv is None else argv)
    write_print_sheets(arguments.output, arguments.pages, arguments.seed, arguments.templates,
                       held_out_split="eval" if arguments.print_pool == "eval" else None,
                       clean_stock=not arguments.simulated_print_texture, hand_fill=arguments.hand_fill,
                       writer_count=arguments.writers)


if __name__ == "__main__":
    main()
