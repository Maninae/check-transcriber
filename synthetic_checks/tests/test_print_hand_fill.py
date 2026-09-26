"""Hand-fill print mode: blank handwritten fields, prompt strips outside the cut line, writer copies, exact labels."""

import csv
import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from synthetic_checks.check_layout import RENDER_DPI, CheckSizeKind, check_size_pixels, inches_to_px
from synthetic_checks.check_templates import build_template_catalog
from synthetic_checks.content.fake_data import sample_check_content
from synthetic_checks.print_hand_fill import PROMPT_ITEM_SEPARATOR, PROMPT_LABEL_BY_FIELD, PROMPT_STRIP_HEIGHT_PX
from synthetic_checks.check_fields import FieldName
from synthetic_checks.print_page_layout import arrange_checks_on_page
from synthetic_checks.print_sheets import write_print_sheets
from synthetic_checks.render_check import SERIAL_MICR_CLEARANCE_INCHES, render_check
from synthetic_checks.stock.stock_render import render_blank_template

TEMPLATE_COUNT = 24
WRITER_COUNT = 3
PAGE_COUNT = 2                  # page 1 personal (upright), page 2 business (rotated)
INK_DARKER_THAN_STOCK = 60      # a pixel this much darker than the blank stock is new ink
MAX_NEW_INK_FRACTION = 0.001

# Default-mode output from main (b728f14) before hand-fill existed: seed 0, 2 pages, 24 templates.
# Page hashes are over decoded pixels, so they do not depend on the PNG encoder.
MAIN_DEFAULT_PAGE_PIXEL_SHA256 = {
    "print_sheet__page=01.png": "0984af45061d59b1ad39c9706a9b6a2807bffa92c58bb622ee656698757185d4",
    "print_sheet__page=02.png": "7416547dae6d4be9dee9f59edf550056eff891e23f1a202174b8d5b20b8c049a",
}
MAIN_DEFAULT_CSV_SHA256 = "2535752e78bf61a3b66bb039f528cc7a91f7002983a1d91e4d339d4ec99d468b"
MAIN_DEFAULT_JSON_SHA256 = "85048edfc47e6bfb0588fb02f18f6168e3eed5ef361449007d04dd736abcfa6b"


@pytest.fixture(scope="module")
def hand_filled(tmp_path_factory):
    output_directory = tmp_path_factory.mktemp("hand_fill")
    write_print_sheets(output_directory, page_count=PAGE_COUNT, seed=0, template_count=TEMPLATE_COUNT, hand_fill=True,
                       writer_count=WRITER_COUNT)
    rows = list(csv.DictReader(open(output_directory / "print_labels.csv")))
    records = json.loads((output_directory / "print_labels.json").read_text())
    return output_directory, rows, records


def check_crop_in_check_frame(page: np.ndarray, record: dict, tile: bool = False) -> np.ndarray:
    """The check (or its whole tile, strip included) cut out of the page and turned back upright."""
    width, height = check_size_pixels(CheckSizeKind(record["size_kind"]))
    arrangement = arrange_checks_on_page(width, height, PROMPT_STRIP_HEIGHT_PX)
    slot = record["slot"] - 1
    x, y = arrangement.origins[slot] if tile else arrangement.check_origin(slot)
    frame_height = height + (PROMPT_STRIP_HEIGHT_PX if tile else 0)
    placed_width, placed_height = (frame_height, width) if arrangement.rotated else (width, frame_height)
    crop = page[y:y + placed_height, x:x + placed_width]
    return np.rot90(crop) if arrangement.rotated else crop   # undo the 90 degree clockwise turn


def pages_by_number(output_directory) -> dict[int, np.ndarray]:
    return {int(path.stem.split("=")[1]): np.asarray(Image.open(path)) for path in output_directory.glob("print_sheet__page=*.png")}


def test_writers_get_serial_suffixed_copies_of_the_same_checks(hand_filled):
    output_directory, rows, _ = hand_filled
    assert len(list(output_directory.glob("print_sheet__page=*.png"))) == WRITER_COUNT * (PAGE_COUNT + 1)
    by_writer = {letter: [row for row in rows if row["writer"] == letter] for letter in "ABC"}
    assert sorted({row["writer"] for row in rows}) == ["A", "B", "C"]
    base_serials = [row["serial"].rsplit("-", 1)[0] for row in by_writer["A"]]
    for letter, writer_rows in by_writer.items():
        assert [row["serial"] for row in writer_rows] == [f"{serial}-{letter}" for serial in base_serials]
        for row, row_a in zip(writer_rows, by_writer["A"]):
            assert {key: value for key, value in row.items() if key.startswith("text__") and key != "text__serial"} == \
                   {key: value for key, value in row_a.items() if key.startswith("text__") and key != "text__serial"}
    instruction_pages = {writer_index * (PAGE_COUNT + 1) + 1 for writer_index in range(WRITER_COUNT)}
    assert not instruction_pages & {int(row["page"]) for row in rows}


def test_labels_carry_writer_and_the_exact_prompt_text(hand_filled):
    _, rows, records = hand_filled
    for row, record in zip(rows, records):
        assert row["hand_fill"] == "True" and row["serial"].endswith(f"-{row['writer']}")
        assert row["handwriting_font_id"] == "" and row["fill_in_style"] == "handwritten"
        filled = row["handwritten_fields"].split(";")
        assert "signature" in filled and "payee" in filled and "memo" in filled
        printed_items = [item for line in row["prompt_strip_text"].split("\n") for item in line.split(PROMPT_ITEM_SEPARATOR)]
        assert printed_items == [f"{PROMPT_LABEL_BY_FIELD[FieldName(field)]}: {row[f'text__{field}']}" for field in filled]
        label_texts = {field["field_name"]: field["text"] for field in record["check_label"]["fields"]}
        assert all(label_texts[field] == row[f"text__{field}"] for field in filled)
        assert {field["field_name"] for field in record["check_label"]["fields"] if field["handwritten"]} == set(filled)


def test_handwritten_fields_are_blank_on_the_printed_page(hand_filled):
    """Inside every blank's writing region the page matches the clean blank stock: rules and labels only, no ink."""
    output_directory, _, records = hand_filled
    pages = pages_by_number(output_directory)
    templates = {template.template_id: template for template in build_template_catalog(TEMPLATE_COUNT)}
    checked_regions = 0
    for record in records:
        check = check_crop_in_check_frame(pages[record["page"]], record).astype(np.int16)
        stock = render_blank_template(templates[record["template_id"]], RENDER_DPI, False).rgb.astype(np.int16)
        for field in record["check_label"]["fields"]:
            if not field["handwritten"]:
                continue
            x0, y0, x1, y1 = field["box"]
            new_ink = (stock[y0:y1, x0:x1] - check[y0:y1, x0:x1]).max(axis=2) > INK_DARKER_THAN_STOCK
            assert new_ink.mean() < MAX_NEW_INK_FRACTION, (record["serial"], field["field_name"], new_ink.mean())
            checked_regions += 1
    assert checked_regions >= 6 * len(records)


@pytest.mark.parametrize("template_index", [0, 3, 5])   # personal classic, business voucher, money order
def test_blank_region_measure_sees_real_handwriting(template_index):
    """Positive control for the blank test: the same measure on the pen-filled check finds ink in every region."""
    template = build_template_catalog(TEMPLATE_COUNT)[template_index]
    stock = render_blank_template(template, RENDER_DPI, False).rgb.astype(np.int16)
    content = None
    for attempt in range(50):
        content = sample_check_content(template, np.random.default_rng([attempt]))
        if content.handwritten_fields and content.memo_text:
            break
    _, blank_label = render_check(template, content, np.random.default_rng(1), simulate_print_texture=False,
                                  leave_handwriting_blank=True)
    filled_image, _ = render_check(template, content, np.random.default_rng(1), simulate_print_texture=False)
    filled = np.asarray(filled_image.convert("RGB")).astype(np.int16)
    for field in blank_label.fields:
        if field.handwritten:
            x0, y0, x1, y1 = field.box
            new_ink = (stock[y0:y1, x0:x1] - filled[y0:y1, x0:x1]).max(axis=2) > INK_DARKER_THAN_STOCK
            assert new_ink.mean() > 10 * MAX_NEW_INK_FRACTION, (template.template_id, field.field_name, new_ink.mean())


def test_prompt_strip_lies_outside_the_cut_rectangle(hand_filled):
    """The strip's text sits below the check's bottom edge (check frame), clear of the cut line, and is printed."""
    output_directory, _, records = hand_filled
    pages = pages_by_number(output_directory)
    clear_rows = inches_to_px(0.03, RENDER_DPI)
    for record in records:
        tile = check_crop_in_check_frame(pages[record["page"]], record, tile=True)
        check_height = record["check_label"]["height_px"]
        strip = tile[check_height + 2:]      # below the dashed cut line on the check's bottom edge
        dark = strip.min(axis=2) < 128
        assert dark.sum() > 500, record["serial"]
        dark_rows = np.flatnonzero(dark[:, 50:-50].any(axis=1))   # ignore corner cut marks at the tile's sides
        assert dark_rows.min() >= clear_rows, record["serial"]


def test_hand_fill_serial_clears_the_micr_line(hand_filled):
    _, _, records = hand_filled
    clearance = inches_to_px(SERIAL_MICR_CLEARANCE_INCHES, RENDER_DPI) - 2
    for record in records:
        boxes = {field["field_name"]: field["box"] for field in record["check_label"]["fields"]}
        serial_box, micr_box = boxes["serial"], boxes["micr"]
        shares_row = serial_box[1] < micr_box[3] and micr_box[1] < serial_box[3]
        assert boxes and (not shares_row or serial_box[2] + clearance <= micr_box[0]), record["serial"]


def test_writers_need_hand_fill(tmp_path):
    with pytest.raises(ValueError, match="--hand-fill"):
        write_print_sheets(tmp_path, page_count=1, seed=0, template_count=TEMPLATE_COUNT, writer_count=2)


def test_default_output_is_unchanged(tmp_path):
    """Without --hand-fill the pages and labels are byte-for-byte what main produced before hand-fill existed."""
    write_print_sheets(tmp_path, page_count=2, seed=0, template_count=TEMPLATE_COUNT)
    page_hashes = {path.name: hashlib.sha256(np.asarray(Image.open(path)).tobytes()).hexdigest()
                   for path in sorted(tmp_path.glob("*.png"))}
    assert page_hashes == MAIN_DEFAULT_PAGE_PIXEL_SHA256
    assert hashlib.sha256((tmp_path / "print_labels.csv").read_bytes()).hexdigest() == MAIN_DEFAULT_CSV_SHA256
    assert hashlib.sha256((tmp_path / "print_labels.json").read_bytes()).hexdigest() == MAIN_DEFAULT_JSON_SHA256
