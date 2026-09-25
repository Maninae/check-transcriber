"""Print mode: Letter PDF + page PNGs + labels keyed by a unique printed serial, both fill-in styles per page."""

import csv
import json

import pytest

from synth.print.print_page_layout import LETTER_SIZE_PX, arrange_checks_on_page
from synth.print.print_sheets import write_print_sheets
from synth.render.check_layout import CheckSizeKind, check_size_pixels


@pytest.fixture(scope="module")
def printed(tmp_path_factory):
    output_directory = tmp_path_factory.mktemp("print")
    pdf_path = write_print_sheets(output_directory, page_count=2, seed=0, template_count=24)
    return output_directory, pdf_path


def test_pdf_pngs_and_labels_are_written(printed):
    output_directory, pdf_path = printed
    assert pdf_path.read_bytes()[:5] == b"%PDF-"
    assert b"MediaBox [ 0 0 612.0 792.0 ]" in pdf_path.read_bytes()
    assert sorted(path.name for path in output_directory.glob("*.png")) == ["print_sheet__page=01.png", "print_sheet__page=02.png"]
    rows = list(csv.DictReader(open(output_directory / "print_labels.csv")))
    serials = [row["serial"] for row in rows]
    assert serials[:3] == ["S-0001", "S-0002", "S-0003"] and len(set(serials)) == len(serials)
    records = json.loads((output_directory / "print_labels.json").read_text())
    assert [record["serial"] for record in records] == serials


def test_every_check_carries_its_serial_and_field_texts(printed):
    output_directory, _ = printed
    for record in json.loads((output_directory / "print_labels.json").read_text()):
        fields = {field["field_name"]: field["text"] for field in record["check_label"]["fields"]}
        assert fields["serial"] == record["serial"]
        assert record["text__payee"] == fields["payee"] and record["text__amount_numeric"] == fields["amount_numeric"]


def test_print_sheets_render_clean_stock(printed):
    """Real printer and paper supply toner grain and fibre, so printed checks come from the clean path."""
    output_directory, _ = printed
    records = json.loads((output_directory / "print_labels.json").read_text())
    assert records and all(record["check_label"]["canonical"]["print_texture"] is False for record in records)
    assert all(record["clean_stock"] for record in records)


def test_each_page_has_handwritten_and_printed_fill_ins(printed):
    output_directory, _ = printed
    rows = list(csv.DictReader(open(output_directory / "print_labels.csv")))
    for page in {row["page"] for row in rows}:
        page_rows = [row for row in rows if row["page"] == page]
        assert {row["fill_in_style"] for row in page_rows} == {"handwritten", "printed"}
        assert any(row["handwritten_fields"] for row in page_rows if row["fill_in_style"] == "handwritten")
        assert all(not row["handwritten_fields"] for row in page_rows if row["fill_in_style"] == "printed")


@pytest.mark.parametrize("size_kind, rotated, count", [(CheckSizeKind.PERSONAL, False, 3), (CheckSizeKind.BUSINESS, True, 2)])
def test_true_size_checks_fit_the_letter_page(size_kind, rotated, count):
    width, height = check_size_pixels(size_kind)
    arrangement = arrange_checks_on_page(width, height)
    assert arrangement.rotated == rotated and len(arrangement.origins) == count
    placed = (height, width) if rotated else (width, height)
    for x, y in arrangement.origins:
        assert x >= 0 and y >= 0 and x + placed[0] <= LETTER_SIZE_PX[0] and y + placed[1] <= LETTER_SIZE_PX[1]
