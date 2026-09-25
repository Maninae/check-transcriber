"""Print mode writes a Letter PDF and a labels CSV keyed by the printed serial."""

import csv

from synth.print.print_sheets import write_print_sheets


def test_print_sheets(tmp_path):
    pdf_path = write_print_sheets(tmp_path, page_count=1, seed=0, template_count=24)
    assert pdf_path.read_bytes()[:5] == b"%PDF-"
    assert b"MediaBox [ 0 0 612.0 792.0 ]" in pdf_path.read_bytes()
    rows = list(csv.DictReader(open(tmp_path / "print_labels.csv")))
    assert [row["serial"] for row in rows] == ["S-0001", "S-0002", "S-0003"]
