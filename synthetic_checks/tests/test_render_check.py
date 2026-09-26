"""Stage 1 invariants: every labeled box lies inside the check and the fake data stays fake."""

import numpy as np
import pytest

from synthetic_checks.check_fields import FieldName
from synthetic_checks.check_templates import build_template_catalog
from synthetic_checks.content.amount_words import spell_whole_dollars
from synthetic_checks.content.fake_data import aba_checksum_is_valid, fake_routing_number, sample_check_content
from synthetic_checks.render_check import render_check

CATALOG = build_template_catalog()


@pytest.mark.parametrize("template_index", range(0, len(CATALOG), 3))
def test_field_boxes_inside_check_and_non_empty(template_index):
    template = CATALOG[template_index]
    rng = np.random.default_rng(template_index)
    image, label = render_check(template, sample_check_content(template, rng, serial="S-0001"), rng)
    assert image.size == (label.width_px, label.height_px)
    for field_label in label.fields + label.preprinted_text:
        x0, y0, x1, y1 = field_label.box
        assert 0 <= x0 < x1 <= label.width_px, field_label
        assert 0 <= y0 < y1 <= label.height_px, field_label
    required = {FieldName.PAYER_NAME, FieldName.CHECK_NUMBER, FieldName.DATE, FieldName.PAYEE,
                FieldName.AMOUNT_NUMERIC, FieldName.AMOUNT_WORDS, FieldName.SIGNATURE, FieldName.MICR, FieldName.SERIAL}
    assert required <= {FieldName(f.field_name) for f in label.fields}


def test_handwritten_box_contains_ink():
    template = CATALOG[0]
    rng = np.random.default_rng(5)
    content = sample_check_content(template, rng)
    content.handwritten_fields = ["payee"]
    image, label = render_check(template, content, rng)
    x0, y0, x1, y1 = label.field_by_name(FieldName.PAYEE).box
    crop = np.asarray(image.convert("RGB"))[y0:y1, x0:x1].astype(int)
    assert (crop.sum(axis=2) < 300).mean() > 0.02  # there is dark ink inside the payee box


def test_routing_numbers_always_fail_aba_checksum():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        routing = fake_routing_number(rng)
        assert len(routing) == 9 and routing.isdigit()
        assert not aba_checksum_is_valid(routing)


def test_aba_checksum_reference_value():
    # 3(1+4+7) + 7(2+5+8) + (3+6+0) = 150, divisible by 10: a made-up number that passes.
    assert aba_checksum_is_valid("123456780")
    assert not aba_checksum_is_valid("123456781")


def test_amount_words():
    assert spell_whole_dollars(1250) == "one thousand two hundred fifty"
    assert spell_whole_dollars(3021) == "three thousand twenty-one"
    assert spell_whole_dollars(515) == "five hundred fifteen"


def test_template_catalog_is_deterministic():
    assert build_template_catalog() == build_template_catalog()
