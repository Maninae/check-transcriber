"""Every layout family renders a full, exactly-labelled check; stock and edges behave (contract C2)."""

import dataclasses
from collections import Counter

import numpy as np
import pytest

from synth.render.check_fields import FieldName
from synth.render.check_layout import LayoutFamily
from synth.render.check_templates import BorderKind, ScenicKind, build_template_catalog, build_template_design
from synth.render.fake_data import sample_check_content
from synth.render.families import PERFORATED_FAMILIES
from synth.render.render_check import render_check
from synth.render.stock_render import render_blank_template

CATALOG = build_template_catalog()
FIRST_TEMPLATE_OF_FAMILY = {family: next(t for t in CATALOG if t.layout_family == family) for family in LayoutFamily}
ALWAYS_REQUIRED = {FieldName.PAYER_NAME, FieldName.CHECK_NUMBER, FieldName.DATE, FieldName.PAYEE, FieldName.AMOUNT_NUMERIC,
                   FieldName.AMOUNT_WORDS, FieldName.BANK_NAME, FieldName.SIGNATURE, FieldName.MICR, FieldName.SERIAL}
INK_CHANNEL_DELTA = 45          # a pixel "has ink" when some channel is this much darker than the blank stock
MIN_INKED_FRACTION = 0.01


def inked_fraction(rendered_rgb: np.ndarray, stock_rgb: np.ndarray, box) -> float:
    """Fraction of pixels in `box` noticeably darker than the blank stock (ink the field added)."""
    x0, y0, x1, y1 = box
    darkening = stock_rgb[y0:y1, x0:x1].astype(int) - rendered_rgb[y0:y1, x0:x1].astype(int)
    return float((darkening.max(axis=2) > INK_CHANNEL_DELTA).mean())


@pytest.mark.parametrize("family", list(LayoutFamily), ids=lambda family: family.value)
@pytest.mark.parametrize("seed", [0, 1])
def test_family_fields_present_inside_and_inked(family, seed):
    template = FIRST_TEMPLATE_OF_FAMILY[family]
    rng = np.random.default_rng([seed, 17])
    content = sample_check_content(template, rng, serial="S-0001")
    image, label = render_check(template, content, rng)
    assert image.mode == "RGBA" and image.size == (label.width_px, label.height_px)
    assert label.canonical["layout_family"] == family.value

    required = set(ALWAYS_REQUIRED)
    if family != LayoutFamily.PERSONAL_NAME_ONLY:
        required.add(FieldName.PAYER_ADDRESS)
    if content.memo_text:
        required.add(FieldName.MEMO)
    present = {FieldName(field_label.field_name) for field_label in label.fields}
    assert required <= present, required - present

    rendered_rgb = np.asarray(image.convert("RGB"))
    stock_rgb = render_blank_template(template).rgb
    for field_label in label.fields:
        x0, y0, x1, y1 = field_label.box
        assert 0 <= x0 < x1 <= label.width_px and 0 <= y0 < y1 <= label.height_px, field_label
        assert inked_fraction(rendered_rgb, stock_rgb, field_label.box) >= MIN_INKED_FRACTION, field_label
    for printed_label in label.preprinted_text:
        x0, y0, x1, y1 = printed_label.box
        assert 0 <= x0 < x1 <= label.width_px and 0 <= y0 < y1 <= label.height_px, printed_label


@pytest.mark.parametrize("family", list(LayoutFamily), ids=lambda family: family.value)
def test_paper_alpha_edges(family):
    template = FIRST_TEMPLATE_OF_FAMILY[family]
    rng = np.random.default_rng(3)
    image, label = render_check(template, sample_check_content(template, rng), rng)
    alpha = np.asarray(image.getchannel("A"))
    height, width = alpha.shape
    margin = 12
    assert (alpha[margin:-margin, margin:-margin] == 255).all()  # all interior is paper
    assert alpha[0, 0] < 255 and alpha[-1, -1] < 255              # rounded corners
    side = label.canonical["perforated_side"]
    if family in PERFORATED_FAMILIES:
        edge_column = alpha[margin:-margin, 0] if side == "left" else alpha[margin:-margin, -1]
        assert (edge_column == 0).mean() > 0.2   # scallops of the torn perforation
        assert (edge_column > 0).mean() > 0.05   # the nubs between them
    else:
        assert side is None
        assert (alpha[margin:-margin, 0] == 255).all() and (alpha[margin:-margin, -1] == 255).all()  # clean cut


def test_catalog_size_balance_and_determinism():
    assert len(CATALOG) >= 64
    family_counts = Counter(template.layout_family for template in CATALOG)
    assert len(family_counts) >= 5
    assert max(family_counts.values()) - min(family_counts.values()) <= 1
    assert [template.template_id for template in CATALOG[:3]] == ["tpl_000", "tpl_001", "tpl_002"]
    assert all(build_template_design(index) == CATALOG[index] for index in (0, 17, 63))
    assert len({(t.layout_family, t.paper_tint_rgb, t.pattern_kind, t.header_font_id, t.border_kind) for t in CATALOG}) == len(CATALOG)


def test_clean_stock_has_no_paper_texture_and_same_geometry():
    """Print sheets render clean stock: flat paper, no misregistration; slots and label boxes match the textured stock."""
    plain = dataclasses.replace(CATALOG[0], pattern_strength=0.0, border_kind=BorderKind.NONE, scenic_kind=ScenicKind.NONE,
                                has_security_fibres=False, template_id="tpl_plain_test")
    clean, textured = render_blank_template(plain, textured=False), render_blank_template(plain, textured=True)
    empty_patch = (slice(10, 40), slice(plain.template_index + 900, plain.template_index + 1000))  # blank area right of the payer block
    assert clean.rgb[empty_patch].std() == 0.0
    assert textured.rgb[empty_patch].std() > 0.5
    assert clean.slots == textured.slots
    assert [label.box for label in clean.preprinted] == [label.box for label in textured.preprinted]


def test_clean_render_is_opaque_and_flagged():
    template = FIRST_TEMPLATE_OF_FAMILY[LayoutFamily.PERSONAL_CLASSIC]
    rng = np.random.default_rng(4)
    image, label = render_check(template, sample_check_content(template, rng), rng, simulate_print_texture=False)
    assert (np.asarray(image.getchannel("A")) == 255).all()
    assert label.canonical["print_texture"] is False and label.canonical["perforated_side"] is None
