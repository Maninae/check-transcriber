"""Printed typefaces: open licenses, glyph coverage, role/kind pools, and the template font hold-out."""

import warnings
from collections import Counter

import pytest

from synth.paths import FONT_DIR
from synth.render.check_layout import CheckSizeKind
from synth.render.check_templates import build_template_catalog, build_template_design, template_printed_font_ids
from synth.render.fonts import APACHE_LICENSE, FONT_SPECS, OFL_LICENSE, FontRole, font_ids_with_role, load_font
from synth.render.handwriting_font_metrics import missing_characters
from synth.render.printed_font_pools import (
    HOLDOUT_FONT_IDS, PRINTED_FONT_PROFILES, PRINTED_FONT_PROFILES_BY_ID, ROLE_STYLE_WEIGHTS_BY_KIND,
    SHARED_PRINTED_FONT_IDS, PrintedRole, is_holdout_template_index, printed_font_pool, profile_suits_role,
)

# Everything laser or offset printing puts on a check: words, amounts, dates, addresses, asterisk fill.
PRINTED_CHARACTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" + "$,./-#&'():*"
PRINTED_FONT_SPECS = [spec for spec in FONT_SPECS if spec.role == FontRole.PRINTED]
CATALOG = build_template_catalog()
MIN_PRINTED_FONTS = 24
MIN_DISTINCT_FONTS_IN_CATALOG = 30
MIN_HOLDOUT_SHARE_ON_HOLDOUT_TEMPLATES = 0.6


def template_role_fonts(template) -> dict[PrintedRole, str]:
    """The template's font for each printed role."""
    return {
        PrintedRole.HEADER: template.header_font_id,
        PrintedRole.BODY: template.body_font_id,
        PrintedRole.LABEL: template.label_font_id,
        PrintedRole.FILL: template.printed_fill_font_id,
    }


def test_printed_font_pool_is_large_and_open_licensed():
    assert len(PRINTED_FONT_SPECS) >= MIN_PRINTED_FONTS
    for spec in PRINTED_FONT_SPECS:
        assert spec.license_name in (OFL_LICENSE, APACHE_LICENSE), spec.font_id
        assert "google/fonts" in spec.source_url and ("/ofl/" in spec.source_url or "/apache/" in spec.source_url), spec.font_id


def test_every_printed_font_has_a_pool_profile():
    assert set(PRINTED_FONT_PROFILES_BY_ID) == set(font_ids_with_role(FontRole.PRINTED))
    assert len(PRINTED_FONT_PROFILES_BY_ID) == len(PRINTED_FONT_PROFILES)


@pytest.mark.parametrize("spec", PRINTED_FONT_SPECS, ids=lambda spec: spec.font_id)
def test_printed_font_covers_every_character_we_print(spec):
    font_path = FONT_DIR / spec.relative_path
    if not font_path.exists():
        warnings.warn(f"font {spec.font_id} not fetched ({font_path}); run python -m synth.render.fetch_fonts")
        pytest.skip(f"MISSING FONT FILE {font_path}")
    assert missing_characters(spec.font_id, PRINTED_CHARACTERS) == []
    if spec.is_variable:  # the pinned named instance must exist in the file
        variation_names = [name.decode() for name in load_font(spec.font_id, 32).get_variation_names()]
        assert spec.instance_name in variation_names


@pytest.mark.parametrize("kind", list(CheckSizeKind))
@pytest.mark.parametrize("role", list(PrintedRole))
def test_every_kind_and_role_has_regular_and_holdout_fonts(kind, role):
    assert printed_font_pool(kind, role, holdout=False)
    assert printed_font_pool(kind, role, holdout=True)


def test_shared_fonts_are_never_held_out():
    assert not set(SHARED_PRINTED_FONT_IDS) & HOLDOUT_FONT_IDS
    assert 0.15 <= len(HOLDOUT_FONT_IDS) / len(PRINTED_FONT_PROFILES) <= 0.3


def test_template_fonts_suit_their_role_and_check_kind():
    for template in CATALOG:
        for role, font_id in template_role_fonts(template).items():
            profile = PRINTED_FONT_PROFILES_BY_ID[font_id]
            assert profile.style in ROLE_STYLE_WEIGHTS_BY_KIND[template.size_kind][role], (template.template_id, role, font_id)
            assert profile_suits_role(profile, role), (template.template_id, role, font_id)


def test_holdout_fonts_only_appear_on_holdout_class_templates():
    for template in CATALOG:
        if not is_holdout_template_index(template.template_index):
            assert not set(template_printed_font_ids(template)) & HOLDOUT_FONT_IDS, template.template_id
    holdout_role_fonts = [font_id for template in CATALOG if is_holdout_template_index(template.template_index)
                          for font_id in template_role_fonts(template).values()]
    assert sum(font_id in HOLDOUT_FONT_IDS for font_id in holdout_role_fonts) / len(holdout_role_fonts) \
        >= MIN_HOLDOUT_SHARE_ON_HOLDOUT_TEMPLATES


def test_catalog_uses_a_wide_spread_of_typefaces():
    used = Counter(font_id for template in CATALOG for font_id in template_printed_font_ids(template))
    assert len(used) >= MIN_DISTINCT_FONTS_IN_CATALOG
    assert len({template.header_font_id for template in CATALOG}) >= 15


def test_template_fonts_are_deterministic_and_shared_fonts_are_opt_in():
    template = build_template_design(17)
    assert template_printed_font_ids(template) == template_printed_font_ids(build_template_design(17))
    with_shared = template_printed_font_ids(template, include_shared=True)
    assert set(SHARED_PRINTED_FONT_IDS) <= set(with_shared)
    assert set(template_printed_font_ids(template)) <= set(with_shared)
