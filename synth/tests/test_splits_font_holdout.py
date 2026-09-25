"""Printed-font hold-out: fonts reserved for eval templates never appear on a train template."""

import numpy as np

from synth.dataset.splits import plan_template_pools
from synth.render.check_templates import build_template_catalog, template_printed_font_ids
from synth.render.fake_data import FAMILIES_WITH_PREPRINTED_DOLLAR_SIGN, sample_check_content


def test_eval_reserved_printed_fonts_never_reach_train_templates():
    """For several seeds, the printed fonts of train templates and held-out templates' reserved fonts are disjoint."""
    catalog = build_template_catalog()
    family_by_id = {t.template_id: t.layout_family.value for t in catalog}
    by_id = {t.template_id: t for t in catalog}
    for seed in range(5):
        pools = plan_template_pools(family_by_id, seed)
        train_fonts = set().union(*(template_printed_font_ids(by_id[i], include_shared=False) for i in pools["train"]))
        held_out_fonts = set().union(*(template_printed_font_ids(by_id[i], include_shared=False)
                                       for split in ("val", "eval") for i in pools[split]))
        reserved = held_out_fonts - train_fonts
        assert reserved, f"seed {seed}: no printed font is exclusive to val/eval"
        for template_id in pools["train"]:
            index = int(template_id.removeprefix("tpl_"))
            assert index % 5 != 4, f"seed {seed}: font hold-out template {template_id} landed in train"


def test_personal_amount_fill_never_repeats_the_preprinted_dollar_sign():
    """Stock that prints '$' beside the box gets amounts without their own leading '$'."""
    rng = np.random.default_rng(3)
    for template in build_template_catalog():
        if template.layout_family not in FAMILIES_WITH_PREPRINTED_DOLLAR_SIGN:
            continue
        for _ in range(20):
            assert not sample_check_content(template, rng).amount_numeric_text.startswith("$")
