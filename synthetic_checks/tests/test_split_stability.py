"""Split stability: open pools never move an id when grown, registries move at most one per boundary; scene counts."""

import numpy as np
import pytest

from synthetic_checks.check_templates import build_template_catalog
from synthetic_checks.content.fake_data import sample_check_content
from synthetic_checks.content.fake_payees_and_banks import BANK_NAMES, PAYEE_NAMES
from synthetic_checks.splits import (
    DEFAULT_SPLIT_FRACTIONS,
    SPLIT_NAMES,
    assign_ids_by_hash_rank,
    assign_ids_by_hash_threshold,
    assign_open_pool_ids,
    plan_split_pools,
    scene_counts_per_split,
)


def split_of_each_id(splits: dict[str, list[str]]) -> dict[str, str]:
    return {item_id: split_name for split_name, ids in splits.items() for item_id in ids}


@pytest.mark.parametrize("seed", range(5))
def test_adding_backgrounds_never_moves_an_existing_one(seed):
    old_ids = [f"flux/bg_{i:04d}.jpg" for i in range(60)]
    new_ids = old_ids + [f"flux/bg_{i:04d}.jpg" for i in range(60, 300)] + [f"photos/p_{i}.jpg" for i in range(40)]
    before = split_of_each_id(assign_open_pool_ids(old_ids, DEFAULT_SPLIT_FRACTIONS, seed, salt=23))
    after = split_of_each_id(assign_open_pool_ids(new_ids, DEFAULT_SPLIT_FRACTIONS, seed, salt=23))
    assert {item_id: after[item_id] for item_id in old_ids} == before


def test_open_pools_stay_balanced_while_small_and_keep_minimums_once_large():
    for seed in range(30):
        small = assign_open_pool_ids([f"flux/{i}.jpg" for i in range(12)], DEFAULT_SPLIT_FRACTIONS, seed, salt=23)
        assert [len(small[name]) for name in SPLIT_NAMES] == [8, 2, 2]
        large = assign_ids_by_hash_threshold([f"flux/{i}.jpg" for i in range(50)], DEFAULT_SPLIT_FRACTIONS, seed, salt=23)
        assert len(large["val"]) >= 2 and len(large["eval"]) >= 2 and 0.55 <= len(large["train"]) / 50 <= 0.85
        tiny = assign_ids_by_hash_threshold(["a", "b", "c"], DEFAULT_SPLIT_FRACTIONS, seed, salt=23)
        assert all(len(tiny[name]) == 1 for name in SPLIT_NAMES)


@pytest.mark.parametrize("seed", range(10))
def test_adding_one_registry_entry_moves_at_most_one_id_per_split_boundary(seed):
    fonts = [f"font_{i}" for i in range(34)]
    before = split_of_each_id(assign_ids_by_hash_rank(fonts, DEFAULT_SPLIT_FRACTIONS, seed, salt=37))
    after = split_of_each_id(assign_ids_by_hash_rank(fonts + ["font_new"], DEFAULT_SPLIT_FRACTIONS, seed, salt=37))
    assert sum(before[font] != after[font] for font in fonts) <= 2


def test_eval_reserved_templates_never_train_even_when_a_family_has_many():
    family_by_id = {f"tpl_{i:03d}": "family" for i in range(0, 60, 1)}
    for seed in range(10):
        pools = plan_split_pools(family_by_id, ["a", "b", "c"], seed)
        assert not any(int(t.removeprefix("tpl_")) % 5 == 4 for t in pools["train"].template_ids)


def test_real_template_catalog_puts_every_family_in_val_and_eval_for_many_seeds():
    catalog = build_template_catalog()
    family_by_id = {template.template_id: template.layout_family.value for template in catalog}
    for seed in range(30):
        pools = plan_split_pools(family_by_id, ["a", "b", "c", "d"], seed)
        for family in set(family_by_id.values()):
            for split_name in ("val", "eval"):
                assert any(family_by_id[t] == family for t in pools[split_name].template_ids), (seed, family, split_name)


@pytest.mark.parametrize("total", range(3, 25))
def test_every_split_gets_a_scene_from_three_scenes_up(total):
    counts = scene_counts_per_split(total, DEFAULT_SPLIT_FRACTIONS)
    assert sum(counts.values()) == total and all(counts[name] >= 1 for name in SPLIT_NAMES)


def test_payee_and_bank_pools_partition_their_lists_and_content_respects_them():
    pools = plan_split_pools({f"tpl_{i:03d}": "family" for i in range(24)}, ["a", "b", "c", "d"], seed=4)
    for kind, names in (("payee_names", PAYEE_NAMES), ("bank_names", BANK_NAMES)):
        split_lists = [getattr(pools[name], kind) for name in SPLIT_NAMES]
        assert sorted(sum(split_lists, [])) == sorted(names) and all(len(ids) >= 2 for ids in split_lists)
    template = build_template_catalog()[0]
    rng = np.random.default_rng(0)
    for _ in range(50):
        content = sample_check_content(template, rng, payee_names=pools["eval"].payee_names, bank_names=pools["eval"].bank_names)
        assert content.payee_canonical in pools["eval"].payee_names and content.bank_name in pools["eval"].bank_names
