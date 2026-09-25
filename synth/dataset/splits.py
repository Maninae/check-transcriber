"""Split rules (contract C4): templates, backgrounds and fonts are partitioned, never individual scenes.

Synthetic scenes reuse templates, backgrounds and handwriting fonts thousands of times, so a
random scene split would leak all three into eval and inflate every score. Instead each id of
each kind belongs to exactly one split, and a scene only combines ids from its own split:
eval checks are printed on stock, laid on surfaces and written by hands train never saw.

- Four independent partitions: template ids, background ids, handwriting font ids, signature font ids.
- Assignment is a seeded shuffle per kind (seed + a per-kind salt), stable for a given seed and id set.
- Templates are stratified by layout family: each family is partitioned on its own, so every
  family appears in val and eval (11 templates per family -> 7 / 2 / 2).
- Templates that carry eval-reserved printed fonts (`is_holdout_template_index`) are dealt to
  eval and val before any other template of their family, so those fonts never reach train.
- Val and eval sizes are rounded, not floored, so small pools (13 signature fonts) still give
  val and eval two ids each instead of one.
"""

import zlib
from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np

from synth.render.fonts import FontRole, font_ids_with_role
from synth.render.printed_font_pools import is_holdout_template_index

SPLIT_NAMES = ("train", "val", "eval")
DEFAULT_SPLIT_FRACTIONS = {"train": 0.7, "val": 0.15, "eval": 0.15}
TEMPLATE_SALT = 11
BACKGROUND_SALT = 23
HANDWRITING_FONT_SALT = 37
SIGNATURE_FONT_SALT = 41


@dataclass(frozen=True)
class SplitPools:
    """The ids one split may draw from; every list is disjoint from the other splits' lists."""

    template_ids: list[str]
    background_ids: list[str]
    handwriting_font_ids: list[str]
    signature_font_ids: list[str]

    def to_dict(self) -> dict:
        """Plain-JSON form."""
        return asdict(self)


def held_out_count(total: int, fraction: float) -> int:
    """How many of `total` ids a held-out split (val or eval) gets: rounded, at least one."""
    return max(1, round(total * fraction))


def assign_ids_to_splits(ids: list[str], fractions: dict[str, float], seed: int, salt: int,
                         stream: int = 0, held_out_first: frozenset[str] = frozenset()) -> dict[str, list[str]]:
    """Partition `ids` across splits in proportion to `fractions`; every split gets at least one id.

    `stream` separates independent shuffles under the same salt (one per stratum). Ids in
    `held_out_first` are dealt to eval, then val, before any other id.
    """
    if len(ids) < len(SPLIT_NAMES):
        raise ValueError(f"need at least {len(SPLIT_NAMES)} ids to fill every split, got {len(ids)}")
    shuffled = [ids[i] for i in np.random.default_rng([seed, salt, stream]).permutation(len(ids))]
    shuffled = [i for i in shuffled if i in held_out_first] + [i for i in shuffled if i not in held_out_first]
    eval_count = held_out_count(len(ids), fractions["eval"])
    val_count = held_out_count(len(ids), fractions["val"])
    if eval_count + val_count >= len(ids):  # tiny pools: leave train at least one id
        eval_count, val_count = 1, 1
    return {
        "train": sorted(shuffled[eval_count + val_count:]),
        "val": sorted(shuffled[eval_count:eval_count + val_count]),
        "eval": sorted(shuffled[:eval_count]),
    }


def assign_ids_to_splits_stratified(group_by_id: dict[str, str], fractions: dict[str, float], seed: int,
                                    salt: int, held_out_first: frozenset[str] = frozenset()) -> dict[str, list[str]]:
    """Partition each group (stratum) separately and merge; groups too small to split are pooled together."""
    ids_by_group = defaultdict(list)
    for item_id, group in sorted(group_by_id.items()):
        ids_by_group[group].append(item_id)
    small_group_ids = [item_id for ids in ids_by_group.values() if len(ids) < len(SPLIT_NAMES) for item_id in ids]
    strata = [(group, ids) for group, ids in sorted(ids_by_group.items()) if len(ids) >= len(SPLIT_NAMES)]
    if small_group_ids:
        strata.append(("pooled_small_groups", small_group_ids))
    merged = {split_name: [] for split_name in SPLIT_NAMES}
    for group, ids in strata:
        if len(ids) < len(SPLIT_NAMES):  # only when the pooled leftovers are still too few: they train
            merged["train"] += ids
            continue
        for split_name, split_ids in assign_ids_to_splits(ids, fractions, seed, salt, zlib.crc32(group.encode()),
                                                         held_out_first).items():
            merged[split_name] += split_ids
    return {split_name: sorted(ids) for split_name, ids in merged.items()}


def plan_template_pools(template_family_by_id: dict[str, str], seed: int,
                        fractions: dict[str, float] = DEFAULT_SPLIT_FRACTIONS) -> dict[str, list[str]]:
    """Template ids per split, stratified by layout family; eval-reserved-font templates go to eval/val first."""
    font_holdout_ids = frozenset(t for t in template_family_by_id if is_holdout_template_index(int(t.removeprefix("tpl_"))))
    return assign_ids_to_splits_stratified(template_family_by_id, fractions, seed, TEMPLATE_SALT, font_holdout_ids)


def plan_split_pools(template_family_by_id: dict[str, str], background_ids: list[str], seed: int,
                     fractions: dict[str, float] = DEFAULT_SPLIT_FRACTIONS) -> dict[str, SplitPools]:
    """Partition every id kind at once; handwriting and signature fonts come from the font registry."""
    by_kind = {
        "template_ids": plan_template_pools(template_family_by_id, seed, fractions),
        "background_ids": assign_ids_to_splits(background_ids, fractions, seed, BACKGROUND_SALT),
        "handwriting_font_ids": assign_ids_to_splits(font_ids_with_role(FontRole.HANDWRITING), fractions, seed,
                                                     HANDWRITING_FONT_SALT),
        "signature_font_ids": assign_ids_to_splits(font_ids_with_role(FontRole.SIGNATURE), fractions, seed,
                                                   SIGNATURE_FONT_SALT),
    }
    return {split_name: SplitPools(**{kind: assignment[split_name] for kind, assignment in by_kind.items()})
            for split_name in SPLIT_NAMES}


def scene_counts_per_split(total_scenes: int, fractions: dict[str, float]) -> dict[str, int]:
    """Scene count per split; rounding remainder goes to train."""
    counts = {name: int(total_scenes * fractions[name]) for name in ("val", "eval")}
    counts["train"] = total_scenes - counts["val"] - counts["eval"]
    return {name: counts[name] for name in SPLIT_NAMES}
