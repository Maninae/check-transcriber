"""Split rules (contract C4): templates, backgrounds and fonts are partitioned, never individual scenes.

Synthetic scenes reuse templates, backgrounds and handwriting fonts thousands of times, so a
random scene split would leak all three into eval and inflate every score. Instead each id of
each kind belongs to exactly one split, and a scene only combines ids from its own split:
eval checks are printed on stock, laid on surfaces and written by hands train never saw.

- Four independent partitions: template ids, background ids, handwriting font ids, signature font ids.
- Assignment is a seeded shuffle per kind (seed + a per-kind salt), stable for a given seed and id set.
- Val and eval sizes are rounded, not floored, so small pools (13 signature fonts) still give
  val and eval two ids each instead of one.
"""

from dataclasses import asdict, dataclass

import numpy as np

from synth.render.fonts import FontRole, font_ids_with_role

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


def assign_ids_to_splits(ids: list[str], fractions: dict[str, float], seed: int, salt: int) -> dict[str, list[str]]:
    """Partition `ids` across splits in proportion to `fractions`; every split gets at least one id."""
    if len(ids) < len(SPLIT_NAMES):
        raise ValueError(f"need at least {len(SPLIT_NAMES)} ids to fill every split, got {len(ids)}")
    shuffled = [ids[i] for i in np.random.default_rng([seed, salt]).permutation(len(ids))]
    eval_count = held_out_count(len(ids), fractions["eval"])
    val_count = held_out_count(len(ids), fractions["val"])
    if eval_count + val_count >= len(ids):  # tiny pools: leave train at least one id
        eval_count, val_count = 1, 1
    return {
        "train": sorted(shuffled[eval_count + val_count:]),
        "val": sorted(shuffled[eval_count:eval_count + val_count]),
        "eval": sorted(shuffled[:eval_count]),
    }


def plan_split_pools(template_ids: list[str], background_ids: list[str], seed: int,
                     fractions: dict[str, float] = DEFAULT_SPLIT_FRACTIONS) -> dict[str, SplitPools]:
    """Partition every id kind at once; handwriting and signature fonts come from the font registry."""
    by_kind = {
        "template_ids": assign_ids_to_splits(template_ids, fractions, seed, TEMPLATE_SALT),
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
