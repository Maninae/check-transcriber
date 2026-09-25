"""Split rules: templates and backgrounds are partitioned, never individual scenes.

Synthetic scenes reuse templates and backgrounds thousands of times, so a random scene
split would leak both into eval and inflate every score. Instead each template id and each
background id belongs to exactly one split, and a scene may only combine a template and a
background from its own split. Assignment is a seeded shuffle, stable for a given seed and id set.
"""

import math

import numpy as np

SPLIT_NAMES = ("train", "val", "eval")
DEFAULT_SPLIT_FRACTIONS = {"train": 0.8, "val": 0.1, "eval": 0.1}
TEMPLATE_SALT = 11
BACKGROUND_SALT = 23


def assign_ids_to_splits(ids: list[str], fractions: dict[str, float], seed: int, salt: int) -> dict[str, list[str]]:
    """Partition `ids` across splits in proportion to `fractions`; every split gets at least one id."""
    if len(ids) < len(SPLIT_NAMES):
        raise ValueError(f"need at least {len(SPLIT_NAMES)} ids to fill every split, got {len(ids)}")
    shuffled = [ids[i] for i in np.random.default_rng([seed, salt]).permutation(len(ids))]
    val_count = max(1, math.floor(len(ids) * fractions["val"]))
    eval_count = max(1, math.floor(len(ids) * fractions["eval"]))
    return {
        "eval": sorted(shuffled[:eval_count]),
        "val": sorted(shuffled[eval_count:eval_count + val_count]),
        "train": sorted(shuffled[eval_count + val_count:]),
    }


def scene_counts_per_split(total_scenes: int, fractions: dict[str, float]) -> dict[str, int]:
    """Scene count per split; rounding remainder goes to train."""
    counts = {name: int(total_scenes * fractions[name]) for name in ("val", "eval")}
    counts["train"] = total_scenes - counts["val"] - counts["eval"]
    return {name: counts[name] for name in SPLIT_NAMES}
