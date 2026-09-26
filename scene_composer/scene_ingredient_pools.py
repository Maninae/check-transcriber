"""The ingredients one scene may draw from: a split's templates, backgrounds, fonts, payees and banks.

Two ways to get a `SceneIngredientPools`, and they agree by construction:
- `ingredient_pools_for_split(split_pool_ids, ...)`: from pools already planned (the dataset builder's
  `build_plan.json`, which also covers procedural backgrounds and resume checks).
- `library_ingredient_pools(seed, split, background_root)`: plan this seed's split of the whole library
  on the spot (on-demand scenes and `SyntheticSceneStream`). Same partition as `make_build_plan` for the
  same seed, background root and template count, so a stream for `eval` never touches a train id.

Split membership is decided in `synthetic_checks/splits.py` (contract C4); this module only packages it.
"""

import functools
from dataclasses import dataclass
from pathlib import Path

from synthetic_backgrounds.background_traits import SceneBackground, describe_backgrounds
from synthetic_backgrounds.loader import list_background_sources
from synthetic_checks.check_templates import DEFAULT_TEMPLATE_COUNT, build_template_catalog, template_family_by_id
from synthetic_checks.splits import SPLIT_NAMES, plan_split_pools

LIBRARY_POOL_CACHE_SIZE = 16


@dataclass(frozen=True)
class SceneIngredientPools:
    """Everything a scene may combine; every tuple is one split's pool, disjoint from the other splits'."""

    template_ids: tuple[str, ...]
    backgrounds: tuple[SceneBackground, ...]   # the split's backgrounds with their surface traits
    handwriting_font_ids: tuple[str, ...]
    signature_font_ids: tuple[str, ...]
    payee_names: tuple[str, ...]
    bank_names: tuple[str, ...]


def require_known_split(split: str) -> None:
    """Fail loud on a typo like 'test': an unknown split would silently get no pools."""
    if split not in SPLIT_NAMES:
        raise ValueError(f"unknown split {split!r}; choose from {SPLIT_NAMES}")


def ingredient_pools_for_split(split_pool_ids: dict[str, list[str]], background_root: Path,
                               background_paths: dict[str, str]) -> SceneIngredientPools:
    """Package one split's planned id lists (a `SplitPools.to_dict()`) with its backgrounds' traits.

    Args:
        split_pool_ids: kind -> ids for one split, as stored in `build_plan.json`.
        background_root: root the background ids are relative to (read for `web/SOURCES.jsonl` traits).
        background_paths: background id -> file path, covering at least this split's backgrounds.
    """
    backgrounds = describe_backgrounds(background_root, [(background_id, background_paths[background_id])
                                                         for background_id in split_pool_ids["background_ids"]])
    return SceneIngredientPools(
        template_ids=tuple(split_pool_ids["template_ids"]), backgrounds=backgrounds,
        handwriting_font_ids=tuple(split_pool_ids["handwriting_font_ids"]),
        signature_font_ids=tuple(split_pool_ids["signature_font_ids"]),
        payee_names=tuple(split_pool_ids["payee_names"]), bank_names=tuple(split_pool_ids["bank_names"]),
    )


@functools.lru_cache(maxsize=LIBRARY_POOL_CACHE_SIZE)
def library_ingredient_pools(seed: int, split: str, background_root: Path,
                             template_count: int = DEFAULT_TEMPLATE_COUNT) -> SceneIngredientPools:
    """This seed's pools for `split`, planned over every template and every accepted background under the root.

    - Cached per process: backgrounds added to the drive later are seen by a new process, not this one.
    - Raises FileNotFoundError when the root has no accepted subfolder (see `synthetic_backgrounds/loader.py`).
    """
    require_known_split(split)
    background_paths = {source.background_id: str(source.file_path) for source in list_background_sources(background_root)}
    if len(background_paths) < len(SPLIT_NAMES):
        raise ValueError(f"need at least {len(SPLIT_NAMES)} backgrounds under {background_root}, found {len(background_paths)}")
    pools = plan_split_pools(template_family_by_id(build_template_catalog(template_count)), sorted(background_paths), seed)
    return ingredient_pools_for_split(pools[split].to_dict(), background_root, background_paths)
