"""Split rules (contract C4): templates, backgrounds, fonts, payees and banks are partitioned, never scenes.

Synthetic scenes reuse templates, backgrounds and handwriting fonts thousands of times, so a
random scene split would leak all three into eval and inflate every score. Instead each id of
each kind belongs to exactly one split, and a scene only combines ids from its own split:
eval checks are printed on stock, laid on surfaces, written by hands and made out to payees
at banks that train never saw.

Every id gets its own score in [0, 1) from blake2b(seed, kind salt, id). Two policies use it:

- Open pools (backgrounds, which grow as FLUX batches land): `assign_open_pool_ids`. From
  `OPEN_POOL_MIN_SIZE_FOR_THRESHOLDS` ids up, each id's split depends on its score alone (eval
  below the eval fraction, val below eval + val, train above), so adding ids never moves an
  existing one and a rebuild after more backgrounds land keeps old train surfaces in train.
  Below that size, per-id buckets are too lopsided (12 backgrounds hashed 5 / 3 / 4), so small
  pools are rank-dealt like registries. Trade-off: the pool reshuffles once, when it crosses
  the threshold. A rare minimum fill (val or eval under two ids) can also move one id later.
- Closed registries (templates, fonts, payees, banks; defined in code, so they only change with
  a `GENERATOR_VERSION` bump): `assign_ids_by_hash_rank` deals ids in score order, eval first,
  in exact proportions. Hash buckets on 13-34 ids come out lopsided (one seed put 44% of the
  handwriting fonts in val/eval); rank dealing cannot. Trade-off: adding one id moves at most
  one existing id across each split boundary.
- Templates are stratified by layout family (dealt per family), so every family appears in
  val and eval. Templates that carry eval-reserved printed fonts (`is_holdout_template_index`)
  are dealt before the others, and held-out counts grow to fit them, so they never reach train.
- Held-out sizes are rounded, not floored, so small pools (13 signature fonts) still give val
  and eval two ids each.
"""

import hashlib
import logging
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass

from synth.render.fake_payees_and_banks import BANK_NAMES, PAYEE_NAMES
from synth.render.fonts import FontRole, font_ids_with_role
from synth.render.printed_font_pools import is_holdout_template_index

logger = logging.getLogger(__name__)

SPLIT_NAMES = ("train", "val", "eval")
DEFAULT_SPLIT_FRACTIONS = {"train": 0.7, "val": 0.15, "eval": 0.15}
TEMPLATE_SALT = 11
BACKGROUND_SALT = 23
HANDWRITING_FONT_SALT = 37
SIGNATURE_FONT_SALT = 41
PAYEE_SALT = 43
BANK_SALT = 47
HASH_DIGEST_BYTES = 8
HASH_SCORE_RANGE = 2 ** (8 * HASH_DIGEST_BYTES)
MIN_SCENES_FOR_HELD_OUT_SPLITS = 3
OPEN_POOL_MIN_SIZE_FOR_THRESHOLDS = 50   # train's share then varies about +-9%; below it, rank-deal
HELD_OUT_MINIMUM = 2   # val / eval size an open pool is topped up to (when its rounded share allows)
SPLIT_SCORE_ORDER = {"eval": 0, "val": 1, "train": 2}   # score ranges, low to high


@dataclass(frozen=True)
class SplitPools:
    """The ids one split may draw from; every list is disjoint from the other splits' lists."""

    template_ids: list[str]
    background_ids: list[str]
    handwriting_font_ids: list[str]
    signature_font_ids: list[str]
    payee_names: list[str]
    bank_names: list[str]

    def to_dict(self) -> dict:
        """Plain-JSON form."""
        return asdict(self)


def stable_split_score(item_id: str, seed: int, salt: int) -> float:
    """The id's own position in [0, 1) for this seed and kind; independent of every other id."""
    digest = hashlib.blake2b(f"{seed}|{salt}|{item_id}".encode(), digest_size=HASH_DIGEST_BYTES).digest()
    return int.from_bytes(digest, "big") / HASH_SCORE_RANGE


def held_out_count(total: int, fraction: float) -> int:
    """How many of `total` ids a held-out split (val or eval) gets: rounded, at least one."""
    return max(1, round(total * fraction))


def require_every_split_fillable(ids: list[str]) -> None:
    """Both policies need one id per split."""
    if len(ids) < len(SPLIT_NAMES):
        raise ValueError(f"need at least {len(SPLIT_NAMES)} ids to fill every split, got {len(ids)}")


def fill_split_minimums(assignment: dict[str, str], scores: dict[str, float], pool_size: int,
                        fractions: dict[str, float]) -> dict[str, str]:
    """Top up train to one id, then eval and val to `min(2, held_out_count)`, from splits above their own minimum.

    Train is the preferred donor for eval and val. The mover is the donor id whose score lies nearest
    the receiver's range (scores run eval < val < train): the id that nearly hashed there anyway.
    """
    assignment = dict(assignment)
    minimums = {"train": 1, **{name: min(HELD_OUT_MINIMUM, held_out_count(pool_size, fractions[name])) for name in ("val", "eval")}}
    for receiver in ("train", "eval", "val"):
        while (counts := Counter(assignment.values()))[receiver] < minimums[receiver]:
            donors = [name for name in ("train", "val", "eval") if name != receiver and counts[name] > minimums[name]]
            if not donors:
                raise ValueError(f"cannot give every split its minimum from a pool of {pool_size}")
            donor = donors[0] if receiver != "train" else max(donors, key=lambda name: counts[name])
            donor_ids = [item_id for item_id, split in assignment.items() if split == donor]
            receiver_is_above = SPLIT_SCORE_ORDER[receiver] > SPLIT_SCORE_ORDER[donor]
            mover = (max if receiver_is_above else min)(donor_ids, key=lambda item_id: (scores[item_id], item_id))
            logger.info("split minimum fill: %s moves %s -> %s", mover, donor, receiver)
            assignment[mover] = receiver
    return assignment


def assign_ids_by_hash_threshold(ids: list[str], fractions: dict[str, float], seed: int, salt: int) -> dict[str, list[str]]:
    """Each id's split depends only on its own score, so adding ids moves none (minimum fill aside)."""
    require_every_split_fillable(ids)
    scores = {item_id: stable_split_score(item_id, seed, salt) for item_id in ids}
    eval_limit = fractions["eval"]
    assignment = {item_id: "eval" if score < eval_limit else "val" if score < eval_limit + fractions["val"] else "train"
                  for item_id, score in scores.items()}
    assignment = fill_split_minimums(assignment, scores, len(ids), fractions)
    return {name: sorted(item_id for item_id, split in assignment.items() if split == name) for name in SPLIT_NAMES}


def assign_ids_by_hash_rank(ids: list[str], fractions: dict[str, float], seed: int, salt: int,
                            held_out_first: frozenset[str] = frozenset()) -> dict[str, list[str]]:
    """Closed registries: deal ids in score order (eval, then val, then train) in exact proportions.

    Ids in `held_out_first` are dealt before all others, and val grows if needed to hold them all,
    so none of them reach train.
    """
    require_every_split_fillable(ids)
    ordered = sorted(ids, key=lambda item_id: (item_id not in held_out_first, stable_split_score(item_id, seed, salt), item_id))
    eval_count = held_out_count(len(ids), fractions["eval"])
    val_count = held_out_count(len(ids), fractions["val"])
    if eval_count + val_count >= len(ids):  # tiny pools: leave train at least one id
        eval_count, val_count = 1, 1
    val_count = max(val_count, len(held_out_first & set(ids)) - eval_count)
    return {
        "train": sorted(ordered[eval_count + val_count:]),
        "val": sorted(ordered[eval_count:eval_count + val_count]),
        "eval": sorted(ordered[:eval_count]),
    }


def assign_open_pool_ids(ids: list[str], fractions: dict[str, float], seed: int, salt: int) -> dict[str, list[str]]:
    """Growing pools: per-id thresholds once large enough to stay balanced, rank dealing before that."""
    if len(ids) >= OPEN_POOL_MIN_SIZE_FOR_THRESHOLDS:
        return assign_ids_by_hash_threshold(ids, fractions, seed, salt)
    return assign_ids_by_hash_rank(ids, fractions, seed, salt)


def assign_ids_to_splits_stratified(group_by_id: dict[str, str], fractions: dict[str, float], seed: int, salt: int,
                                    held_out_first: frozenset[str] = frozenset()) -> dict[str, list[str]]:
    """Rank-deal each group (stratum) separately and merge; groups too small to split are pooled together."""
    ids_by_group = defaultdict(list)
    for item_id, group in sorted(group_by_id.items()):
        ids_by_group[group].append(item_id)
    small_group_ids = [item_id for ids in ids_by_group.values() if len(ids) < len(SPLIT_NAMES) for item_id in ids]
    strata = [ids for _, ids in sorted(ids_by_group.items()) if len(ids) >= len(SPLIT_NAMES)]
    if small_group_ids:
        strata.append(small_group_ids)
    merged = {split_name: [] for split_name in SPLIT_NAMES}
    for ids in strata:
        if len(ids) < len(SPLIT_NAMES):  # only when the pooled leftovers are still too few: they train
            merged["train"] += [item_id for item_id in ids if item_id not in held_out_first]
            merged["eval"] += [item_id for item_id in ids if item_id in held_out_first]
            continue
        for split_name, split_ids in assign_ids_by_hash_rank(ids, fractions, seed, salt, held_out_first).items():
            merged[split_name] += split_ids
    return {split_name: sorted(ids) for split_name, ids in merged.items()}


def plan_template_pools(template_family_by_id: dict[str, str], seed: int,
                        fractions: dict[str, float] = DEFAULT_SPLIT_FRACTIONS) -> dict[str, list[str]]:
    """Template ids per split, stratified by layout family; eval-reserved-font templates are never train."""
    font_holdout_ids = frozenset(t for t in template_family_by_id if is_holdout_template_index(int(t.removeprefix("tpl_"))))
    return assign_ids_to_splits_stratified(template_family_by_id, fractions, seed, TEMPLATE_SALT, font_holdout_ids)


def plan_registry_pools(seed: int, fractions: dict[str, float] = DEFAULT_SPLIT_FRACTIONS) -> dict[str, dict[str, list[str]]]:
    """Closed registries per kind: handwriting and signature fonts (font registry), payee and bank names."""
    registries = {
        "handwriting_font_ids": (font_ids_with_role(FontRole.HANDWRITING), HANDWRITING_FONT_SALT),
        "signature_font_ids": (font_ids_with_role(FontRole.SIGNATURE), SIGNATURE_FONT_SALT),
        "payee_names": (list(PAYEE_NAMES), PAYEE_SALT),
        "bank_names": (list(BANK_NAMES), BANK_SALT),
    }
    return {kind: assign_ids_by_hash_rank(ids, fractions, seed, salt) for kind, (ids, salt) in registries.items()}


def plan_split_pools(template_family_by_id: dict[str, str], background_ids: list[str], seed: int,
                     fractions: dict[str, float] = DEFAULT_SPLIT_FRACTIONS) -> dict[str, SplitPools]:
    """Partition every id kind at once (templates, backgrounds, then the closed registries)."""
    by_kind = {
        "template_ids": plan_template_pools(template_family_by_id, seed, fractions),
        "background_ids": assign_open_pool_ids(background_ids, fractions, seed, BACKGROUND_SALT),
        **plan_registry_pools(seed, fractions),
    }
    return {split_name: SplitPools(**{kind: assignment[split_name] for kind, assignment in by_kind.items()})
            for split_name in SPLIT_NAMES}


def scene_counts_per_split(total_scenes: int, fractions: dict[str, float]) -> dict[str, int]:
    """Scene count per split: val and eval floor their fraction but get at least one scene from 3 scenes up."""
    held_out_minimum = 1 if total_scenes >= MIN_SCENES_FOR_HELD_OUT_SPLITS else 0
    counts = {name: max(held_out_minimum, int(total_scenes * fractions[name])) for name in ("val", "eval")}
    counts["train"] = total_scenes - counts["val"] - counts["eval"]
    return {name: counts[name] for name in SPLIT_NAMES}
