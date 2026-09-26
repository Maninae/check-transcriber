"""Which framing regime each scene of a split uses, given a mix like {"wide": 0.4, "close": 0.45, "single": 0.15}.

Assignment is by blocks of `REGIME_BLOCK_SIZE` consecutive scene indices: each block holds the
mix's exact counts (largest-remainder rounding to 1/20), shuffled by `(seed, split, block)`. So:
- a scene's regime depends only on (seed, split, scene_index, mix): an endless stream and a fixed
  build agree without knowing the total;
- any multiple of 20 scenes hits the mix exactly (600 scenes at close 0.75 / single 0.25 = 450 / 150);
- the shuffle uses its own rng, never the scene's, so a wide scene composes exactly as it did before regimes.
"""

import numpy as np

from scene_composer.geometry.framing_regimes import FramingRegime, parse_framing_regime
from synthetic_checks.splits import SPLIT_NAMES

REGIME_BLOCK_SIZE = 20
REGIME_SHUFFLE_STREAM = 0x5C0FE   # keeps the shuffle rng's seed sequence apart from every scene rng's
DEFAULT_FRAMING_REGIME_MIX = {
    FramingRegime.WIDE.value: 0.40,
    FramingRegime.CLOSE.value: 0.45,
    FramingRegime.SINGLE.value: 0.15,
}
MIX_SUM_TOLERANCE = 1e-6


def normalize_framing_regime_mix(mix: dict[str, float]) -> dict[str, float]:
    """Validated mix in FramingRegime order, zero shares dropped; fails loud on unknown names or a bad sum."""
    shares = {parse_framing_regime(name).value: float(share) for name, share in mix.items()}
    if any(share < 0 for share in shares.values()) or abs(sum(shares.values()) - 1.0) > MIX_SUM_TOLERANCE:
        raise ValueError(f"framing regime mix must be non-negative shares summing to 1, got {mix}")
    return {regime.value: shares[regime.value] for regime in FramingRegime if shares.get(regime.value, 0.0) > 0}


def parse_framing_regime_mix(text: str) -> dict[str, float]:
    """'close=0.75,single=0.25' -> {'close': 0.75, 'single': 0.25}."""
    pairs = [item.split("=", 1) for item in text.split(",") if item.strip()]
    if not pairs or any(len(pair) != 2 for pair in pairs):
        raise ValueError(f"expected name=share pairs like 'close=0.75,single=0.25', got {text!r}")
    return normalize_framing_regime_mix({name.strip(): float(share) for name, share in pairs})


def regime_counts_per_block(mix: dict[str, float]) -> dict[str, int]:
    """How many of each regime one block holds: floor of share x block size, remainders by largest fraction."""
    exact = {name: share * REGIME_BLOCK_SIZE for name, share in mix.items()}
    counts = {name: int(np.floor(value)) for name, value in exact.items()}
    by_remainder = sorted(exact, key=lambda name: (-(exact[name] - counts[name]), list(mix).index(name)))
    for name in by_remainder[:REGIME_BLOCK_SIZE - sum(counts.values())]:
        counts[name] += 1
    return counts


def framing_regime_for_scene(seed: int, split: str, scene_index: int, mix: dict[str, float]) -> str:
    """The regime scene `scene_index` of `split` gets under `mix` (see the module docstring)."""
    mix = normalize_framing_regime_mix(mix)
    block_index, position = divmod(scene_index, REGIME_BLOCK_SIZE)
    block = [name for name, count in regime_counts_per_block(mix).items() for _ in range(count)]
    shuffle_rng = np.random.default_rng([seed, SPLIT_NAMES.index(split), block_index, REGIME_SHUFFLE_STREAM])
    return block[int(shuffle_rng.permutation(REGIME_BLOCK_SIZE)[position])]
