"""A writer's hand habits: how glyphs are sized, spaced, slanted and wobbled on a line.

Leaf module (no engine imports) so layout, warp and the Writer can all share it. Lengths
are fractions of the x-height, so the same habits read the same at any writing size.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class HandHabits:
    """Per-writer geometry habits, fixed for every field one person fills in."""

    slant_degrees: float                 # forward (+) or backward (-) lean, applied as a shear
    letter_spacing_scale: float          # multiplies each glyph's advance
    word_spacing_scale: float            # multiplies the space advance
    glyph_size_jitter: float             # std of per-glyph size, relative
    glyph_width_jitter: float            # std of per-glyph horizontal squash/stretch, relative
    glyph_rotation_jitter_degrees: float # std of per-glyph rotation
    glyph_baseline_jitter: float         # std of per-glyph vertical offset, x-heights
    baseline_wander: float               # amplitude of the slow baseline wave, x-heights
    baseline_drift_slope: float          # uphill (-) / downhill (+) drift, px per px
    elastic_warp: float                  # amplitude of the glyph-scale shape warp, x-heights
