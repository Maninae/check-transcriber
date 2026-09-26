"""Layouts for the close and single framing regimes (framing_regimes.py), built from placement.py's parts.

- close: 2-6 checks laid out tightly for a close photo, rarely a lone sideways check. Gapped grids
  use small gaps and often zero gaps (edges touching); loose layouts overlap slightly (each check keeps >= 80% visible,
  against 70% in the wide regime), or fan when there are at most 4 checks.
- single: one check at any rotation. Mostly upright or flipped, sometimes quarter-turned, and a
  fifth of the time at an arbitrary angle, as someone photographing one check by hand holds it.

Returns the same `(layout mode, placements)` shape as `placement.plan_layout`; single scenes use
the mode name `single`.
"""

import numpy as np

from scene_composer.geometry.placement import (
    FAN_MAX_CHECKS,
    CheckPlacement,
    choose_column_count,
    clipped_normal,
    pack_rows,
    plan_fan_layout,
    rotated_extent,
    sample_rotations,
    spread_until_visible,
)

CLOSE_LOOSE_PROBABILITY = 0.45
CLOSE_FAN_SHARE = 0.25
CLOSE_TOUCHING_PROBABILITY = 0.35
CLOSE_GAP_RANGE_INCHES = (0.02, 0.45)
CLOSE_MIN_GAP_INCHES = 0.0
CLOSE_TOUCHING_GAP_RANGE_INCHES = (-0.03, 0.04)   # edges touch; a hair negative lets corners just overlap
CLOSE_OVERLAP_GAP_RANGE_INCHES = (-0.45, 0.05)
CLOSE_MIN_VISIBLE_FRACTION = 0.8
CLOSE_GRID_JITTER_DEGREES = (1.5, 4.0)             # (sigma, max)
CLOSE_LOOSE_JITTER_DEGREES = (3.0, 8.0)
CLOSE_QUARTER_TURN_PROBABILITY = 0.02   # per check; a lone sideways check breaks the tight rows (v1 wide: 0.07)
SINGLE_UPRIGHT_JITTER_DEGREES = (4.0, 10.0)
SINGLE_ROTATION_SHARES = {
    "upright": 0.5,
    "flipped": 0.15,
    "quarter_turn": 0.12,
    "any_angle": 0.23,
}


def plan_close_grid_layout(sizes_inches: list[tuple[float, float]], rng: np.random.Generator) -> list[CheckPlacement]:
    """A tight grid: small gaps, or edges touching."""
    rotations = sample_rotations(len(sizes_inches), rng, *CLOSE_GRID_JITTER_DEGREES, CLOSE_QUARTER_TURN_PROBABILITY)
    extents = [rotated_extent(w, h, angle) for (w, h), angle in zip(sizes_inches, rotations)]
    touching = rng.random() < CLOSE_TOUCHING_PROBABILITY
    gap_range = CLOSE_TOUCHING_GAP_RANGE_INCHES if touching else CLOSE_GAP_RANGE_INCHES
    min_gap = CLOSE_TOUCHING_GAP_RANGE_INCHES[0] if touching else CLOSE_MIN_GAP_INCHES
    columns = choose_column_count(extents, float(np.mean(gap_range)), rng)
    return pack_rows(extents, rotations, columns, gap_range, min_gap, rng)


def plan_close_overlap_layout(sizes_inches: list[tuple[float, float]], rng: np.random.Generator) -> list[CheckPlacement]:
    """Tight packing with slight overlaps both ways; spread until each check keeps 80% visible."""
    rotations = sample_rotations(len(sizes_inches), rng, *CLOSE_LOOSE_JITTER_DEGREES, CLOSE_QUARTER_TURN_PROBABILITY)
    extents = [rotated_extent(w, h, angle) for (w, h), angle in zip(sizes_inches, rotations)]
    columns = choose_column_count(extents, 0.0, rng)
    order = rng.permutation(len(sizes_inches))  # paste order differs from reading order, so overlaps go both ways
    placements = pack_rows(extents, rotations, columns, CLOSE_OVERLAP_GAP_RANGE_INCHES, CLOSE_OVERLAP_GAP_RANGE_INCHES[0], rng)
    return spread_until_visible(sizes_inches, [placements[i] for i in order], [int(i) for i in order], CLOSE_MIN_VISIBLE_FRACTION)


def plan_close_layout(sizes_inches: list[tuple[float, float]], rng: np.random.Generator) -> tuple[str, list[CheckPlacement]]:
    """Pick a close-regime layout mode and plan it."""
    if len(sizes_inches) > 1 and rng.random() < CLOSE_LOOSE_PROBABILITY:
        if rng.random() < CLOSE_FAN_SHARE and len(sizes_inches) <= FAN_MAX_CHECKS:
            return "loose_fan", plan_fan_layout(sizes_inches, rng)
        return "loose_overlap", plan_close_overlap_layout(sizes_inches, rng)
    return "grid", plan_close_grid_layout(sizes_inches, rng)


def sample_single_check_rotation(rng: np.random.Generator) -> float:
    """Rotation (ccw degrees) of a single hand-photographed check."""
    kind = str(rng.choice(list(SINGLE_ROTATION_SHARES), p=list(SINGLE_ROTATION_SHARES.values())))
    if kind == "any_angle":
        return float(rng.uniform(0.0, 360.0))
    if kind == "quarter_turn":
        base = 90.0 * rng.choice([1, 3])
    else:
        base = 180.0 if kind == "flipped" else 0.0
    return float(base + clipped_normal(rng, *SINGLE_UPRIGHT_JITTER_DEGREES))


def plan_single_layout(sizes_inches: list[tuple[float, float]], rng: np.random.Generator) -> tuple[str, list[CheckPlacement]]:
    """One check at the sheet origin, rotated; more than one check is a caller bug."""
    if len(sizes_inches) != 1:
        raise ValueError(f"the single framing regime lays out exactly one check, got {len(sizes_inches)}")
    return "single", [CheckPlacement(0.0, 0.0, sample_single_check_rotation(rng))]
