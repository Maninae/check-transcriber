"""Paper coverage (contract C2): the rendered check's alpha, 255 on paper and 0 where there is none.

The full rectangle stays the label frame (`CheckLabel.width_px/height_px`); alpha only removes
a few pixels at the edge:
- every check: slightly rounded, unevenly worn corners;
- checks torn from a checkbook or money-order pad: one short side is a torn perforation, a row of
  shallow scallops (half holes) separated by the small paper nubs that were the perf bridges;
- business and laser stock: clean guillotine cuts.
"""

import numpy as np

CORNER_RADIUS_INCHES = (0.008, 0.035)
PERFORATION_PITCH_INCHES = (0.035, 0.055)
PERFORATION_DEPTH_INCHES = (0.007, 0.014)
PERFORATION_HOLE_FRACTION = (0.5, 0.7)


def round_corner(alpha: np.ndarray, corner: str, radius_px: float) -> None:
    """Cut an anti-aliased quarter-circle corner of `radius_px` into `alpha` in place ('tl','tr','br','bl')."""
    size = int(np.ceil(radius_px)) + 1
    height, width = alpha.shape
    local_y, local_x = np.mgrid[0:size, 0:size].astype(np.float32) + 0.5
    distance_from_center = np.hypot(radius_px - local_x, radius_px - local_y)
    inside = np.clip(radius_px - distance_from_center + 0.5, 0, 1)
    outside_arc = (local_x < radius_px) & (local_y < radius_px)
    corner_patch = np.where(outside_arc, inside, 1.0)
    if "r" in corner:
        corner_patch = corner_patch[:, ::-1]
    if "b" in corner:
        corner_patch = corner_patch[::-1, :]
    rows = slice(0, size) if "t" in corner else slice(height - size, height)
    columns = slice(0, size) if "l" in corner else slice(width - size, width)
    alpha[rows, columns] = np.minimum(alpha[rows, columns], corner_patch)


def perforation_profile(height: int, dpi: int, rng: np.random.Generator) -> np.ndarray:
    """Inset (pixels) of the torn edge at every row: scallops at the holes, ragged nubs between them."""
    pitch = rng.uniform(*PERFORATION_PITCH_INCHES) * dpi
    depth = rng.uniform(*PERFORATION_DEPTH_INCHES) * dpi
    hole_fraction = rng.uniform(*PERFORATION_HOLE_FRACTION)
    rows = np.arange(height, dtype=np.float32) + rng.uniform(0, pitch)
    position_in_pitch = (rows % pitch) / pitch
    inside_hole = position_in_pitch < hole_fraction
    hole_position = np.clip(position_in_pitch / hole_fraction, 0, 1) * 2 - 1  # -1..1 across the hole
    scallop = np.sqrt(np.clip(1 - hole_position ** 2, 0, 1)) * depth
    nub_tear = rng.uniform(0, 0.35 * depth, height)  # the bridge fibres tear unevenly, never cleanly flush
    return np.where(inside_hole, np.maximum(scallop, nub_tear), nub_tear).astype(np.float32)


def apply_perforation(alpha: np.ndarray, side: str, dpi: int, rng: np.random.Generator) -> None:
    """Tear the 'left' or 'right' short edge along a perforation, in place."""
    height, width = alpha.shape
    inset = perforation_profile(height, dpi, rng)
    strip_width = int(np.ceil(inset.max())) + 2
    column_centers = np.arange(strip_width, dtype=np.float32)[None, :] + 0.5
    strip = np.clip(column_centers - inset[:, None], 0, 1)
    if side == "left":
        alpha[:, :strip_width] = np.minimum(alpha[:, :strip_width], strip)
    else:
        alpha[:, width - strip_width:] = np.minimum(alpha[:, width - strip_width:], strip[:, ::-1])


def make_paper_alpha(width: int, height: int, dpi: int, perforated_side: str | None, rng: np.random.Generator) -> np.ndarray:
    """uint8 (height, width) paper coverage for one physical check."""
    alpha = np.ones((height, width), np.float32)
    if perforated_side is not None:
        apply_perforation(alpha, perforated_side, dpi, rng)
    for corner in ("tl", "tr", "br", "bl"):
        round_corner(alpha, corner, rng.uniform(*CORNER_RADIUS_INCHES) * dpi)
    return np.round(alpha * 255).astype(np.uint8)
