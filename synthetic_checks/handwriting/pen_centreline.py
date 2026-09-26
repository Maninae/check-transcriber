"""Reduce rendered glyph shapes to the one-pixel centreline a pen tip would have travelled.

Zhang-Suen thinning (Zhang & Suen 1984, CACM 27(3)), vectorized: each pass packs the 8
neighbours of every pixel into one byte and looks up "may this pixel be deleted?" in a
256-entry table, so a pass is a handful of numpy slices instead of a Python loop.

- Neighbour order follows the paper: P2 = north, then clockwise to P9 = north-west.
- Input is a boolean mask; output is a boolean 8-connected centreline of the same shape.
- The number of passes is about half the stroke width, so thin fonts thin fast.
- Thick strokes leave short side spurs at corners; `prune_centreline_spurs` trims them.
"""

import cv2
import numpy as np

# (row offset, column offset) of P2..P9, clockwise from north.
NEIGHBOUR_OFFSETS = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))
MIN_NEIGHBOURS_TO_DELETE = 2
MAX_NEIGHBOURS_TO_DELETE = 6
MAX_THINNING_PASSES = 64
NEIGHBOURHOOD_KERNEL = np.ones((3, 3), np.uint8)


def build_deletion_tables() -> tuple[np.ndarray, np.ndarray]:
    """Lookup tables (one per Zhang-Suen sub-iteration): neighbour code -> deletable."""
    first_pass_table = np.zeros(256, bool)
    second_pass_table = np.zeros(256, bool)
    for code in range(256):
        p2, p3, p4, p5, p6, p7, p8, p9 = ((code >> bit) & 1 for bit in range(8))
        ring = (p2, p3, p4, p5, p6, p7, p8, p9, p2)
        neighbour_count = sum(ring[:8])
        zero_to_one_transitions = sum(1 for a, b in zip(ring, ring[1:]) if a == 0 and b == 1)
        shared = (MIN_NEIGHBOURS_TO_DELETE <= neighbour_count <= MAX_NEIGHBOURS_TO_DELETE
                  and zero_to_one_transitions == 1)
        first_pass_table[code] = shared and p2 * p4 * p6 == 0 and p4 * p6 * p8 == 0
        second_pass_table[code] = shared and p2 * p4 * p8 == 0 and p2 * p6 * p8 == 0
    return first_pass_table, second_pass_table


FIRST_PASS_TABLE, SECOND_PASS_TABLE = build_deletion_tables()


def neighbour_codes(mask: np.ndarray) -> np.ndarray:
    """Pack each pixel's 8 neighbours (P2 = bit 0 ... P9 = bit 7) into a uint8 code."""
    padded = np.pad(mask.astype(np.uint8), 1)
    height, width = mask.shape
    codes = np.zeros((height, width), np.uint8)
    for bit, (row_offset, column_offset) in enumerate(NEIGHBOUR_OFFSETS):
        codes |= padded[1 + row_offset:1 + row_offset + height, 1 + column_offset:1 + column_offset + width] << bit
    return codes


def thin_to_centreline(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning of a boolean mask to a one-pixel-wide boolean centreline."""
    skeleton = mask.astype(bool).copy()
    for _ in range(MAX_THINNING_PASSES):
        changed = False
        for deletion_table in (FIRST_PASS_TABLE, SECOND_PASS_TABLE):
            deletable = skeleton & deletion_table[neighbour_codes(skeleton)]
            if deletable.any():
                skeleton &= ~deletable
                changed = True
        if not changed:
            break
    return skeleton


def centreline_endpoints(skeleton: np.ndarray) -> np.ndarray:
    """Boolean map of stroke ends: centreline pixels with exactly one centreline neighbour."""
    codes = neighbour_codes(skeleton)
    neighbour_count = np.unpackbits(codes[..., None], axis=-1).sum(axis=-1)
    return skeleton & (neighbour_count == 1)


def centreline_junctions(skeleton: np.ndarray) -> np.ndarray:
    """Boolean map of stroke crossings and sharp joins: centreline pixels with 3+ neighbours."""
    codes = neighbour_codes(skeleton)
    neighbour_count = np.unpackbits(codes[..., None], axis=-1).sum(axis=-1)
    return skeleton & (neighbour_count >= 3)


def prune_centreline_spurs(skeleton: np.ndarray, spur_length_px: int) -> np.ndarray:
    """Remove side branches shorter than `spur_length_px` that thinning leaves on thick strokes.

    - Peel endpoints `spur_length_px` times (spurs vanish, real strokes shorten), then regrow
      the surviving stroke ends along the original centreline by the same amount.
    - Components that disappear entirely (i-dots, periods, commas) are restored whole.
    - A spur's base pixel touches the main stroke, is never an endpoint, and survives as a
      one-pixel nub; it lies inside the pen radius of the stroke, so it inks nothing extra.
    """
    pruned = skeleton.copy()
    for _ in range(spur_length_px):
        pruned &= ~centreline_endpoints(pruned)
    frontier = centreline_endpoints(pruned)
    grown = pruned.copy()
    for _ in range(spur_length_px):
        frontier = cv2.dilate(frontier.astype(np.uint8), NEIGHBOURHOOD_KERNEL).astype(bool) & skeleton & ~grown
        if not frontier.any():
            break
        grown |= frontier
    component_count, component_labels = cv2.connectedComponents(skeleton.astype(np.uint8), connectivity=8)
    surviving = np.zeros(component_count, bool)
    surviving[np.unique(component_labels[grown])] = True
    return grown | (skeleton & ~surviving[component_labels])
