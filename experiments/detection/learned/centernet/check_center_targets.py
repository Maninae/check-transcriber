"""Encode ground-truth check corners (input-pixel coords) into CenterNet training targets.

Per check we pick an ANCHOR point, put a heatmap peak on the output cell holding it,
and regress the four ordered corners as offsets from that cell's center:

- Anchor: the corner-quad center when it lies on the canvas, otherwise the centroid
  of the on-canvas part (partly out-of-frame checks). Checks with almost nothing on
  the canvas are skipped. The decoder does not need this rule: it only ever reads
  "peak cell + offsets".
- Heatmap: a Gaussian rotated with the check (sigma along each axis a fixed fraction
  of that side's length), peak exactly 1.0 at the anchor cell, max-merged across checks.
- Offsets: in OUTPUT CELLS, `corner / stride - (cell + 0.5)`, TL,TR,BR,BL x/y. They are
  supervised on every cell where this check's Gaussian is >= `OFFSET_REGION_MIN_GAUSSIAN`
  and larger than any other check's (TTFNet-style dense supervision), weighted by the
  Gaussian value; the anchor cell always gets weight 1. Corners off the canvas are
  still regressed.
- `corner_offset_normalizer` holds each supervised cell's check diagonal (cells), so
  the loss is scale-invariant across small and large checks.
"""

import numpy as np
from shapely.geometry import Polygon, box

from experiments.detection.learned.centernet.centernet_config import CORNER_OFFSET_CHANNELS, OUTPUT_STRIDE

GAUSSIAN_SIGMA_FRACTION_OF_SIDE = 0.1
MIN_GAUSSIAN_SIGMA_CELLS = 0.8
OFFSET_REGION_MIN_GAUSSIAN = 0.5
MIN_ON_CANVAS_AREA_FRACTION = 0.15  # of the full quad area; smaller slivers are not targets


def choose_anchor_point(corners: np.ndarray, canvas_size_pixels: int) -> np.ndarray | None:
    """Anchor in input pixels (see module docstring), or None when the check is barely on canvas."""
    quad_polygon = Polygon(corners).buffer(0)
    if quad_polygon.area <= 0:
        return None
    on_canvas_polygon = quad_polygon.intersection(box(0, 0, canvas_size_pixels, canvas_size_pixels))
    if on_canvas_polygon.area < MIN_ON_CANVAS_AREA_FRACTION * quad_polygon.area:
        return None
    quad_center = corners.mean(axis=0)
    if np.all(quad_center >= 0) and np.all(quad_center < canvas_size_pixels):
        return quad_center
    centroid = on_canvas_polygon.centroid
    return np.clip(np.array([centroid.x, centroid.y]), 0, canvas_size_pixels - 1e-3)


def rotated_gaussian_on_grid(
    corners_in_cells: np.ndarray, anchor_cell: np.ndarray, grid_size: int
) -> np.ndarray:
    """Gaussian over the grid, centered on the anchor cell center, axes aligned with the check."""
    top_edge = corners_in_cells[1] - corners_in_cells[0]
    bottom_edge = corners_in_cells[2] - corners_in_cells[3]
    width_axis = top_edge + bottom_edge
    width_axis /= max(np.linalg.norm(width_axis), 1e-6)
    height_axis = np.array([-width_axis[1], width_axis[0]])
    check_width = (np.linalg.norm(top_edge) + np.linalg.norm(bottom_edge)) / 2
    check_height = (
        np.linalg.norm(corners_in_cells[3] - corners_in_cells[0])
        + np.linalg.norm(corners_in_cells[2] - corners_in_cells[1])
    ) / 2
    sigma_width = max(GAUSSIAN_SIGMA_FRACTION_OF_SIDE * check_width, MIN_GAUSSIAN_SIGMA_CELLS)
    sigma_height = max(GAUSSIAN_SIGMA_FRACTION_OF_SIDE * check_height, MIN_GAUSSIAN_SIGMA_CELLS)
    rows, columns = np.mgrid[0:grid_size, 0:grid_size]
    delta_x = columns - anchor_cell[0]
    delta_y = rows - anchor_cell[1]
    along_width = delta_x * width_axis[0] + delta_y * width_axis[1]
    along_height = delta_x * height_axis[0] + delta_y * height_axis[1]
    return np.exp(-(along_width**2 / (2 * sigma_width**2) + along_height**2 / (2 * sigma_height**2)))


def encode_check_targets(corner_sets: list[np.ndarray], input_size_pixels: int) -> dict[str, np.ndarray]:
    """Build the target maps for one input image.

    Args:
        corner_sets: per check, (4, 2) TL/TR/BR/BL corners in input-canvas pixels.
        input_size_pixels: square canvas side; must be a multiple of `OUTPUT_STRIDE`.
    Returns:
        data_dict with `center_heatmap_target` (1, G, G), `corner_offset_target` (8, G, G),
        `corner_offset_weight` (1, G, G), `corner_offset_normalizer` (1, G, G) and
        `anchor_cells` (N, 2) int (column, row) of every encoded check.
    """
    grid_size = input_size_pixels // OUTPUT_STRIDE
    heatmap = np.zeros((grid_size, grid_size), dtype=np.float32)
    owner_gaussian = np.zeros((grid_size, grid_size), dtype=np.float32)  # strongest Gaussian so far per cell
    offsets = np.zeros((CORNER_OFFSET_CHANNELS, grid_size, grid_size), dtype=np.float32)
    weights = np.zeros((grid_size, grid_size), dtype=np.float32)
    normalizers = np.ones((grid_size, grid_size), dtype=np.float32)
    anchor_cells = []
    rows, columns = np.mgrid[0:grid_size, 0:grid_size]
    cell_centers = np.stack([columns + 0.5, rows + 0.5], axis=0)  # (2, G, G) in cells
    for corners in corner_sets:
        anchor_point = choose_anchor_point(np.asarray(corners, dtype=np.float64), input_size_pixels)
        if anchor_point is None:
            continue
        corners_in_cells = np.asarray(corners, dtype=np.float64) / OUTPUT_STRIDE
        anchor_cell = np.clip(np.floor(anchor_point / OUTPUT_STRIDE).astype(int), 0, grid_size - 1)
        gaussian = rotated_gaussian_on_grid(corners_in_cells, anchor_cell, grid_size).astype(np.float32)
        heatmap = np.maximum(heatmap, gaussian)
        owned_cells = (gaussian >= OFFSET_REGION_MIN_GAUSSIAN) & (gaussian > owner_gaussian)
        owned_cells[anchor_cell[1], anchor_cell[0]] = True
        owner_gaussian = np.where(owned_cells, gaussian, owner_gaussian)
        corner_offsets_per_cell = corners_in_cells.reshape(CORNER_OFFSET_CHANNELS, 1, 1) - np.tile(
            cell_centers, (4, 1, 1)
        )
        offsets[:, owned_cells] = corner_offsets_per_cell[:, owned_cells]
        weights[owned_cells] = gaussian[owned_cells]
        diagonal_cells = np.linalg.norm(corners_in_cells[2] - corners_in_cells[0])
        normalizers[owned_cells] = max(diagonal_cells, 1.0)
        anchor_cells.append(anchor_cell)
    return {
        "center_heatmap_target": heatmap[None],
        "corner_offset_target": offsets,
        "corner_offset_weight": weights[None],
        "corner_offset_normalizer": normalizers[None],
        "anchor_cells": np.asarray(anchor_cells, dtype=np.int64).reshape(-1, 2),
    }
