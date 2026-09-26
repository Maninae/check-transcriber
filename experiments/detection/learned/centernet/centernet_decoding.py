"""Turn the network's heatmap and corner-offset maps into ordered check quads (numpy only).

Deliberately dependency-light so it ports line-for-line to JavaScript for the web app:

1. A peak is a cell whose heatmap probability is the maximum of its 3x3 neighborhood
   and at least `score_threshold` (this replaces NMS: one check, one peak).
2. Keep the `max_detections` highest peaks.
3. Corners (input pixels) = (cell + 0.5 + offset) * OUTPUT_STRIDE, in TL,TR,BR,BL order.
4. Map input pixels back to the source image with the inverse letterbox affine.

The heatmap passed in must already be sigmoid probabilities (the exported ONNX graph
applies the sigmoid).
"""

import numpy as np

from experiments.detection.learned.centernet.centernet_config import CORNER_COUNT, OUTPUT_STRIDE
from experiments.detection.learned.centernet.letterbox_geometry import apply_affine_to_points, invert_affine_matrix

DEFAULT_SCORE_THRESHOLD = 0.3
DEFAULT_MAX_DETECTIONS = 32


def find_heatmap_peaks(heatmap: np.ndarray, score_threshold: float, max_detections: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """3x3 local maxima above threshold, best first; returns (rows, columns, scores)."""
    grid_height, grid_width = heatmap.shape
    padded = np.pad(heatmap, 1, mode="constant", constant_values=-np.inf)
    neighborhood_max = np.max(
        [padded[row_shift : row_shift + grid_height, column_shift : column_shift + grid_width]
         for row_shift in range(3) for column_shift in range(3)],
        axis=0,
    )
    peak_mask = (heatmap >= neighborhood_max) & (heatmap >= score_threshold)
    rows, columns = np.nonzero(peak_mask)
    scores = heatmap[rows, columns]
    best_first = np.argsort(-scores, kind="stable")[:max_detections]
    return rows[best_first], columns[best_first], scores[best_first]


def decode_checks_in_input_pixels(
    heatmap: np.ndarray,
    corner_offsets: np.ndarray,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    max_detections: int = DEFAULT_MAX_DETECTIONS,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode one image's maps.

    Args:
        heatmap: (G, G) probabilities.
        corner_offsets: (8, G, G) offsets in output cells, TL,TR,BR,BL x/y.
    Returns:
        (corners (N, 4, 2) in input pixels, scores (N,)).
    """
    rows, columns, scores = find_heatmap_peaks(heatmap, score_threshold, max_detections)
    peak_offsets = corner_offsets[:, rows, columns].T.reshape(-1, CORNER_COUNT, 2)
    cell_centers = np.stack([columns + 0.5, rows + 0.5], axis=1)[:, None, :]
    corners = (cell_centers + peak_offsets) * OUTPUT_STRIDE
    return corners.astype(np.float64), scores.astype(np.float64)


def map_input_corners_to_source(corners_in_input: np.ndarray, source_to_input_affine: np.ndarray) -> np.ndarray:
    """Undo the letterbox (or any 2x3 affine) for (N, 4, 2) corners."""
    return apply_affine_to_points(invert_affine_matrix(source_to_input_affine), corners_in_input)
