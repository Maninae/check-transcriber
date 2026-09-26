"""Track a side's edge sample by sample with dynamic programming (Viterbi), for curved sides.

A curl or a lifted corner bends the paper edge only near one end, which a single
polynomial per side cannot follow. Here the edge offset may change by at most
`maximum_step_offsets` between neighbouring samples (a smoothness prior) and we pick the
path through the (sample x offset) score grid with the largest total clipped score, minus
a small cost per offset step. Path offsets are then refined to sub-pixel with a parabola.

Corners come from local straight fits to the stretch of each side nearest the corner
(`fit_corner_local_curves` in the coordinator), so a bent end decides its own corner.
Plain loops over samples: portable to JS.
"""

import numpy as np

from experiments.detection.refinement.edge_profile_sampling import SideScoreProfiles


def track_edge_path(
    profiles: SideScoreProfiles, score_clip: float, maximum_step_offsets: int, step_cost: float, centre_cost_per_pixel: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Best smooth path through the score grid.

    `centre_cost_per_pixel` pulls the path toward offset 0 (the previous pass's side),
    so it does not wander onto a printed rule parallel to the edge.

    Returns (positions px, sub-pixel offsets px, clipped score at the path) for samples
    whose path cell is a finite score.
    """
    scores = profiles.scores
    valid = np.isfinite(scores)
    clipped = np.where(valid, np.clip(scores, -score_clip, score_clip), 0.0)
    path_scores = clipped - centre_cost_per_pixel * np.abs(profiles.normal_offsets)[None, :]
    number_of_samples, number_of_offsets = clipped.shape
    # Viterbi: best predecessor = max over offset steps in [-k, k] of padded, shifted slices.
    steps = np.arange(-maximum_step_offsets, maximum_step_offsets + 1)
    step_penalties = step_cost * np.abs(steps)[:, None]
    padding = maximum_step_offsets
    accumulated = path_scores[0].copy()
    back_pointers = np.zeros((number_of_samples, number_of_offsets), dtype=np.int64)
    padded = np.full(number_of_offsets + 2 * padding, -np.inf)
    offset_indices = np.arange(number_of_offsets)
    for sample_index in range(1, number_of_samples):
        padded[padding : padding + number_of_offsets] = accumulated
        candidates = np.stack([padded[padding + step : padding + step + number_of_offsets] for step in steps]) - step_penalties
        best_step_index = np.argmax(candidates, axis=0)
        accumulated = path_scores[sample_index] + candidates[best_step_index, offset_indices]
        back_pointers[sample_index] = offset_indices + steps[best_step_index]
    path = np.empty(number_of_samples, dtype=np.int64)
    path[-1] = int(np.argmax(accumulated))
    for sample_index in range(number_of_samples - 1, 0, -1):
        path[sample_index - 1] = back_pointers[sample_index, path[sample_index]]

    rows = np.arange(number_of_samples)
    centre_scores = scores[rows, path]
    left = scores[rows, np.clip(path - 1, 0, number_of_offsets - 1)]
    right = scores[rows, np.clip(path + 1, 0, number_of_offsets - 1)]
    with np.errstate(invalid="ignore"):
        denominator = left - 2 * centre_scores + right
    usable_parabola = np.isfinite(left) & np.isfinite(right) & (denominator < 0) & (path > 0) & (path < number_of_offsets - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        shift = np.where(usable_parabola, 0.5 * (left - right) / denominator, 0.0)
    shift = np.clip(np.nan_to_num(shift), -0.5, 0.5)
    keep = np.isfinite(centre_scores)
    return profiles.positions_pixels[keep], (profiles.normal_offsets[path] + shift)[keep], clipped[rows, path][keep]
