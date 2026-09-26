"""Find a side's paper edge from its score grid: line-level search, then sub-pixel points.

Per-sample peaks are unreliable: a faint paper edge on a same-coloured surface loses to
printed text at most samples. But the paper edge runs the WHOLE side, while text rows,
rules and background texture cover only part of it or scatter. So we integrate the
(clipped) scores along candidate lines, a small Radon transform over angles near the
approximate side, and choose the OUTERMOST line whose integral reaches a fraction of the
best: printed borders and rules always lie inside the paper edge.

Then `extract_edge_points_near_centre` reads each sample's own peak within a few pixels
of that line, with a parabola fit for sub-pixel offset, for the robust curve fit.
"""

import numpy as np

from experiments.detection.refinement.edge_profile_sampling import SideScoreProfiles


def search_outermost_strong_line(
    profiles: SideScoreProfiles,
    maximum_angle_degrees: float,
    angle_step_degrees: float,
    outermost_line_ratio: float,
    minimum_mean_score: float,
    absolute_score_clip: float,
) -> tuple[float, float] | None:
    """Return (offset at the side's midpoint, slope d offset / d position) or None.

    Scores are clipped to [-c, +c], c = min(absolute_score_clip, median of per-sample
    maxima): any edge with at least that contrast saturates, so strong ink cannot outweigh
    a moderate paper edge present at every sample, and "outermost" decides among them.
    Keeping the NEGATIVE half matters: over background texture the score is zero-mean
    noise and integrates to ~0, where a positive-only clip would turn it into a fake line.
    """
    valid = np.isfinite(profiles.scores)
    valid_rows = valid.any(axis=1)
    if valid_rows.sum() < 2:
        return None
    positive_scores = np.where(valid, np.clip(profiles.scores, 0, None), 0.0)
    clip_value = min(absolute_score_clip, float(np.median(positive_scores[valid_rows].max(axis=1))))
    if clip_value <= 0:
        return None
    clipped_scores = np.where(valid, np.clip(profiles.scores, -clip_value, clip_value), 0.0)

    relative_positions = profiles.positions_pixels - profiles.side_length / 2
    angles = np.deg2rad(np.arange(-maximum_angle_degrees, maximum_angle_degrees + 1e-9, angle_step_degrees))
    slopes = np.tan(angles)
    number_of_samples, number_of_offsets = clipped_scores.shape
    offset_indices = np.arange(number_of_offsets)
    sample_rows = np.arange(number_of_samples)[:, None]
    line_scores = np.empty((len(slopes), number_of_offsets))
    for slope_index, slope in enumerate(slopes):
        shifted_indices = offset_indices[None, :] + np.round(slope * relative_positions).astype(int)[:, None]
        in_range = (shifted_indices >= 0) & (shifted_indices < number_of_offsets)
        gathered = clipped_scores[sample_rows, np.clip(shifted_indices, 0, number_of_offsets - 1)]
        line_scores[slope_index] = np.where(in_range, gathered, 0.0).sum(axis=0)

    best_slope_index_per_offset = np.argmax(line_scores, axis=0)
    best_score_per_offset = line_scores[best_slope_index_per_offset, offset_indices]
    global_best = float(best_score_per_offset.max())
    if global_best < minimum_mean_score * valid_rows.sum():
        return None
    left = np.concatenate([[-np.inf], best_score_per_offset[:-1]])
    right = np.concatenate([best_score_per_offset[1:], [-np.inf]])
    qualifying = (best_score_per_offset >= left) & (best_score_per_offset >= right) & (best_score_per_offset >= outermost_line_ratio * global_best)
    chosen_offset_index = int(np.flatnonzero(qualifying)[-1])
    return float(profiles.normal_offsets[chosen_offset_index]), float(slopes[best_slope_index_per_offset[chosen_offset_index]])


def extract_edge_points_near_centre(
    profiles: SideScoreProfiles, centre_offsets: np.ndarray, tolerance_pixels: float, minimum_score: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-sample sub-pixel peak within +-tolerance of `centre_offsets` (S,).

    Returns (positions along side px, sub-pixel normal offsets px, peak scores) for the
    samples whose peak is an interior maximum above `minimum_score`.
    """
    offsets = profiles.normal_offsets
    allowed = np.abs(offsets[None, :] - centre_offsets[:, None]) <= tolerance_pixels
    windowed_scores = np.where(allowed, profiles.scores, -np.inf)
    best_indices = np.argmax(windowed_scores, axis=1)
    sample_rows = np.arange(len(best_indices))
    best_scores = windowed_scores[sample_rows, best_indices]
    last_index = windowed_scores.shape[1] - 1
    left = windowed_scores[sample_rows, np.clip(best_indices - 1, 0, last_index)]
    right = windowed_scores[sample_rows, np.clip(best_indices + 1, 0, last_index)]
    interior_peak = (best_indices > 0) & (best_indices < last_index) & np.isfinite(left) & np.isfinite(right)
    keep = interior_peak & np.isfinite(best_scores) & (best_scores >= minimum_score)
    denominator = left - 2 * best_scores + right
    with np.errstate(invalid="ignore", divide="ignore"):
        sub_pixel_shift = np.where(keep & (denominator < 0), 0.5 * (left - right) / denominator, 0.0)
    sub_pixel_shift = np.clip(np.nan_to_num(sub_pixel_shift), -0.5, 0.5)
    peak_offsets = offsets[best_indices] + sub_pixel_shift
    return profiles.positions_pixels[keep], peak_offsets[keep], best_scores[keep]
