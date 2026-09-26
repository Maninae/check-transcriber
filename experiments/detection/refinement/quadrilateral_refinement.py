"""Refine an approximate check quadrilateral to sub-pixel corners using the paper's edges.

Per check, sampling the full-resolution uint8 image directly with bilinear remap (only
the profile grids are touched, so there is no crop to copy):

1. Paper colour = median of a grid inside the input quad.
2. Each pass, for each side: score a (position x normal offset) grid
   (`edge_profile_sampling`), pick the outermost strong line by integrating scores along
   candidate lines (`side_line_search`), then grow a robust curve from it: extract
   sub-pixel peaks within a few px of the current curve, refit, repeat
   (`robust_side_curve_fitting`). Pass 1 is straight and mostly inward; pass 2 is
   narrow and quadratic, so a curled side's bow is followed to the physical corner.
3. A side without enough support (off the frame, hidden, no contrast) keeps its line.
4. Corners = intersections of adjacent curves, so the input corner order is preserved.
5. Guard rails vs the INPUT quad: a corner that moved further than the cap reverts; a
   non-convex, flipped or badly resized quad reverts entirely.
"""

import cv2
import numpy as np

from experiments.detection.predictions.detected_check import DetectedCheck
from experiments.detection.refinement.edge_profile_sampling import score_side_edge_profiles
from experiments.detection.refinement.refinement_config import CornerRefinementConfig
from experiments.detection.refinement.robust_side_curve_fitting import (
    SideCurve,
    fit_robust_side_curve,
    intersect_side_curves,
)
from experiments.detection.refinement.side_edge_tracking import track_edge_path
from experiments.detection.refinement.side_line_search import (
    extract_edge_points_near_centre,
    search_outermost_strong_line,
)

PAPER_COLOUR_GRID_SIZE = 24


def compute_short_side_length(corners: np.ndarray) -> float:
    """Mean length of the shorter pair of opposite sides."""
    side_lengths = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
    return float(min(side_lengths[0] + side_lengths[2], side_lengths[1] + side_lengths[3]) / 2)


def compute_signed_area(corners: np.ndarray) -> float:
    """Shoelace signed area (positive for clockwise order in image coordinates)."""
    x, y = corners[:, 0], corners[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def is_convex_with_orientation(corners: np.ndarray, expected_sign: float) -> bool:
    """True when every turn has the same sign as `expected_sign` (convex, same winding)."""
    edges = np.roll(corners, -1, axis=0) - corners
    next_edges = np.roll(edges, -1, axis=0)
    turn_cross_products = edges[:, 0] * next_edges[:, 1] - edges[:, 1] * next_edges[:, 0]
    return bool(np.all(turn_cross_products * expected_sign > 0))


def estimate_paper_colour(image: np.ndarray, corners: np.ndarray, inset_fraction: float) -> np.ndarray:
    """Median colour over a bilinear grid spanning the quad's central region (ink is a minority)."""
    grid_values = np.linspace(inset_fraction, 1 - inset_fraction, PAPER_COLOUR_GRID_SIZE)
    along_top, along_left = np.meshgrid(grid_values, grid_values)
    top = corners[0] * (1 - along_top[..., None]) + corners[1] * along_top[..., None]
    bottom = corners[3] * (1 - along_top[..., None]) + corners[2] * along_top[..., None]
    grid_points = top * (1 - along_left[..., None]) + bottom * along_left[..., None]
    sampled = cv2.remap(
        image, grid_points[..., 0].astype(np.float32), grid_points[..., 1].astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    number_of_channels = image.shape[2] if image.ndim == 3 else 1
    return np.median(sampled.reshape(-1, number_of_channels).astype(np.float64), axis=0)


def band_pixels(fraction: float, short_side: float, config: CornerRefinementConfig) -> float:
    """A band fraction of the short side, clamped to the configured pixel range."""
    return float(np.clip(fraction * short_side, config.minimum_band_pixels, config.maximum_band_pixels))


def score_side(image, corners, side_index, pass_settings, paper_colour, config):
    """Score grid for one side of the current quad; returns (profiles, number of samples)."""
    side_start, side_end = corners[side_index], corners[(side_index + 1) % 4]
    side_length = float(np.hypot(*(side_end - side_start)))
    number_of_samples = int(np.clip(side_length / config.sample_spacing_pixels, config.minimum_samples_per_side, pass_settings["maximum_samples"]))
    corner_margin = max(config.corner_margin_fraction * side_length, config.minimum_corner_margin_pixels)
    profiles = score_side_edge_profiles(
        image, side_start, side_end, corners.mean(axis=0),
        pass_settings["inward_band"], pass_settings["outward_band"], number_of_samples, corner_margin,
        config.tangential_offsets_pixels, config.inner_window_pixels, config.outer_window_pixels,
        config.far_outer_gap_pixels, config.far_outer_window_pixels, paper_colour,
        config.edge_score_mode, config.background_window_pixels, config.minimum_paper_background_contrast,
    )
    return profiles, number_of_samples


def refine_one_side(profiles, number_of_samples: int, pass_settings: dict, config, random_generator) -> SideCurve:
    """Line search (pass 1) then curve growing for one side; a zero-offset curve on failure."""
    side_length = profiles.side_length
    current_curve = SideCurve(profiles.side_start, profiles.unit_tangent, profiles.unit_normal, side_length, np.zeros(2))
    if pass_settings["search_line"]:
        maximum_angle = float(np.clip(
            np.degrees(np.arctan(2 * pass_settings["inward_band"] / max(side_length, 1.0))),
            config.minimum_line_angle_degrees, config.maximum_line_angle_degrees,
        ))
        line = search_outermost_strong_line(
            profiles, maximum_angle, config.line_angle_step_degrees, config.outermost_line_ratio,
            config.minimum_edge_score, config.line_score_clip,
        )
        if line is None:
            return current_curve
        offset_at_middle, slope = line
        seed_curve = SideCurve(profiles.side_start, profiles.unit_tangent, profiles.unit_normal, side_length, np.array([offset_at_middle, slope * side_length / 2]))
    else:
        seed_curve = current_curve
    required_inliers = max(config.minimum_inlier_count, int(np.ceil(config.minimum_inlier_fraction * number_of_samples)))
    fitted_curve = None
    for _ in range(config.curve_grow_iterations):
        guide_curve = fitted_curve or seed_curve
        centre_offsets = guide_curve.offsets_at(profiles.positions_pixels)
        positions, offsets, strengths = extract_edge_points_near_centre(profiles, centre_offsets, config.point_tolerance_pixels, config.minimum_edge_score)
        candidate = fit_robust_side_curve(
            seed_curve, positions, offsets, strengths, pass_settings["degree"], config.ransac_inlier_distance_pixels, config.ransac_iterations, random_generator
        )
        if candidate is None or candidate.inlier_count < required_inliers:
            break
        fitted_curve = candidate
    return fitted_curve or current_curve


def track_side_points(profiles, config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Viterbi-tracked edge points of one side (positions, offsets, strengths), weak ones dropped."""
    positions, offsets, strengths = track_edge_path(
        profiles, config.line_score_clip, config.track_maximum_step_offsets, config.track_step_cost, config.track_centre_cost_per_pixel
    )
    strong = strengths >= config.minimum_edge_score
    return positions[strong], offsets[strong], strengths[strong]


def fit_corner_local_line(side_curve: SideCurve, points, near_start: bool, fraction: float, config, random_generator) -> SideCurve | None:
    """Straight fit to the tracked points near one corner, IF they bend away from the side curve.

    Returns None (keep the side curve) when there are too few points or when their median
    deviation from `side_curve` is below `corner_bend_threshold_pixels`.
    """
    positions, offsets, strengths = points
    if len(positions) == 0:
        return None
    relative = positions / max(side_curve.side_length, 1e-9)
    keep = relative <= fraction if near_start else relative >= 1 - fraction
    if keep.sum() < config.minimum_inlier_count:
        return None
    if abs(float(np.median(offsets[keep] - side_curve.offsets_at(positions[keep])))) < config.corner_bend_threshold_pixels:
        return None
    frame = side_curve
    local = fit_robust_side_curve(frame, positions[keep], offsets[keep], strengths[keep], 1, config.ransac_inlier_distance_pixels, config.ransac_iterations, random_generator)
    if local is None or local.inlier_count < max(config.minimum_inlier_count, int(0.5 * keep.sum())):
        return None
    return local


def refine_check_quadrilateral(
    image: np.ndarray, approximate_corners: np.ndarray, config: CornerRefinementConfig | None = None
) -> tuple[np.ndarray, dict]:
    """Move each corner of an approximate quad onto the paper's physical corner.

    Args:
        image: full-resolution grey (H, W) or BGR (H, W, 3) uint8 image.
        approximate_corners: (4, 2) quad in either winding; the order is preserved.
        config: tuning knobs; defaults tuned on val.
    Returns:
        (refined (4, 2) float64 corners, diagnostics: bands, which sides fell back per
        pass, which guard rails fired).
    """
    config = config or CornerRefinementConfig()
    input_corners = np.asarray(approximate_corners, dtype=np.float64)
    random_generator = np.random.default_rng(config.random_seed)
    short_side = compute_short_side_length(input_corners)
    pass_settings_list = [
        {
            "inward_band": band_pixels(inward_fraction, short_side, config),
            "outward_band": band_pixels(outward_fraction, short_side, config),
            "search_line": pass_index == 0,
            "degree": degree,
            "maximum_samples": maximum_samples,
        }
        for pass_index, (inward_fraction, outward_fraction, degree, maximum_samples) in enumerate(zip(
            config.inward_band_fraction_per_pass, config.outward_band_fraction_per_pass, config.curve_degree_per_pass,
            config.maximum_samples_per_side_per_pass,
        ))
    ]
    paper_colour = estimate_paper_colour(image, input_corners, config.paper_sample_inset_fraction)
    diagnostics: dict = {"paper_colour": paper_colour.tolist(), "passes": []}

    current_corners = input_corners.copy()
    for pass_index, pass_settings in enumerate(pass_settings_list):
        tracking_pass = config.final_pass_mode == "track" and pass_index == len(pass_settings_list) - 1 and pass_index > 0
        new_corners = current_corners.copy()
        scored_sides = [score_side(image, current_corners, side_index, pass_settings, paper_colour, config) for side_index in range(4)]
        side_curves = [refine_one_side(profiles, count, pass_settings, config, random_generator) for profiles, count in scored_sides]
        corner_curve_pairs = [(side_curves[(corner_index - 1) % 4], side_curves[corner_index]) for corner_index in range(4)]
        local_corner_fits = [False] * 4
        if tracking_pass:
            tracked_points = [track_side_points(profiles, config) for profiles, _ in scored_sides]
            for corner_index in range(4):
                incoming_side, outgoing_side = (corner_index - 1) % 4, corner_index
                incoming_local = fit_corner_local_line(side_curves[incoming_side], tracked_points[incoming_side], False, config.corner_local_fraction, config, random_generator)
                outgoing_local = fit_corner_local_line(side_curves[outgoing_side], tracked_points[outgoing_side], True, config.corner_local_fraction, config, random_generator)
                local_corner_fits[corner_index] = incoming_local is not None or outgoing_local is not None
                corner_curve_pairs[corner_index] = (incoming_local or side_curves[incoming_side], outgoing_local or side_curves[outgoing_side])
        for corner_index, (incoming_curve, outgoing_curve) in enumerate(corner_curve_pairs):
            intersection = intersect_side_curves(incoming_curve, outgoing_curve)
            if intersection is not None:
                new_corners[corner_index] = intersection
        diagnostics["passes"].append({**pass_settings, "local_corner_fits": local_corner_fits, "sides_without_support": [curve.inlier_count == 0 for curve in side_curves]})
        current_corners = new_corners
    refined_corners = apply_guard_rails(input_corners, current_corners, pass_settings_list[0]["inward_band"], config, diagnostics)
    return refined_corners, diagnostics


def apply_guard_rails(input_corners: np.ndarray, refined_corners: np.ndarray, first_band: float, config, diagnostics: dict) -> np.ndarray:
    """Revert corners (or the whole quad) whose refinement looks implausible."""
    guarded = refined_corners.copy()
    corner_moves = np.linalg.norm(guarded - input_corners, axis=1)
    too_far = (corner_moves > config.maximum_corner_move_band_multiple * first_band) | ~np.all(np.isfinite(guarded), axis=1)
    guarded[too_far] = input_corners[too_far]
    diagnostics["corners_reverted_for_distance"] = too_far.tolist()
    input_side_lengths = np.linalg.norm(np.roll(input_corners, -1, axis=0) - input_corners, axis=1)
    guarded_side_lengths = np.linalg.norm(np.roll(guarded, -1, axis=0) - guarded, axis=1)
    length_change = np.abs(guarded_side_lengths / np.maximum(input_side_lengths, 1e-9) - 1)
    plausible = is_convex_with_orientation(guarded, np.sign(compute_signed_area(input_corners))) and bool(
        np.all(length_change <= config.maximum_side_length_change_fraction)
    )
    diagnostics["quad_reverted"] = not plausible
    return guarded if plausible else input_corners.copy()


def refine_detected_checks(
    image_bgr: np.ndarray, detected_checks: list[DetectedCheck], config: CornerRefinementConfig | None = None
) -> list[DetectedCheck]:
    """Refine every detection's corners; score, orientation flag and corner order are kept."""
    refined_checks = []
    for detected_check in detected_checks:
        refined_corners, diagnostics = refine_check_quadrilateral(image_bgr, detected_check.corners, config)
        refined_checks.append(
            DetectedCheck(
                corners=refined_corners,
                score=detected_check.score,
                orientation_known=detected_check.orientation_known,
                extras={**detected_check.extras, "refinement": diagnostics},
            )
        )
    return refined_checks
