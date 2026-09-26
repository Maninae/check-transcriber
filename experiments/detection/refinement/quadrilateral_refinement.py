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
from experiments.detection.refinement.overlap_masking import (
    dilate_convex_quad,
    mask_cells_inside_other_quads,
    points_inside_convex_quad,
    select_nearby_quads,
)
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
MINIMUM_PAPER_COLOUR_SAMPLES = 40


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


def estimate_paper_colour(image: np.ndarray, corners: np.ndarray, inset_fraction: float, other_quads: list[np.ndarray]) -> np.ndarray:
    """Median colour over a bilinear grid spanning the quad's central region (ink is a minority).

    Grid points inside another detection's quad are skipped (unless that leaves too few).
    """
    grid_values = np.linspace(inset_fraction, 1 - inset_fraction, PAPER_COLOUR_GRID_SIZE)
    along_top, along_left = np.meshgrid(grid_values, grid_values)
    top = corners[0] * (1 - along_top[..., None]) + corners[1] * along_top[..., None]
    bottom = corners[3] * (1 - along_top[..., None]) + corners[2] * along_top[..., None]
    grid_points = top * (1 - along_left[..., None]) + bottom * along_left[..., None]
    sampled = cv2.remap(
        image, grid_points[..., 0].astype(np.float32), grid_points[..., 1].astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    number_of_channels = image.shape[2] if image.ndim == 3 else 1
    colours = sampled.reshape(-1, number_of_channels).astype(np.float64)
    covered = np.zeros(len(colours), dtype=bool)
    for quad in other_quads:
        covered |= points_inside_convex_quad(grid_points.reshape(-1, 2), quad)
    if (~covered).sum() >= MINIMUM_PAPER_COLOUR_SAMPLES:
        colours = colours[~covered]
    return np.median(colours, axis=0)


def band_pixels(fraction: float, short_side: float, config: CornerRefinementConfig) -> float:
    """A band fraction of the short side, clamped to the configured pixel range."""
    return float(np.clip(fraction * short_side, config.minimum_band_pixels, config.maximum_band_pixels))


def score_side(image, corners, side_index, pass_settings, paper_colour, config, dilated_other_quads):
    """Score grid for one side of the current quad, cells over other checks masked.

    Returns (profiles, number of samples).
    """
    side_start, side_end = corners[side_index], corners[(side_index + 1) % 4]
    side_length = float(np.hypot(*(side_end - side_start)))
    number_of_samples = int(np.clip(side_length / config.sample_spacing_pixels, config.minimum_samples_per_side, pass_settings["maximum_samples"]))
    corner_margin = max(config.corner_margin_fraction * side_length, config.minimum_corner_margin_pixels)
    profiles = score_side_edge_profiles(
        image, side_start, side_end, corners.mean(axis=0),
        pass_settings["inward_band"], pass_settings["outward_band"], number_of_samples, corner_margin,
        config.tangential_offsets_pixels, config.inner_window_pixels, config.outer_window_pixels,
        paper_colour,
        config.edge_score_mode, config.background_window_pixels, config.minimum_paper_background_contrast,
        config.tent_beyond_paper, config.contact_line_half_width_pixels, config.contact_line_contrast_unit,
        config.contact_line_below_contrast,
    )
    mask_cells_inside_other_quads(profiles, dilated_other_quads)
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
            config.minimum_edge_score, config.line_score_clip, config.keep_input_line_ratio,
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


def refine_side_narrow_first(image, corners, side_index, pass_settings, paper_colour, config, dilated_other_quads, random_generator, narrow_band):
    """Search a small band around the input side first; widen only if it finds no well-supported edge.

    A good detector side already sits within a few px of the paper edge, and the wide
    band's outermost-line rule can only make it worse. The narrow band cannot fool itself
    into printed rules when the side is deep inside the paper: its per-sample background
    colour is then paper-coloured, those samples carry no evidence and support fails.
    Returns ((profiles, count), curve, whether the narrow band was accepted).
    """
    if narrow_band is not None:
        narrow_settings = {**pass_settings, "inward_band": narrow_band, "outward_band": narrow_band}
        profiles, count = score_side(image, corners, side_index, narrow_settings, paper_colour, config, dilated_other_quads)
        curve = refine_one_side(profiles, count, narrow_settings, config, random_generator)
        if curve.inlier_count >= config.narrow_first_minimum_support * count:
            return (profiles, count), curve, True
    profiles, count = score_side(image, corners, side_index, pass_settings, paper_colour, config, dilated_other_quads)
    return (profiles, count), refine_one_side(profiles, count, pass_settings, config, random_generator), False


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


def describe_side_curve(curve: SideCurve, number_of_samples: int, short_side: float) -> dict:
    """Evidence summary of one fitted side (what the acceptance gate looks at)."""
    start_offset, end_offset = curve.offsets_at(np.array([0.0, curve.side_length]))
    return {
        "support_fraction": curve.inlier_count / max(number_of_samples, 1),
        "median_strength": curve.median_inlier_strength,
        "residual_rms": curve.inlier_residual_rms,
        "move_fraction_of_short_side": float(max(abs(start_offset), abs(end_offset)) / max(short_side, 1e-9)),
        "start_point": (curve.side_start + curve.unit_normal * start_offset).tolist(),
        "end_point": (curve.side_start + curve.unit_tangent * curve.side_length + curve.unit_normal * end_offset).tolist(),
    }


def refine_check_quadrilateral(
    image: np.ndarray,
    approximate_corners: np.ndarray,
    config: CornerRefinementConfig | None = None,
    other_check_quads: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, dict]:
    """Move each corner of an approximate quad onto the paper's physical corner.

    Args:
        image: full-resolution grey (H, W) or BGR (H, W, 3) uint8 image.
        approximate_corners: (4, 2) quad in either winding; the order is preserved.
        config: tuning knobs; defaults tuned on val.
        other_check_quads: the other detections in the same image; score cells inside
            them (dilated) are ignored so a side cannot lock onto another check's edge.
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
    largest_band = max(max(settings["inward_band"], settings["outward_band"]) for settings in pass_settings_list)
    nearby_quads = select_nearby_quads(input_corners, [np.asarray(quad, dtype=np.float64) for quad in (other_check_quads or [])], 2 * largest_band)
    dilated_other_quads = [dilate_convex_quad(quad, config.other_check_dilation_pixels) for quad in nearby_quads] if config.mask_other_checks else []
    paper_colour = estimate_paper_colour(image, input_corners, config.paper_sample_inset_fraction, nearby_quads if config.mask_other_checks else [])
    diagnostics: dict = {"paper_colour": paper_colour.tolist(), "passes": []}

    current_corners = input_corners.copy()
    for pass_index, pass_settings in enumerate(pass_settings_list):
        tracking_pass = config.final_pass_mode == "track" and pass_index == len(pass_settings_list) - 1 and pass_index > 0
        new_corners = current_corners.copy()
        scored_sides, side_curves, narrow_accepted = [], [], []
        for side_index in range(4):
            scored, curve, accepted = refine_side_narrow_first(
                image, current_corners, side_index, pass_settings, paper_colour, config, dilated_other_quads, random_generator,
                narrow_band=band_pixels(config.narrow_first_band_fraction, short_side, config) if pass_index == 0 and config.narrow_first_band_fraction > 0 else None,
            )
            scored_sides.append(scored)
            side_curves.append(curve)
            narrow_accepted.append(accepted)
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
        diagnostics["passes"].append({
            **pass_settings,
            "local_corner_fits": local_corner_fits,
            "sides_accepted_in_narrow_band": narrow_accepted,
            "sides_without_support": [curve.inlier_count == 0 for curve in side_curves],
            "side_statistics": [describe_side_curve(curve, count, short_side) for curve, (_, count) in zip(side_curves, scored_sides)],
        })
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
    """Refine every detection's corners; score, orientation flag and corner order are kept.

    Each check is refined knowing the others' (input) quads, see `overlap_masking`.
    """
    refined_checks = []
    for check_index, detected_check in enumerate(detected_checks):
        other_quads = [other.corners for other_index, other in enumerate(detected_checks) if other_index != check_index]
        refined_corners, diagnostics = refine_check_quadrilateral(image_bgr, detected_check.corners, config, other_quads)
        refined_checks.append(
            DetectedCheck(
                corners=refined_corners,
                score=detected_check.score,
                orientation_known=detected_check.orientation_known,
                extras={**detected_check.extras, "refinement": diagnostics},
            )
        )
    return refined_checks
