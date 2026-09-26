"""Refine an approximate check quadrilateral to sub-pixel corners using the paper's edges.

Per check, on a full-resolution float crop around the quad:

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
from experiments.detection.refinement.side_line_search import (
    extract_edge_points_near_centre,
    search_outermost_strong_line,
)

CROP_PADDING_PIXELS = 8
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


def extract_float_crop(image: np.ndarray, corners: np.ndarray, padding: float) -> tuple[np.ndarray, np.ndarray]:
    """Float32 (H, W, C) crop covering the quad plus `padding`, and its full-image origin."""
    image_height, image_width = image.shape[:2]
    x_min, y_min = np.floor(corners.min(axis=0) - padding).astype(int)
    x_max, y_max = np.ceil(corners.max(axis=0) + padding).astype(int)
    x_min, y_min = min(max(x_min, 0), image_width - 1), min(max(y_min, 0), image_height - 1)
    x_max, y_max = max(min(x_max, image_width - 1), x_min), max(min(y_max, image_height - 1), y_min)
    crop = image[y_min : y_max + 1, x_min : x_max + 1].astype(np.float32)
    if crop.ndim == 2:
        crop = crop[:, :, None]
    return crop, np.array([x_min, y_min], dtype=np.float64)


def estimate_paper_colour(crop: np.ndarray, crop_origin: np.ndarray, corners: np.ndarray, inset_fraction: float) -> np.ndarray:
    """Median colour over a bilinear grid spanning the quad's central region (ink is a minority)."""
    grid_values = np.linspace(inset_fraction, 1 - inset_fraction, PAPER_COLOUR_GRID_SIZE)
    along_top, along_left = np.meshgrid(grid_values, grid_values)
    top = corners[0] * (1 - along_top[..., None]) + corners[1] * along_top[..., None]
    bottom = corners[3] * (1 - along_top[..., None]) + corners[2] * along_top[..., None]
    grid_points = top * (1 - along_left[..., None]) + bottom * along_left[..., None] - crop_origin
    sampled = cv2.remap(
        crop, grid_points[..., 0].astype(np.float32), grid_points[..., 1].astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    return np.median(sampled.reshape(-1, crop.shape[2]), axis=0)


def band_pixels(fraction: float, short_side: float, config: CornerRefinementConfig) -> float:
    """A band fraction of the short side, clamped to the configured pixel range."""
    return float(np.clip(fraction * short_side, config.minimum_band_pixels, config.maximum_band_pixels))


def refine_one_side(crop, crop_origin, image_size, corners, side_index, pass_settings, paper_colour, config, random_generator) -> SideCurve:
    """Line search then curve growing for one side; returns a zero-offset curve on failure."""
    side_start, side_end = corners[side_index], corners[(side_index + 1) % 4]
    side_length = float(np.hypot(*(side_end - side_start)))
    number_of_samples = int(np.clip(side_length / config.sample_spacing_pixels, config.minimum_samples_per_side, config.maximum_samples_per_side))
    corner_margin = max(config.corner_margin_fraction * side_length, config.minimum_corner_margin_pixels)
    profiles = score_side_edge_profiles(
        crop, crop_origin, image_size, side_start, side_end, corners.mean(axis=0),
        pass_settings["inward_band"], pass_settings["outward_band"], number_of_samples, corner_margin,
        config.tangential_offsets_pixels, config.inner_window_pixels, config.outer_window_pixels, paper_colour,
    )
    current_curve = SideCurve(profiles.side_start, profiles.unit_tangent, profiles.unit_normal, side_length, np.zeros(2))
    line = search_outermost_strong_line(
        profiles, pass_settings["maximum_angle"], config.line_angle_step_degrees, config.outermost_line_ratio,
        config.minimum_edge_score, config.line_score_clip,
    )
    if line is None:
        return current_curve
    offset_at_middle, slope = line
    half_length = side_length / 2
    seed_curve = SideCurve(profiles.side_start, profiles.unit_tangent, profiles.unit_normal, side_length, np.array([offset_at_middle, slope * half_length]))
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
            "maximum_angle": maximum_angle,
            "degree": degree,
        }
        for inward_fraction, outward_fraction, maximum_angle, degree in zip(
            config.inward_band_fraction_per_pass, config.outward_band_fraction_per_pass,
            config.maximum_line_angle_degrees_per_pass, config.curve_degree_per_pass,
        )
    ]
    largest_band = max(max(settings["inward_band"], settings["outward_band"]) for settings in pass_settings_list)
    window_reach = max(config.inner_window_pixels, config.outer_window_pixels) + max(abs(offset) for offset in config.tangential_offsets_pixels)
    crop, crop_origin = extract_float_crop(image, input_corners, 2 * largest_band + window_reach + CROP_PADDING_PIXELS)
    image_size = (image.shape[1], image.shape[0])
    paper_colour = estimate_paper_colour(crop, crop_origin, input_corners, config.paper_sample_inset_fraction)
    diagnostics: dict = {"paper_colour": paper_colour.tolist(), "passes": []}

    current_corners = input_corners.copy()
    for pass_settings in pass_settings_list:
        side_curves = [
            refine_one_side(crop, crop_origin, image_size, current_corners, side_index, pass_settings, paper_colour, config, random_generator)
            for side_index in range(4)
        ]
        new_corners = current_corners.copy()
        for corner_index in range(4):
            intersection = intersect_side_curves(side_curves[(corner_index - 1) % 4], side_curves[corner_index])
            if intersection is not None:
                new_corners[corner_index] = intersection
        diagnostics["passes"].append({**pass_settings, "sides_without_support": [curve.inlier_count == 0 for curve in side_curves]})
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
