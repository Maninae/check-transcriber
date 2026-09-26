"""Top-level classical check detector: full-res BGR photo in, check quadrilaterals out.

Pipeline (each stage in its own module):
1. `working_image_channels`: resize, text-suppressed lightness, chroma, paper score,
   texture, gradients.
2. `candidate_regions`: smooth-paper, paper-score and edge-bounded-cell masks, their
   check-sized connected regions.
3. `quadrilateral_fitting`: approxPolyN quad + per-side line refit for each region;
   `line_segment_extraction` + `line_quadrilateral_hypotheses`: rectangles from parallel
   Hough segment pairs. Every quad is then snapped (`edge_line_snapping`) to the nearest
   lightness step.
4. `quadrilateral_verification`: shape gates and an edge-support score;
   `interior_appearance`: the inside must look like printed paper;
   `interior_seam_detection`: no other check's border may cross the inside.
5. `candidate_selection`: greedy duplicate / containment suppression.
6. `full_resolution_edge_refinement`: sub-pixel side snapping on the original image.

Output corners are clockwise, full-resolution, `orientation_known=False` (a classical
detector cannot tell a check's top from its bottom).
"""

import cv2
import numpy as np

from experiments.detection.classical.candidate_regions import extract_all_candidate_regions
from experiments.detection.classical.candidate_selection import (
    VerifiedCandidate,
    select_non_overlapping_candidates,
)
from experiments.detection.classical.classical_detector_config import ClassicalDetectorConfig
from experiments.detection.classical.edge_line_snapping import snap_quadrilateral_sides_to_edges
from experiments.detection.classical.interior_appearance import (
    interior_looks_like_check,
    measure_interior_appearance,
)
from experiments.detection.classical.full_resolution_edge_refinement import refine_corners_at_full_resolution
from experiments.detection.classical.interior_seam_detection import measure_interior_seam_strength
from experiments.detection.classical.line_quadrilateral_hypotheses import build_line_quadrilateral_hypotheses
from experiments.detection.classical.line_segment_extraction import extract_line_segments
from experiments.detection.classical.quadrilateral_fitting import fit_quadrilateral_to_contour
from experiments.detection.classical.quadrilateral_fitting import FittedQuadrilateral
from experiments.detection.classical.quadrilateral_geometry import order_corners_clockwise
from experiments.detection.classical.quadrilateral_verification import (
    measure_quadrilateral_evidence,
    passes_geometry_gates,
)
from experiments.detection.classical.working_image_channels import (
    WorkingImageChannels,
    build_working_image_channels,
)
from experiments.detection.predictions.detected_check import DetectedCheck


def collect_verified_candidates(
    channels: WorkingImageChannels, config: ClassicalDetectorConfig
) -> list[VerifiedCandidate]:
    """Every region-derived quad that passes the shape gates and the evidence threshold."""
    fitted_quads = []
    for region in extract_all_candidate_regions(channels, config):
        fitted_quad = fit_quadrilateral_to_contour(
            region.contour,
            region.source_name,
            minimum_rectangularity=config.minimum_region_rectangularity,
            aspect_range=(min(config.minimum_aspect_ratio, config.border_truncated_aspect_range[0]),
                          max(config.maximum_aspect_ratio, config.border_truncated_aspect_range[1])),
        )
        if fitted_quad is not None and fitted_quad.rectangularity >= config.minimum_region_rectangularity:
            fitted_quads.append(fitted_quad)
    if config.use_line_hypotheses:
        line_segments = extract_line_segments(channels, config)
        for corners in build_line_quadrilateral_hypotheses(line_segments, channels.long_side_pixels, config):
            fitted_quads.append(FittedQuadrilateral(corners, config.line_hypothesis_rectangularity, "lines"))

    verified_candidates = []
    for fitted_quad in fitted_quads:
        verified_candidate = verify_fitted_quadrilateral(fitted_quad, channels, config)
        if verified_candidate is not None:
            verified_candidates.append(verified_candidate)
    return verified_candidates


def verify_fitted_quadrilateral(
    fitted_quad: FittedQuadrilateral, channels: WorkingImageChannels, config: ClassicalDetectorConfig
) -> VerifiedCandidate | None:
    """Snap, then run the gates cheapest first; a VerifiedCandidate or None when any gate fails."""
    image_size = (channels.lightness.shape[1], channels.lightness.shape[0])
    if not passes_geometry_gates(fitted_quad.corners, image_size, config):
        return None  # cheap pre-check: snapping rarely rescues a badly shaped quad
    snap_search_radius = max(2, int(round(config.working_snap_search_fraction * channels.long_side_pixels)))
    snapped_corners = order_corners_clockwise(
        snap_quadrilateral_sides_to_edges(
            channels.lightness, fitted_quad.corners, snap_search_radius,
            config.boundary_samples_per_side, config.working_snap_minimum_step,
        )
    )
    if not passes_geometry_gates(snapped_corners, image_size, config):
        return None
    evidence = measure_quadrilateral_evidence(snapped_corners, channels, config)
    if evidence.score < config.minimum_verification_score:
        return None
    if np.sort(evidence.side_supports)[1] < config.minimum_edge_support:
        return None
    if fitted_quad.source_name == "lines" and evidence.score < config.minimum_line_hypothesis_score:
        return None
    if measure_interior_seam_strength(snapped_corners, channels, config.seam_gradient_threshold) > config.maximum_interior_seam_strength:
        return None
    if not interior_looks_like_check(measure_interior_appearance(snapped_corners, channels, config), config):
        return None
    return VerifiedCandidate(
        corners=snapped_corners,
        score=evidence.score,
        rectangularity=fitted_quad.rectangularity,
        side_supports=evidence.side_supports,
        side_strengths=evidence.side_strengths,
        source_name=fitted_quad.source_name,
        weakest_side_strength_rank_weight=config.weakest_side_strength_rank_weight,
    )


def detect_checks_classical(image_bgr: np.ndarray, config: ClassicalDetectorConfig | None = None) -> list[DetectedCheck]:
    """Find every check in a full-resolution BGR photo."""
    config = config or ClassicalDetectorConfig()
    channels = build_working_image_channels(image_bgr, config)
    selected_candidates = select_non_overlapping_candidates(
        collect_verified_candidates(channels, config), config, channels.lightness.shape
    )
    full_resolution_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if config.refine_corners_at_full_resolution else None

    detected_checks = []
    for candidate in selected_candidates:
        # pixel-center convention: working pixel centers map to (x + 0.5) * scale - 0.5
        full_resolution_corners = (candidate.corners + 0.5) * channels.scale_to_full_resolution - 0.5
        if full_resolution_gray is not None:
            full_resolution_corners = refine_corners_at_full_resolution(full_resolution_gray, full_resolution_corners, config)
        detected_checks.append(
            DetectedCheck(
                corners=order_corners_clockwise(full_resolution_corners),
                score=float(min(candidate.score, 1.0)),
                orientation_known=False,
                extras={"source_name": candidate.source_name, "side_supports": candidate.side_supports.tolist()},
            )
        )
    return detected_checks
