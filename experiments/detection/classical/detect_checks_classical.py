"""Top-level classical check detector: full-res BGR photo in, check quadrilaterals out.

Pipeline (each stage in its own module):
1. `working_image_channels`: resize, text-suppressed lightness, chroma, paper score,
   texture, gradients.
2. `candidate_regions`: smooth-paper, paper-score and edge-bounded-cell masks, their
   check-sized connected regions.
3. `quadrilateral_fitting`: approxPolyN quad + per-side line refit for each region,
   then `edge_line_snapping` moves each side onto the nearest lightness step.
4. `quadrilateral_verification`: shape gates and an edge-support score;
   `interior_appearance`: the inside must look like printed paper.
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
from experiments.detection.classical.quadrilateral_fitting import fit_quadrilateral_to_contour
from experiments.detection.classical.quadrilateral_geometry import order_corners_clockwise
from experiments.detection.classical.quadrilateral_fitting import FittedQuadrilateral
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
    snap_search_radius = max(2, int(round(config.working_snap_search_fraction * channels.long_side_pixels)))
    verified_candidates = []
    for region in extract_all_candidate_regions(channels, config):
        fitted_quad = fit_quadrilateral_to_contour(region.contour, region.source_name)
        if fitted_quad is None or fitted_quad.rectangularity < config.minimum_region_rectangularity:
            continue
        snapped_corners = order_corners_clockwise(
            snap_quadrilateral_sides_to_edges(
                channels.lightness, fitted_quad.corners, snap_search_radius,
                config.boundary_samples_per_side, config.working_snap_minimum_step,
            )
        )
        fitted_quad = FittedQuadrilateral(snapped_corners, fitted_quad.rectangularity, fitted_quad.source_name)
        if not passes_geometry_gates(fitted_quad.corners, channels.area_pixels, config):
            continue
        evidence = measure_quadrilateral_evidence(fitted_quad.corners, channels, config)
        if evidence.score < config.minimum_verification_score:
            continue
        if np.sort(evidence.side_supports)[1] < config.minimum_edge_support:
            continue
        if not interior_looks_like_check(measure_interior_appearance(fitted_quad.corners, channels, config), config):
            continue
        verified_candidates.append(
            VerifiedCandidate(
                corners=fitted_quad.corners,
                score=evidence.score,
                rectangularity=fitted_quad.rectangularity,
                side_supports=evidence.side_supports,
                source_name=region.source_name,
            )
        )
    return verified_candidates


def detect_checks_classical(image_bgr: np.ndarray, config: ClassicalDetectorConfig | None = None) -> list[DetectedCheck]:
    """Find every check in a full-resolution BGR photo."""
    config = config or ClassicalDetectorConfig()
    channels = build_working_image_channels(image_bgr, config)
    selected_candidates = select_non_overlapping_candidates(collect_verified_candidates(channels, config), config)
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
