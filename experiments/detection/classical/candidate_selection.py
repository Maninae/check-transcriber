"""Pick the final set of checks from the pooled, verified candidate quads.

Each mask family usually finds the same check, so candidates arrive in near-duplicate
clusters. Selection is a greedy non-maximum suppression by rank (evidence score, then
rectangularity), with two suppression rules:

- duplicate: IoU with an already-kept quad above `duplicate_iou_threshold`;
- containment: most of the candidate's area lies inside a kept quad (a text box or an
  inner printed border of a check that was already found);
- coverage: most of the candidate's area is already covered by the union of kept quads
  (a line-built quad spanning two found checks and the gap between them).

Finally, a scene-level size prior: checks in one photo share the camera distance, so a
kept quad smaller than `relative_area_floor` x the median kept area is a fragment (a
printed box or a shadow-split piece of a check) and is dropped.

Genuinely overlapping checks (a stack in `loose_overlap`) share far less than the
duplicate threshold, so both survive.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from experiments.detection.classical.classical_detector_config import ClassicalDetectorConfig
from experiments.detection.classical.quadrilateral_geometry import (
    convex_polygon_intersection_area,
    convex_quadrilateral_iou,
    quadrilateral_area,
)

RECTANGULARITY_RANK_WEIGHT = 0.25


@dataclass
class VerifiedCandidate:
    """A candidate that passed the gates, with everything selection ranks on."""

    corners: np.ndarray  # (4, 2) working pixels, clockwise
    score: float
    rectangularity: float
    side_supports: np.ndarray
    side_strengths: np.ndarray
    source_name: str
    weakest_side_strength_rank_weight: float = 0.0

    @property
    def rank_value(self) -> float:
        """Sort key: edge evidence, then border strength and rectangularity as tie-breakers.

        The weakest side's strength matters most: a quad overshooting a check's end along a
        collinear background line has one weak, borrowed side (or an image-border side,
        which only scores the threshold level) and loses to the tight quad.
        """
        weakest_side_strength = float(np.min(self.side_strengths))
        return (
            self.score
            + RECTANGULARITY_RANK_WEIGHT * min(self.rectangularity, 1.0)
            + self.weakest_side_strength_rank_weight * weakest_side_strength
        )


def is_suppressed_by_kept(
    candidate: VerifiedCandidate, kept_candidates: list[VerifiedCandidate], config: ClassicalDetectorConfig
) -> bool:
    """Whether a kept candidate already explains this one (duplicate or containment)."""
    candidate_area = quadrilateral_area(candidate.corners)
    for kept_candidate in kept_candidates:
        if convex_quadrilateral_iou(candidate.corners, kept_candidate.corners) > config.duplicate_iou_threshold:
            return True
        shared_area = convex_polygon_intersection_area(candidate.corners, kept_candidate.corners)
        smaller_area = min(candidate_area, quadrilateral_area(kept_candidate.corners))
        if smaller_area > 0 and shared_area / smaller_area > config.maximum_contained_fraction:
            return True
    return False


def fraction_covered_by_mask(corners: np.ndarray, kept_mask: np.ndarray) -> float:
    """Share of a quad's rasterized pixels already set in the kept-union mask."""
    candidate_mask = np.zeros_like(kept_mask)
    cv2.fillConvexPoly(candidate_mask, np.rint(corners).astype(np.int32), 1)
    candidate_pixel_count = int(candidate_mask.sum())
    if candidate_pixel_count == 0:
        return 1.0
    return float((candidate_mask & kept_mask).sum()) / candidate_pixel_count


def select_non_overlapping_candidates(
    candidates: list[VerifiedCandidate], config: ClassicalDetectorConfig, image_shape: tuple[int, int]
) -> list[VerifiedCandidate]:
    """Greedy suppression in rank order; `image_shape` is the working (height, width)."""
    kept_candidates: list[VerifiedCandidate] = []
    kept_union_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for candidate in sorted(candidates, key=lambda item: item.rank_value, reverse=True):
        if is_suppressed_by_kept(candidate, kept_candidates, config):
            continue
        if kept_candidates and fraction_covered_by_mask(candidate.corners, kept_union_mask) > config.maximum_covered_fraction:
            continue
        kept_candidates.append(candidate)
        cv2.fillConvexPoly(kept_union_mask, np.rint(candidate.corners).astype(np.int32), 1)
    return drop_fragments_by_relative_area(kept_candidates, config.relative_area_floor)


def drop_fragments_by_relative_area(candidates: list[VerifiedCandidate], relative_area_floor: float) -> list[VerifiedCandidate]:
    """Remove quads much smaller than the scene's median kept quad (needs 2+ candidates)."""
    if len(candidates) < 2:
        return candidates
    median_area = float(np.median([quadrilateral_area(candidate.corners) for candidate in candidates]))
    return [candidate for candidate in candidates if quadrilateral_area(candidate.corners) >= relative_area_floor * median_area]
