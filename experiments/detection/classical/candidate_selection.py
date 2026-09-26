"""Pick the final set of checks from the pooled, verified candidate quads.

Each mask family usually finds the same check, so candidates arrive in near-duplicate
clusters. Selection is a greedy non-maximum suppression by rank (evidence score, then
rectangularity), with two suppression rules:

- duplicate: IoU with an already-kept quad above `duplicate_iou_threshold`;
- containment: most of the candidate's area lies inside a kept quad (a text box or an
  inner printed border of a check that was already found).

Genuinely overlapping checks (a stack in `loose_overlap`) share far less than the
duplicate threshold, so both survive.
"""

from dataclasses import dataclass

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
    source_name: str

    @property
    def rank_value(self) -> float:
        """Sort key: edge evidence first, rectangularity as a softer tie-breaker."""
        return self.score + RECTANGULARITY_RANK_WEIGHT * min(self.rectangularity, 1.0)


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


def select_non_overlapping_candidates(
    candidates: list[VerifiedCandidate], config: ClassicalDetectorConfig
) -> list[VerifiedCandidate]:
    """Greedy suppression in rank order."""
    kept_candidates: list[VerifiedCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.rank_value, reverse=True):
        if not is_suppressed_by_kept(candidate, kept_candidates, config):
            kept_candidates.append(candidate)
    return kept_candidates
