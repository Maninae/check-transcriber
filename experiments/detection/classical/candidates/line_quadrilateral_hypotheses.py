"""Rectangle hypotheses from pairs of parallel edge segments (the line-based generator).

Masks fail when paper and background look alike (pink check on pale wood, any check on
a white sheet) or when checks touch; their straight borders still show. Construction:

1. Parallel pair: segments i, j within `line_parallel_tolerance_degrees`, separated by a
   check-sized distance, whose extents overlap along their shared direction.
2. End positions along that direction: the pair's own extent ends, plus every roughly
   perpendicular segment ("end cap") lying between the two lines, e.g. the seam between
   two touching checks or a check's short border.
3. Every (start, end) choice of positions whose span has a check aspect with the pair's
   separation, and which both segments actually cover, becomes a quad: the two pair
   lines intersected with the two end lines (a cap's own line when the end is a cap, so
   perspective is preserved, else a perpendicular).

Hypotheses are deliberately generous; the shared snap / border / interior verification
decides which are checks. Straight-line textures (gingham, tartan, tile grids) would make
this combinatorial, so it is budgeted: pairs are visited longest first up to
`line_maximum_pairs`, each pair keeps its longest `MAXIMUM_CAPS_PER_PAIR` caps, and only
the `line_maximum_hypotheses` best-supported hypotheses are returned (support = how much of
the four sides observed segments cover).
"""

import numpy as np

from experiments.detection.classical.classical_detector_config import ClassicalDetectorConfig
from experiments.detection.classical.geometry.quadrilateral_geometry import intersect_lines, order_corners_clockwise

MINIMUM_PAIR_OVERLAP_FRACTION = 0.25  # of the shorter segment
MINIMUM_SPAN_COVERAGE_FRACTION = 0.4  # each pair segment must cover this much of the hypothesis span
MINIMUM_CAP_COVERAGE_FRACTION = 0.5  # an end cap must span this much of the pair separation
POSITION_DEDUPLICATION_PIXELS = 4.0
HYPOTHESIS_DEDUPLICATION_GRID_PIXELS = 6.0
MAXIMUM_CAPS_PER_PAIR = 6
CAP_SUPPORT_WEIGHT = 0.5  # a real cap line at an end adds this much support per end


def segment_lines(segments: np.ndarray) -> np.ndarray:
    """(N, 4) lines as (vx, vy, x0, y0) with unit direction, anchored at each segment start."""
    directions = segments[:, 2:] - segments[:, :2]
    directions = directions / np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-9)
    return np.concatenate([directions, segments[:, :2]], axis=1)


def interval_overlap(first_interval: tuple[float, float], second_interval: tuple[float, float]) -> float:
    """Length of the intersection of two 1-D intervals (0 when disjoint)."""
    return max(0.0, min(first_interval[1], second_interval[1]) - max(first_interval[0], second_interval[0]))


def build_line_quadrilateral_hypotheses(
    segments: np.ndarray, image_long_side: int, config: ClassicalDetectorConfig
) -> list[np.ndarray]:
    """All rectangle hypotheses (4, 2 clockwise corners) from the segment set."""
    if len(segments) < 2:
        return []
    lines = segment_lines(segments)
    angles = np.degrees(np.arctan2(lines[:, 1], lines[:, 0])) % 180.0
    lengths = np.hypot(segments[:, 2] - segments[:, 0], segments[:, 3] - segments[:, 1])
    minimum_separation = config.line_minimum_separation_fraction * image_long_side
    parallel_tolerance = config.line_parallel_tolerance_degrees
    aspect_low, aspect_high = config.minimum_aspect_ratio, config.maximum_aspect_ratio

    angle_difference = np.abs(angles[:, None] - angles[None, :])
    angle_difference = np.minimum(angle_difference, 180.0 - angle_difference)
    first_indices, second_indices = np.nonzero(np.triu(angle_difference < parallel_tolerance, k=1))
    pair_order = np.argsort(-(lengths[first_indices] + lengths[second_indices]))[: config.line_maximum_pairs]
    first_indices, second_indices = first_indices[pair_order], second_indices[pair_order]
    perpendicular_mask = np.abs(angle_difference - 90.0) < config.line_perpendicular_tolerance_degrees

    hypotheses = []
    for first_index, second_index in zip(first_indices, second_indices):
        direction = lines[first_index, :2]
        normal = np.array([-direction[1], direction[0]])
        origin = segments[first_index, :2]
        second_offsets = (segments[second_index].reshape(2, 2) - origin) @ normal
        separation = float(abs(second_offsets.mean()))
        if separation < minimum_separation:
            continue
        first_interval = tuple(sorted(((segments[first_index].reshape(2, 2) - origin) @ direction).tolist()))
        second_interval = tuple(sorted(((segments[second_index].reshape(2, 2) - origin) @ direction).tolist()))
        if interval_overlap(first_interval, second_interval) < MINIMUM_PAIR_OVERLAP_FRACTION * min(
            lengths[first_index], lengths[second_index]
        ):
            continue

        # candidate end positions: the pair's extent ends, plus perpendicular caps between the lines
        end_positions = [(min(first_interval[0], second_interval[0]), None), (max(first_interval[1], second_interval[1]), None)]
        cap_indices = np.flatnonzero(perpendicular_mask[first_index] & perpendicular_mask[second_index])
        cap_indices = cap_indices[np.argsort(-lengths[cap_indices])]
        accepted_cap_count = 0
        for cap_index in cap_indices:
            if accepted_cap_count >= MAXIMUM_CAPS_PER_PAIR:
                break
            cap_points = segments[cap_index].reshape(2, 2) - origin
            cap_normal_extent = sorted((cap_points @ normal).tolist())
            pair_normal_extent = sorted([0.0, float(second_offsets.mean())])
            if interval_overlap(tuple(cap_normal_extent), tuple(pair_normal_extent)) < MINIMUM_CAP_COVERAGE_FRACTION * separation:
                continue
            end_positions.append((float((cap_points @ direction).mean()), cap_index))
            accepted_cap_count += 1
        end_positions.sort(key=lambda item: item[0])
        deduplicated_positions = []
        for position, cap_index in end_positions:
            if deduplicated_positions and position - deduplicated_positions[-1][0] < POSITION_DEDUPLICATION_PIXELS:
                if cap_index is not None:
                    deduplicated_positions[-1] = (position, cap_index)  # prefer a real cap line
                continue
            deduplicated_positions.append((position, cap_index))

        for start_index in range(len(deduplicated_positions)):
            for end_index in range(start_index + 1, len(deduplicated_positions)):
                start_position, start_cap = deduplicated_positions[start_index]
                end_position, end_cap = deduplicated_positions[end_index]
                span = end_position - start_position
                aspect = max(span, separation) / max(min(span, separation), 1e-6)
                if not aspect_low <= aspect <= aspect_high:
                    continue
                first_coverage = interval_overlap(first_interval, (start_position, end_position))
                second_coverage = interval_overlap(second_interval, (start_position, end_position))
                if min(first_coverage, second_coverage) < MINIMUM_SPAN_COVERAGE_FRACTION * span:
                    continue
                support = (first_coverage + second_coverage) / (2.0 * span) + CAP_SUPPORT_WEIGHT * (
                    (start_cap is not None) + (end_cap is not None)
                ) / 2.0
                end_lines = []
                for position, cap_index in ((start_position, start_cap), (end_position, end_cap)):
                    if cap_index is not None:
                        end_lines.append(lines[cap_index])
                    else:
                        end_lines.append(np.concatenate([normal, origin + position * direction]))
                corners = [
                    intersect_lines(lines[first_index], end_lines[0]),
                    intersect_lines(lines[first_index], end_lines[1]),
                    intersect_lines(lines[second_index], end_lines[1]),
                    intersect_lines(lines[second_index], end_lines[0]),
                ]
                if any(corner is None for corner in corners):
                    continue
                hypotheses.append((support, order_corners_clockwise(np.array(corners))))
    hypotheses.sort(key=lambda item: item[0], reverse=True)
    return deduplicate_hypotheses([corners for _, corners in hypotheses])[: config.line_maximum_hypotheses]


def deduplicate_hypotheses(hypotheses: list[np.ndarray]) -> list[np.ndarray]:
    """Drop hypotheses whose corners all round to the same grid cell as an earlier one."""
    seen_keys = set()
    unique_hypotheses = []
    for corners in hypotheses:
        key = tuple(np.rint(corners / HYPOTHESIS_DEDUPLICATION_GRID_PIXELS).astype(int).ravel().tolist())
        if key not in seen_keys:
            seen_keys.add(key)
            unique_hypotheses.append(corners)
    return unique_hypotheses
