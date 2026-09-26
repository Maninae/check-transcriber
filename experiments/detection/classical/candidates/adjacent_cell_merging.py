"""Offer unions of neighbouring edge-bounded cells as extra candidate regions.

An edge-bounded cell is fenced by every edge, not only a check's own border, so one
check can be cut into two cells by an edge that crosses it: a hand-shadow boundary, a
fold crease, a printed color band across a money order. Neither half is check-shaped,
but their union is. We find cell pairs that share a long boundary and emit each union
(with the separating edge filled by a closing) as a candidate; the usual fitting and
verification decide whether it is a check.

Adjacency without per-pair loops: a max filter and a min filter of the label image over a
window slightly wider than the edge band give, at each edge pixel, the largest and
smallest neighbouring labels. Where they differ, that pixel sits between those two cells;
counting (min, max) pairs measures the shared boundary length.
"""

import cv2
import numpy as np

from experiments.detection.classical.candidates.candidate_region import CandidateRegion

NO_LABEL_SENTINEL = np.int32(2**30)


def find_adjacent_cell_pairs(
    labels: np.ndarray, eligible_labels: np.ndarray, window_size: int, minimum_shared_pixels: int
) -> list[tuple[int, int]]:
    """Pairs of eligible labels whose cells touch across an edge band of `window_size`."""
    eligible_lookup = np.zeros(int(labels.max()) + 1, dtype=bool)
    eligible_lookup[eligible_labels] = True
    eligible_label_image = np.where(eligible_lookup[labels], labels, 0).astype(np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    largest_neighbour = cv2.dilate(eligible_label_image, kernel).astype(np.int64)
    smallest_source = np.where(eligible_label_image > 0, eligible_label_image, float(NO_LABEL_SENTINEL))
    smallest_neighbour = cv2.erode(smallest_source, kernel).astype(np.int64)
    between_two_cells = (largest_neighbour > 0) & (smallest_neighbour < NO_LABEL_SENTINEL) & (
        smallest_neighbour != largest_neighbour
    )
    if not between_two_cells.any():
        return []
    pair_codes = smallest_neighbour[between_two_cells] * (int(labels.max()) + 1) + largest_neighbour[between_two_cells]
    unique_codes, code_counts = np.unique(pair_codes, return_counts=True)
    label_modulus = int(labels.max()) + 1
    return [
        (int(code // label_modulus), int(code % label_modulus))
        for code, count in zip(unique_codes, code_counts)
        if count >= minimum_shared_pixels
    ]


def merged_pair_regions(
    labels: np.ndarray,
    stats: np.ndarray,
    cell_pairs: list[tuple[int, int]],
    gap_closing_size: int,
    maximum_area: float,
    source_name: str,
) -> list[CandidateRegion]:
    """One candidate region per pair: the two cells plus the edge band between them."""
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (gap_closing_size, gap_closing_size))
    regions = []
    for first_label, second_label in cell_pairs:
        combined_area = float(stats[first_label, cv2.CC_STAT_AREA] + stats[second_label, cv2.CC_STAT_AREA])
        if combined_area > maximum_area:
            continue
        left = int(min(stats[first_label, 0], stats[second_label, 0]))
        top = int(min(stats[first_label, 1], stats[second_label, 1]))
        right = int(max(stats[first_label, 0] + stats[first_label, 2], stats[second_label, 0] + stats[second_label, 2]))
        bottom = int(max(stats[first_label, 1] + stats[first_label, 3], stats[second_label, 1] + stats[second_label, 3]))
        label_crop = labels[top:bottom, left:right]
        union_mask = ((label_crop == first_label) | (label_crop == second_label)).astype(np.uint8)
        union_mask = cv2.morphologyEx(
            cv2.copyMakeBorder(union_mask, gap_closing_size, gap_closing_size, gap_closing_size, gap_closing_size, cv2.BORDER_CONSTANT, value=0),
            cv2.MORPH_CLOSE,
            closing_kernel,
        )
        contours, _ = cv2.findContours(union_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        largest_contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
        largest_contour = largest_contour + np.array([left - gap_closing_size, top - gap_closing_size])
        regions.append(CandidateRegion(largest_contour.astype(np.int32), combined_area, f"{source_name}_pair"))
    return regions
