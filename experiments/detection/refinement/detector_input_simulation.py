"""Simulate approximate detector quads from ground-truth corners, and score corner error.

Perturbations (each keeps the GT corner order, so corner i of the output pairs with GT i):

- `obb`: the minimum-area rotated rectangle of the GT corners, exactly what an oriented
  box detector can at best output (no keystone, no curl).
- `jitter`: every corner moved by a uniform random distance in [0, N] px, random direction.
- `scale`: the quad scaled about its centroid by 1 + u, u uniform in [-S, S].
"""

import cv2
import numpy as np

PERTURBATION_KINDS = ("obb", "jitter", "scale")


def align_corner_order_to_reference(corners: np.ndarray, reference_corners: np.ndarray) -> np.ndarray:
    """Cyclically shift `corners` (same winding as reference forced) to best match it."""
    candidates = [corners, corners[::-1]]
    best_corners, best_total = corners, np.inf
    for candidate in candidates:
        for shift in range(4):
            shifted = np.roll(candidate, shift, axis=0)
            total = float(np.linalg.norm(shifted - reference_corners, axis=1).sum())
            if total < best_total:
                best_corners, best_total = shifted, total
    return best_corners


def simulate_oriented_box_corners(ground_truth_corners: np.ndarray) -> np.ndarray:
    """Minimum-area rotated rectangle of the GT corners, in GT corner order."""
    box_corners = cv2.boxPoints(cv2.minAreaRect(ground_truth_corners.astype(np.float32))).astype(np.float64)
    return align_corner_order_to_reference(box_corners, ground_truth_corners)


def simulate_jittered_corners(ground_truth_corners: np.ndarray, maximum_jitter_pixels: float, random_generator) -> np.ndarray:
    """Move each corner by a uniform [0, N] px distance in a uniform random direction."""
    distances = random_generator.uniform(0, maximum_jitter_pixels, size=4)
    angles = random_generator.uniform(0, 2 * np.pi, size=4)
    return ground_truth_corners + np.stack([np.cos(angles), np.sin(angles)], axis=1) * distances[:, None]


def simulate_scaled_corners(ground_truth_corners: np.ndarray, maximum_scale_fraction: float, random_generator) -> np.ndarray:
    """Scale the quad about its centroid by 1 + u, u uniform in [-S, S]."""
    centroid = ground_truth_corners.mean(axis=0)
    scale_factor = 1 + random_generator.uniform(-maximum_scale_fraction, maximum_scale_fraction)
    return centroid + (ground_truth_corners - centroid) * scale_factor


def simulate_detector_corners(
    ground_truth_corners: np.ndarray, perturbation_kind: str, perturbation_amount: float, random_generator
) -> np.ndarray:
    """Dispatch to one perturbation; `perturbation_amount` is N px (jitter) or S (scale)."""
    if perturbation_kind == "obb":
        return simulate_oriented_box_corners(ground_truth_corners)
    if perturbation_kind == "jitter":
        return simulate_jittered_corners(ground_truth_corners, perturbation_amount, random_generator)
    if perturbation_kind == "scale":
        return simulate_scaled_corners(ground_truth_corners, perturbation_amount, random_generator)
    raise ValueError(f"unknown perturbation {perturbation_kind!r}; expected one of {PERTURBATION_KINDS}")


def compute_corner_errors(predicted_corners: np.ndarray, ground_truth_corners: np.ndarray) -> np.ndarray:
    """(4,) Euclidean error of each corner against its GT counterpart (same order)."""
    return np.linalg.norm(predicted_corners - ground_truth_corners, axis=1)
