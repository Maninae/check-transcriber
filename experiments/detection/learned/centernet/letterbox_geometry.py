"""Letterbox transform between a source image and the square network input.

The image is scaled so its long side equals the input size and centered on a
square canvas. The transform is a 2x3 affine matrix so augmentation can compose
further rotation/scale/translation into the same single warp, and the decoder can
invert any of them with `invert_affine_matrix`.
"""

import numpy as np


def letterbox_affine_matrix(image_width: int, image_height: int, input_size_pixels: int) -> np.ndarray:
    """2x3 affine mapping source pixels onto the centered, aspect-preserving square canvas."""
    scale = input_size_pixels / max(image_width, image_height)
    pad_x = (input_size_pixels - image_width * scale) / 2
    pad_y = (input_size_pixels - image_height * scale) / 2
    return np.array([[scale, 0.0, pad_x], [0.0, scale, pad_y]], dtype=np.float64)


def apply_affine_to_points(affine_matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Map (..., 2) points through a 2x3 affine."""
    return points @ affine_matrix[:, :2].T + affine_matrix[:, 2]


def invert_affine_matrix(affine_matrix: np.ndarray) -> np.ndarray:
    """Inverse of a 2x3 affine, as another 2x3 affine."""
    homogeneous = np.vstack([affine_matrix, [0.0, 0.0, 1.0]])
    return np.linalg.inv(homogeneous)[:2]


def compose_affine_matrices(outer_matrix: np.ndarray, inner_matrix: np.ndarray) -> np.ndarray:
    """outer(inner(x)) as one 2x3 affine."""
    outer_homogeneous = np.vstack([outer_matrix, [0.0, 0.0, 1.0]])
    inner_homogeneous = np.vstack([inner_matrix, [0.0, 0.0, 1.0]])
    return (outer_homogeneous @ inner_homogeneous)[:2]
