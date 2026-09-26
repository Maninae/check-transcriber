"""Camera-pass invariants: no photometric op moves content (labels are never recomputed after it)."""

import numpy as np
import pytest

from scene_composer.camera.camera_effects import MOTION_HALF_LENGTH_RANGE
from scene_composer.camera.camera_pipeline import denoise_and_sharpen, motion_blur

DOT_IMAGE_SIZE = 41
DOT_CENTER = (20.0, 17.0)   # (x, y), deliberately off the image centre


def dot_image() -> np.ndarray:
    """A single bright pixel on black, RGB float32."""
    image = np.zeros((DOT_IMAGE_SIZE, DOT_IMAGE_SIZE, 3), np.float32)
    image[int(DOT_CENTER[1]), int(DOT_CENTER[0])] = 1.0
    return image


def centroid(image: np.ndarray) -> np.ndarray:
    """Intensity-weighted (x, y) centroid of channel 0."""
    weights = image[..., 0]
    y_grid, x_grid = np.mgrid[0:weights.shape[0], 0:weights.shape[1]]
    return np.array([(weights * x_grid).sum(), (weights * y_grid).sum()]) / weights.sum()


@pytest.mark.parametrize("half_length", range(*MOTION_HALF_LENGTH_RANGE))
@pytest.mark.parametrize("angle_degrees", [0.0, 17.0, 45.0, 90.0, 133.0, 179.0])
def test_motion_blur_does_not_shift_content(half_length, angle_degrees):
    blurred = motion_blur(dot_image(), half_length, angle_degrees)
    assert np.linalg.norm(centroid(blurred) - np.array(DOT_CENTER)) < 0.1


def test_isp_finish_does_not_shift_content():
    finished = denoise_and_sharpen(dot_image() * 0.5 + 0.25, chroma_sigma=1.5, luma_denoise=0.6, sharpen_sigma=1.2, sharpen_amount=0.8)
    assert np.linalg.norm(centroid(finished - 0.25) - np.array(DOT_CENTER)) < 0.1
