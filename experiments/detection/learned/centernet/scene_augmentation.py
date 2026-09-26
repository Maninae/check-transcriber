"""Training augmentation for check scenes, operating on a data_dict of image + ordered corners.

Geometry is ONE affine warp from the source image straight to the square network
input: letterbox, then a random rotation (any angle), scale and translation about
the canvas center. Corners go through the same matrix and keep their TL,TR,BR,BL
order (the check's own corners, not image-axis ones), so a rotated check still
reports its own top-left first. No flips: a mirror would reverse the corner winding
and teach a mirrored order.

Photometric jitter (brightness/contrast/saturation/hue/gamma, blur, noise, JPEG
re-encode) runs on the already-warped input, which is cheaper than on the source.

data_dict keys:
- `source_image_rgb` (H, W, 3) uint8 and `check_corner_sets` list[(4, 2)] source pixels (in)
- `input_image_rgb` (S, S, 3) uint8, `input_check_corner_sets` list[(4, 2)] input pixels,
  `source_to_input_affine` (2, 3) (out)
"""

import cv2
import numpy as np

from experiments.detection.learned.centernet.centernet_config import LETTERBOX_FILL_VALUE
from experiments.detection.learned.centernet.letterbox_geometry import (
    apply_affine_to_points,
    compose_affine_matrices,
    letterbox_affine_matrix,
)

ANY_ANGLE_ROTATION_PROBABILITY = 0.5  # otherwise a multiple of 90 degrees plus small jitter
RIGHT_ANGLE_JITTER_DEGREES = 12.0
SCALE_RANGE = (0.65, 1.35)
TRANSLATE_FRACTION_OF_INPUT = 0.12
BRIGHTNESS_SHIFT_RANGE = 40.0
CONTRAST_RANGE = (0.7, 1.3)
SATURATION_RANGE = (0.6, 1.4)
HUE_SHIFT_RANGE_DEGREES = 8.0
GAMMA_RANGE = (0.75, 1.35)
BLUR_PROBABILITY = 0.3
BLUR_KERNEL_SIZES = (3, 5, 7)
NOISE_PROBABILITY = 0.3
NOISE_SIGMA_RANGE = (2.0, 9.0)
JPEG_PROBABILITY = 0.3
JPEG_QUALITY_RANGE = (35, 90)


def sample_training_affine(
    source_width: int, source_height: int, input_size_pixels: int, rng: np.random.Generator
) -> np.ndarray:
    """Letterbox composed with a random rotation/scale/translation about the canvas center."""
    if rng.random() < ANY_ANGLE_ROTATION_PROBABILITY:
        rotation_degrees = rng.uniform(0.0, 360.0)
    else:
        rotation_degrees = 90.0 * rng.integers(0, 4) + rng.uniform(-RIGHT_ANGLE_JITTER_DEGREES, RIGHT_ANGLE_JITTER_DEGREES)
    scale = rng.uniform(*SCALE_RANGE)
    canvas_center = (input_size_pixels / 2, input_size_pixels / 2)
    random_affine = cv2.getRotationMatrix2D(canvas_center, rotation_degrees, scale).astype(np.float64)
    random_affine[:, 2] += rng.uniform(-1, 1, size=2) * TRANSLATE_FRACTION_OF_INPUT * input_size_pixels
    letterbox = letterbox_affine_matrix(source_width, source_height, input_size_pixels)
    return compose_affine_matrices(random_affine, letterbox)


def warp_scene_to_network_input(data_dict: dict, source_to_input_affine: np.ndarray, input_size_pixels: int) -> dict:
    """Warp image and corners with one affine; fills the `input_*` keys (INTER_AREA when shrinking)."""
    source_image = data_dict["source_image_rgb"]
    scale = float(np.sqrt(abs(np.linalg.det(source_to_input_affine[:, :2]))))
    if scale < 0.75:
        # Pre-shrink with area averaging so the warp does not alias fine check print.
        prescale = min(1.0, scale * 1.5)
        source_image = cv2.resize(source_image, None, fx=prescale, fy=prescale, interpolation=cv2.INTER_AREA)
        prescale_affine = np.array([[1 / prescale, 0, 0], [0, 1 / prescale, 0]], dtype=np.float64)
        warp_affine = compose_affine_matrices(source_to_input_affine, prescale_affine)
    else:
        warp_affine = source_to_input_affine
    data_dict["input_image_rgb"] = cv2.warpAffine(
        source_image,
        warp_affine,
        (input_size_pixels, input_size_pixels),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(LETTERBOX_FILL_VALUE,) * 3,
    )
    data_dict["input_check_corner_sets"] = [
        apply_affine_to_points(source_to_input_affine, np.asarray(corners, dtype=np.float64))
        for corners in data_dict["check_corner_sets"]
    ]
    data_dict["source_to_input_affine"] = source_to_input_affine
    return data_dict


def apply_photometric_augmentation(image_rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Color, gamma, blur, sensor noise and JPEG artifacts; geometry untouched."""
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-HUE_SHIFT_RANGE_DEGREES, HUE_SHIFT_RANGE_DEGREES) / 2) % 180
    hsv[..., 1] *= rng.uniform(*SATURATION_RANGE)
    image = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    image = (image - 128.0) * rng.uniform(*CONTRAST_RANGE) + 128.0 + rng.uniform(-BRIGHTNESS_SHIFT_RANGE, BRIGHTNESS_SHIFT_RANGE)
    image = 255.0 * (np.clip(image, 0, 255) / 255.0) ** rng.uniform(*GAMMA_RANGE)
    if rng.random() < BLUR_PROBABILITY:
        kernel_size = int(rng.choice(BLUR_KERNEL_SIZES))
        image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    if rng.random() < NOISE_PROBABILITY:
        image = image + rng.normal(0.0, rng.uniform(*NOISE_SIGMA_RANGE), size=image.shape)
    image_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    if rng.random() < JPEG_PROBABILITY:
        quality = int(rng.integers(*JPEG_QUALITY_RANGE))
        _, encoded = cv2.imencode(".jpg", image_uint8, [cv2.IMWRITE_JPEG_QUALITY, quality])
        image_uint8 = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    return image_uint8


def augment_scene_for_training(data_dict: dict, input_size_pixels: int, rng: np.random.Generator) -> dict:
    """Full training augmentation: random geometric warp then photometric jitter."""
    source_height, source_width = data_dict["source_image_rgb"].shape[:2]
    affine = sample_training_affine(source_width, source_height, input_size_pixels, rng)
    data_dict = warp_scene_to_network_input(data_dict, affine, input_size_pixels)
    data_dict["input_image_rgb"] = apply_photometric_augmentation(data_dict["input_image_rgb"], rng)
    return data_dict


def letterbox_scene_for_evaluation(data_dict: dict, input_size_pixels: int) -> dict:
    """Deterministic letterbox only (validation and inference)."""
    source_height, source_width = data_dict["source_image_rgb"].shape[:2]
    affine = letterbox_affine_matrix(source_width, source_height, input_size_pixels)
    return warp_scene_to_network_input(data_dict, affine, input_size_pixels)
