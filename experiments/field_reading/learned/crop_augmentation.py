"""Photometric and geometric augmentation for line-recognizer training crops (RGB uint8 in, RGB out).

Designed around the two failure sources the app sees:
- imprecise localization: `jitter_box_and_crop` moves each box edge by up to +-10% of the box
  height inside a wide-margin context crop, then applies the app's standard crop margin rule;
- tiny, soft text in phone photos: downscale-upscale, blur, JPEG, noise, contrast and ink colour.
Each op fires independently with its own probability; all randomness comes from the passed Generator.
"""

import cv2
import numpy as np

from experiments.field_reading.data_access.field_crop import field_crop_window

BOX_JITTER_FRACTION_OF_HEIGHT = 0.10
DOWNSCALE_PROBABILITY, DOWNSCALE_RANGE = 0.45, (0.25, 0.9)
BLUR_PROBABILITY, BLUR_SIGMA_RANGE = 0.3, (0.3, 1.4)
JPEG_PROBABILITY, JPEG_QUALITY_RANGE = 0.4, (20, 90)
CONTRAST_PROBABILITY, CONTRAST_RANGE, BRIGHTNESS_RANGE = 0.7, (0.45, 1.3), (-45, 45)
GAMMA_PROBABILITY, GAMMA_RANGE = 0.3, (0.6, 1.6)
CHANNEL_SHUFFLE_PROBABILITY = 0.25
INK_TINT_PROBABILITY = 0.3
AFFINE_PROBABILITY, ROTATION_DEGREES, SHEAR_RANGE = 0.5, 2.5, 0.25
NOISE_PROBABILITY, NOISE_SIGMA_RANGE = 0.35, (2.0, 12.0)


def jitter_box_and_crop(context_image: np.ndarray, box_in_context: list[float], generator: np.random.Generator,
                        jitter_fraction: float = BOX_JITTER_FRACTION_OF_HEIGHT) -> np.ndarray:
    """Move each box edge by U(-f, f) * box height, then cut with the standard field-crop margin."""
    x0, y0, x1, y1 = box_in_context
    box_height = y1 - y0
    offsets = generator.uniform(-jitter_fraction, jitter_fraction, size=4) * box_height
    jittered = [x0 + offsets[0], y0 + offsets[1], x1 + offsets[2], y1 + offsets[3]]
    if jittered[2] - jittered[0] < 4 or jittered[3] - jittered[1] < 4:
        jittered = [x0, y0, x1, y1]
    wx0, wy0, wx1, wy1 = field_crop_window(jittered, context_image.shape[1], context_image.shape[0])
    return context_image[wy0:wy1, wx0:wx1]


def apply_ink_tint(image: np.ndarray, generator: np.random.Generator) -> np.ndarray:
    """Recolour dark (ink) pixels toward a random pen colour, keeping the paper as is."""
    grey = image.mean(axis=2, keepdims=True) / 255.0
    ink_weight = np.clip(1.0 - grey * 1.4, 0.0, 1.0)
    pen_colour = generator.uniform(0, 140, size=3).reshape(1, 1, 3)
    tinted = image * (1 - ink_weight) + pen_colour * ink_weight
    return tinted.clip(0, 255).astype(np.uint8)


def apply_affine(image: np.ndarray, generator: np.random.Generator) -> np.ndarray:
    """Small rotation + horizontal shear (italic-ish handwriting), border replicated."""
    height, width = image.shape[:2]
    angle = np.deg2rad(generator.uniform(-ROTATION_DEGREES, ROTATION_DEGREES))
    shear = generator.uniform(-SHEAR_RANGE, SHEAR_RANGE)
    linear = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]) @ np.array([[1, shear], [0, 1]])
    centre = np.array([width / 2, height / 2])
    matrix = np.hstack([linear, (centre - linear @ centre).reshape(2, 1)])
    return cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def augment_line_crop(image: np.ndarray, generator: np.random.Generator) -> np.ndarray:
    """Apply the random photometric/geometric chain to one RGB crop."""
    if generator.random() < AFFINE_PROBABILITY:
        image = apply_affine(image, generator)
    if generator.random() < CHANNEL_SHUFFLE_PROBABILITY:
        image = image[:, :, generator.permutation(3)]
    if generator.random() < INK_TINT_PROBABILITY:
        image = apply_ink_tint(image, generator)
    if generator.random() < CONTRAST_PROBABILITY:
        contrast = generator.uniform(*CONTRAST_RANGE)
        brightness = generator.uniform(*BRIGHTNESS_RANGE)
        image = np.clip((image.astype(np.float32) - 128) * contrast + 128 + brightness, 0, 255).astype(np.uint8)
    if generator.random() < GAMMA_PROBABILITY:
        gamma = generator.uniform(*GAMMA_RANGE)
        image = (255 * (image / 255.0) ** gamma).astype(np.uint8)
    if generator.random() < DOWNSCALE_PROBABILITY:
        height, width = image.shape[:2]
        scale = generator.uniform(*DOWNSCALE_RANGE)
        small = cv2.resize(image, (max(2, int(width * scale)), max(2, int(height * scale))), interpolation=cv2.INTER_AREA)
        image = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
    if generator.random() < BLUR_PROBABILITY:
        image = cv2.GaussianBlur(image, (0, 0), generator.uniform(*BLUR_SIGMA_RANGE))
    if generator.random() < NOISE_PROBABILITY:
        noise = generator.normal(0, generator.uniform(*NOISE_SIGMA_RANGE), size=image.shape)
        image = np.clip(image + noise, 0, 255).astype(np.uint8)
    if generator.random() < JPEG_PROBABILITY:
        quality = int(generator.integers(*JPEG_QUALITY_RANGE))
        encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])[1]
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return image
