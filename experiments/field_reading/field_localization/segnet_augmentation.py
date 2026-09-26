"""Augmentations for the field-mask net: geometric jitter (as a homography) and photometric noise.

- Geometry mimics imperfect upstream detector corners: each content corner moves up to
  CORNER_JITTER_FRACTION of the canvas width, plus a small global scale and shift. No 180-degree
  flips (orientation is fixed upstream).
- Photometry mimics phone photos: brightness/contrast/gamma, blur, sensor noise, JPEG re-encode,
  occasional grayscale and ink-colour channel swaps.
"""

import cv2
import numpy as np

CORNER_JITTER_FRACTION = 0.015
GLOBAL_SCALE_RANGE = (0.94, 1.04)
GLOBAL_SHIFT_FRACTION = 0.015


def random_content_homography(content_width: float, content_height: float, canvas_width: int,
                              random_generator: np.random.Generator) -> np.ndarray:
    """3x3 homography moving the content rectangle's corners to jittered positions on the canvas."""
    content_corners = np.float32([[0, 0], [content_width, 0], [content_width, content_height], [0, content_height]])
    scale = random_generator.uniform(*GLOBAL_SCALE_RANGE)
    centre = np.float32([content_width / 2, content_height / 2])
    shift = random_generator.uniform(-GLOBAL_SHIFT_FRACTION, GLOBAL_SHIFT_FRACTION, 2) * canvas_width
    corner_jitter = random_generator.uniform(-CORNER_JITTER_FRACTION, CORNER_JITTER_FRACTION, (4, 2)) * canvas_width
    jittered_corners = (content_corners - centre) * scale + centre + shift + corner_jitter
    return cv2.getPerspectiveTransform(content_corners, jittered_corners.astype(np.float32))


def warp_box_to_axis_aligned(box: list[float], homography: np.ndarray) -> list[float]:
    """Warp a box's 4 corners and return their axis-aligned extent."""
    x0, y0, x1, y1 = box
    corners = np.float32([[[x0, y0]], [[x1, y0]], [[x1, y1]], [[x0, y1]]])
    warped = cv2.perspectiveTransform(corners, homography)[:, 0]
    return [float(warped[:, 0].min()), float(warped[:, 1].min()), float(warped[:, 0].max()), float(warped[:, 1].max())]


def apply_photometric_augmentation(canvas_rgb: np.ndarray, random_generator: np.random.Generator) -> np.ndarray:
    """Randomly perturb a uint8 RGB image's tone, sharpness, noise and compression."""
    image = canvas_rgb.astype(np.float32)
    contrast = random_generator.uniform(0.7, 1.3)
    brightness = random_generator.uniform(-35, 35)
    image = (image - 128) * contrast + 128 + brightness
    image = 255 * (np.clip(image, 0, 255) / 255) ** random_generator.uniform(0.75, 1.35)
    if random_generator.random() < 0.3:
        image = cv2.GaussianBlur(image, (0, 0), random_generator.uniform(0.4, 1.4))
    if random_generator.random() < 0.3:
        image = image + random_generator.normal(0, random_generator.uniform(2, 8), image.shape)
    image = np.clip(image, 0, 255).astype(np.uint8)
    if random_generator.random() < 0.1:
        image = np.repeat(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)[:, :, None], 3, axis=2)
    if random_generator.random() < 0.15:
        image = np.ascontiguousarray(image[:, :, random_generator.permutation(3)])
    if random_generator.random() < 0.3:
        quality = int(random_generator.integers(30, 90))
        _, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    return image
