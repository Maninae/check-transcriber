"""Build the per-pixel signal maps the candidate generators threshold.

The working copy is the input resized so its long side is `working_long_side_pixels`.
From it we derive, all as float32 maps of the working size:

- `lightness`: Lab L after a grayscale closing that erases thin dark text strokes, so a
  check's interior reads as one flat bright area instead of text on paper.
- `chroma`: distance from neutral in Lab a/b. Paper is near-neutral; rugs, wood and
  most fabrics are not.
- `paper_score`: lightness - w * chroma, high on bright neutral paper.
- `texture_std`: local standard deviation of `lightness`. Paper is smooth; carpet,
  crochet, rugs and stone are textured, which separates white checks from white fabric.
- `lab_a`, `lab_b`: blurred Lab color channels (centered on 0), for inside-vs-outside
  color contrast when paper and background have the same lightness.
- `print_residue`: closed lightness minus raw lightness, large only on thin dark marks
  (printed text, rules, MICR, handwriting). Every check carries print; tiles, planks
  and plain sheets do not, which makes it the main false-positive discriminator.
- `gradient_magnitude`: Sobel magnitude of the lightly blurred `lightness`, the edge
  evidence used both for edge-bounded cells and for verifying a quad's sides.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from experiments.detection.classical.classical_detector_config import (
    ClassicalDetectorConfig,
    odd_kernel_size,
)

GRADIENT_PRE_BLUR_SIGMA = 1.0
COLOR_BLUR_SIGMA = 1.5


@dataclass
class WorkingImageChannels:
    """Signal maps for one working-resolution image, plus the scale back to full res."""

    working_bgr: np.ndarray
    scale_to_full_resolution: float  # multiply working coordinates by this
    lightness: np.ndarray
    chroma: np.ndarray
    lab_a: np.ndarray
    lab_b: np.ndarray
    paper_score: np.ndarray
    print_residue: np.ndarray
    texture_std: np.ndarray
    gradient_x: np.ndarray
    gradient_y: np.ndarray
    gradient_magnitude: np.ndarray

    @property
    def long_side_pixels(self) -> int:
        """Long side of the working image."""
        return int(max(self.lightness.shape[:2]))

    @property
    def area_pixels(self) -> int:
        """Pixel count of the working image."""
        return int(self.lightness.shape[0] * self.lightness.shape[1])


def resize_to_working_resolution(image_bgr: np.ndarray, working_long_side_pixels: int) -> tuple[np.ndarray, float]:
    """Downscale (never upscale) so the long side is at most the working size.

    Returns the working image and the factor that maps working coordinates to full res
    (the pixel-center offset is applied by the caller).
    """
    full_long_side = max(image_bgr.shape[:2])
    if full_long_side <= working_long_side_pixels:
        return image_bgr, 1.0
    downscale = working_long_side_pixels / full_long_side
    working_bgr = cv2.resize(image_bgr, None, fx=downscale, fy=downscale, interpolation=cv2.INTER_AREA)
    return working_bgr, full_long_side / max(working_bgr.shape[:2])


def local_standard_deviation(signal: np.ndarray, window_size: int) -> np.ndarray:
    """Box-window standard deviation (boxFilter + sqrBoxFilter, both in OpenCV.js)."""
    window = (window_size, window_size)
    local_mean = cv2.boxFilter(signal, cv2.CV_32F, window)
    local_mean_of_squares = cv2.sqrBoxFilter(signal, cv2.CV_32F, window)
    return np.sqrt(np.maximum(local_mean_of_squares - local_mean * local_mean, 0.0))


def build_working_image_channels(image_bgr: np.ndarray, config: ClassicalDetectorConfig) -> WorkingImageChannels:
    """Compute every signal map the detector uses from a full-resolution BGR image."""
    working_bgr, scale_to_full_resolution = resize_to_working_resolution(image_bgr, config.working_long_side_pixels)
    long_side = max(working_bgr.shape[:2])
    lab_image = cv2.cvtColor(working_bgr, cv2.COLOR_BGR2LAB)

    text_kernel_size = odd_kernel_size(config.text_suppression_kernel_fraction, long_side)
    text_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (text_kernel_size, text_kernel_size))
    raw_lightness = lab_image[..., 0]
    closed_lightness = cv2.morphologyEx(raw_lightness, cv2.MORPH_CLOSE, text_kernel)
    print_residue = cv2.subtract(closed_lightness, raw_lightness).astype(np.float32)
    lightness = closed_lightness.astype(np.float32)

    color_a = cv2.GaussianBlur(lab_image[..., 1].astype(np.float32) - 128.0, (0, 0), COLOR_BLUR_SIGMA)
    color_b = cv2.GaussianBlur(lab_image[..., 2].astype(np.float32) - 128.0, (0, 0), COLOR_BLUR_SIGMA)
    chroma = np.sqrt(color_a * color_a + color_b * color_b)
    paper_score = lightness - config.chroma_weight_in_paper_score * chroma

    texture_window = odd_kernel_size(config.texture_window_fraction, long_side)
    texture_std = local_standard_deviation(lightness, texture_window)

    blurred_lightness = cv2.GaussianBlur(lightness, (0, 0), GRADIENT_PRE_BLUR_SIGMA)
    gradient_x = cv2.Sobel(blurred_lightness, cv2.CV_32F, 1, 0, ksize=3) / 4.0
    gradient_y = cv2.Sobel(blurred_lightness, cv2.CV_32F, 0, 1, ksize=3) / 4.0
    gradient_magnitude = cv2.magnitude(gradient_x, gradient_y)

    return WorkingImageChannels(
        working_bgr=working_bgr,
        scale_to_full_resolution=scale_to_full_resolution,
        lightness=lightness,
        chroma=chroma,
        lab_a=color_a,
        lab_b=color_b,
        paper_score=paper_score,
        print_residue=print_residue,
        texture_std=texture_std,
        gradient_x=gradient_x,
        gradient_y=gradient_y,
        gradient_magnitude=gradient_magnitude,
    )
