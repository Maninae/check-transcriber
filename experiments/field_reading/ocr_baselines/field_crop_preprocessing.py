"""Image preprocessing variants applied to a field crop before Tesseract reads it.

Each step is optional and parameterized by `TesseractPreprocessingConfig`, so the config search
can toggle them independently. Every step has a direct OpenCV.js equivalent (cvtColor, resize,
threshold / adaptiveThreshold, morphologyEx, copyMakeBorder), which is what the app would run.

- Rescale: Tesseract's LSTM is most accurate with text roughly 25-40 px tall; the crop's height
  is used as a proxy for text height (the crop is the ink box plus a ~15% margin on each side).
- Ruled-line removal: payee / memo / legal-line crops include the printed rule under the ink,
  which Tesseract often reads as underscores or merges into descenders.
- White border: Tesseract segments poorly when ink touches the image edge.
"""

from dataclasses import dataclass

import cv2
import numpy as np

WHITE_BORDER_PX = 12
RULED_LINE_MIN_LENGTH_FRACTION = 0.35   # of crop width: longer horizontal runs are rule lines, not strokes
ADAPTIVE_THRESHOLD_BLOCK_FRACTION = 0.6  # of text height
ADAPTIVE_THRESHOLD_OFFSET = 12


@dataclass(frozen=True)
class TesseractPreprocessingConfig:
    """Which preprocessing steps to apply to a field crop."""

    target_crop_height_px: int | None = None   # None keeps the rectified crop's native scale
    binarization: str = "none"                 # none | otsu | adaptive
    remove_ruled_lines: bool = False


def remove_long_horizontal_lines(gray_image: np.ndarray) -> np.ndarray:
    """Paint long dark horizontal runs (rule lines, box edges) with the local paper colour."""
    ink_mask = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel_length = max(15, int(gray_image.shape[1] * RULED_LINE_MIN_LENGTH_FRACTION))
    line_mask = cv2.morphologyEx(ink_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1)))
    line_mask = cv2.dilate(line_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    paper_level = int(np.percentile(gray_image, 90))
    cleaned = gray_image.copy()
    cleaned[line_mask > 0] = paper_level
    return cleaned


def binarize(gray_image: np.ndarray, method: str) -> np.ndarray:
    """Black ink on white paper; `none` returns the grayscale unchanged."""
    if method == "none":
        return gray_image
    if method == "otsu":
        return cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if method == "adaptive":
        block_size = max(11, int(gray_image.shape[0] * ADAPTIVE_THRESHOLD_BLOCK_FRACTION) | 1)
        return cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                     block_size, ADAPTIVE_THRESHOLD_OFFSET)
    raise ValueError(f"unknown binarization {method!r}")


def preprocess_field_crop(crop_rgb: np.ndarray, config: TesseractPreprocessingConfig) -> np.ndarray:
    """Grayscale -> optional rescale -> optional rule removal -> optional binarization -> white border."""
    gray_image = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY) if crop_rgb.ndim == 3 else crop_rgb
    if config.target_crop_height_px is not None and gray_image.shape[0] > 0:
        scale = config.target_crop_height_px / gray_image.shape[0]
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        gray_image = cv2.resize(gray_image, None, fx=scale, fy=scale, interpolation=interpolation)
    if config.remove_ruled_lines:
        gray_image = remove_long_horizontal_lines(gray_image)
    gray_image = binarize(gray_image, config.binarization)
    paper_level = 255 if config.binarization != "none" else int(np.percentile(gray_image, 90))
    return cv2.copyMakeBorder(gray_image, WHITE_BORDER_PX, WHITE_BORDER_PX, WHITE_BORDER_PX, WHITE_BORDER_PX,
                              cv2.BORDER_CONSTANT, value=paper_level)
