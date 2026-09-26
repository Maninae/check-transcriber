"""Final sub-pixel corners: snap each side to the paper edge on the original image.

The working-resolution quad is accurate to a working pixel or two, which is 2-5
full-resolution pixels. We crop the quad's neighbourhood from the full-res grayscale,
erase thin dark print (grayscale closing) so printed rules and handwriting do not register
as steps, and run `edge_line_snapping` with a search radius proportional to image size.
"""

import cv2
import numpy as np

from experiments.detection.classical.classical_detector_config import ClassicalDetectorConfig
from experiments.detection.classical.geometry.edge_line_snapping import snap_quadrilateral_sides_to_edges

CROP_MARGIN_PIXELS = 40
TEXT_SUPPRESSION_KERNEL_PIXELS = 5
CROP_BLUR_SIGMA = 1.0


def refine_corners_at_full_resolution(
    full_resolution_gray: np.ndarray, corners: np.ndarray, config: ClassicalDetectorConfig
) -> np.ndarray:
    """Refined (4, 2) corners in full-res pixels; each side falls back to its input line."""
    image_height, image_width = full_resolution_gray.shape
    search_radius = max(3, int(round(config.refinement_search_fraction * max(image_width, image_height))))
    left = int(max(0, np.floor(corners[:, 0].min()) - CROP_MARGIN_PIXELS))
    top = int(max(0, np.floor(corners[:, 1].min()) - CROP_MARGIN_PIXELS))
    right = int(min(image_width, np.ceil(corners[:, 0].max()) + CROP_MARGIN_PIXELS))
    bottom = int(min(image_height, np.ceil(corners[:, 1].max()) + CROP_MARGIN_PIXELS))
    if right - left < 8 or bottom - top < 8:
        return corners
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (TEXT_SUPPRESSION_KERNEL_PIXELS, TEXT_SUPPRESSION_KERNEL_PIXELS))
    crop_intensity = cv2.morphologyEx(full_resolution_gray[top:bottom, left:right], cv2.MORPH_CLOSE, kernel)
    crop_intensity = cv2.GaussianBlur(crop_intensity, (0, 0), CROP_BLUR_SIGMA).astype(np.float32)
    crop_origin = np.array([left, top], dtype=np.float64)
    snapped_crop_corners = snap_quadrilateral_sides_to_edges(
        crop_intensity,
        corners - crop_origin,
        search_radius,
        config.refinement_samples_per_side,
        config.refinement_minimum_step_strength,
        full_image_size=(image_width, image_height),
        corner_offset=crop_origin,
    )
    return snapped_crop_corners + crop_origin
