"""Rectify one check from a scene photo to a flat, upright crop, the way the app's stage 3 does.

The app perspective-warps each detected quadrilateral to a fixed-width landscape rectangle
(spec section 5, step 3). Here the quadrilateral is the ground-truth `corners` (the check's own
TL, TR, BR, BL), so the crop comes out upright as well; the app gets upright via its step 4.

- Height follows the physical aspect of the check size, so letters are not squashed.
- Field quads are mapped through the SAME homography as the pixels. On curled or folded paper
  the rectified text is not perfectly flat (neither is the app's), but the field's enclosing
  photo quad maps to an enclosing quad in the crop, so the box still contains the ink.
- Pixel convention: the homography works in OpenCV's (and the scene labels') pixel-centre
  coordinates, so the check's corners land on the crop's outer pixel edges (-0.5 and size - 0.5).
  `box_in_check_crop` is reported in pixel-edge coordinates (pixel i spans [i, i + 1]) like COCO.
"""

import cv2
import numpy as np

from scene_composer.geometry.perspective import apply_homography
from synthetic_checks.check_layout import CHECK_SIZE_INCHES, CheckSizeKind

RECTIFIED_CHECK_WIDTH_PX = 1600
FIELD_CROP_MARGIN_FRACTION = 0.15   # of the field box height, on every side
FIELD_CROP_MIN_MARGIN_PX = 6
PIXEL_CENTRE_TO_EDGE_OFFSET = 0.5


def rectified_check_size(size_kind: str, corners: np.ndarray) -> tuple[int, int]:
    """(width, height) of the rectified crop; unknown size kinds fall back to the photo quad's own aspect."""
    if size_kind in {kind.value for kind in CheckSizeKind}:
        width_inches, height_inches = CHECK_SIZE_INCHES[CheckSizeKind(size_kind)]
        aspect = height_inches / width_inches
    else:
        top, bottom = np.linalg.norm(corners[1] - corners[0]), np.linalg.norm(corners[2] - corners[3])
        left, right = np.linalg.norm(corners[3] - corners[0]), np.linalg.norm(corners[2] - corners[1])
        aspect = (left + right) / max(top + bottom, 1e-6)
    return RECTIFIED_CHECK_WIDTH_PX, int(round(RECTIFIED_CHECK_WIDTH_PX * aspect))


def photo_to_rectified_homography(corners: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """3x3 map from photo pixels to the rectified crop (corners -> the crop's outer edges), pixel-centre coordinates."""
    width, height = size
    target = np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float32) - PIXEL_CENTRE_TO_EDGE_OFFSET
    return cv2.getPerspectiveTransform(corners.astype(np.float32), target).astype(np.float64)


def rectify_check(photo_rgb: np.ndarray, corners: list[list[float]], size_kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Warp one check to its flat upright crop; returns (crop RGB, photo->crop homography).

    Parts of the check outside the photo come out black, as they would in the app.
    """
    corner_array = np.asarray(corners, np.float64)
    size = rectified_check_size(size_kind, corner_array)
    homography = photo_to_rectified_homography(corner_array, size)
    crop = cv2.warpPerspective(photo_rgb, homography, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return crop, homography


def field_box_in_crop(quad: list[list[float]], homography: np.ndarray, crop_size: tuple[int, int]) -> list[float] | None:
    """Axis-aligned box [x0, y0, x1, y1] (pixel edges) of a photo field quad inside the rectified crop, clipped; None if empty."""
    width, height = crop_size
    mapped = apply_homography(np.asarray(quad, np.float64), homography) + PIXEL_CENTRE_TO_EDGE_OFFSET
    x0, y0 = np.clip(mapped.min(axis=0), 0, [width, height])
    x1, y1 = np.clip(mapped.max(axis=0), 0, [width, height])
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    return [round(float(v), 1) for v in (x0, y0, x1, y1)]


def field_crop_box(box: list[float], crop_size: tuple[int, int]) -> list[int]:
    """Integer crop window around a field box: the box plus a small margin, clipped to the check crop."""
    width, height = crop_size
    x0, y0, x1, y1 = box
    margin = max(FIELD_CROP_MIN_MARGIN_PX, FIELD_CROP_MARGIN_FRACTION * (y1 - y0))
    return [int(max(0, np.floor(x0 - margin))), int(max(0, np.floor(y0 - margin))),
            int(min(width, np.ceil(x1 + margin))), int(min(height, np.ceil(y1 + margin)))]
