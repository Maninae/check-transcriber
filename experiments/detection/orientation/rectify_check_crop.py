"""Perspective-warp a check quadrilateral to a small landscape crop for orientation checks.

Given corners in clockwise image order (any starting corner), the crop is taken with
the long side horizontal: if the side from corner 0 to corner 1 is the short one, the
order is rolled by one first. The result is either upright or upside down; the
orientation classifier decides which, then `apply_upside_down_decision` rolls the
corner order by two when needed so it starts at the check's own top-left.
"""

import cv2
import numpy as np

ORIENTATION_CROP_WIDTH = 224
ORIENTATION_CROP_HEIGHT = 96


def roll_corners_to_long_side_first(corners: np.ndarray) -> np.ndarray:
    """Roll the clockwise corner order so side 0->1 is a long side (top or bottom edge)."""
    first_side_length = np.linalg.norm(corners[1] - corners[0]) + np.linalg.norm(corners[2] - corners[3])
    second_side_length = np.linalg.norm(corners[2] - corners[1]) + np.linalg.norm(corners[3] - corners[0])
    if first_side_length >= second_side_length:
        return corners.copy()
    return np.roll(corners, -1, axis=0)


def warp_quadrilateral_to_crop(
    image: np.ndarray,
    corners: np.ndarray,
    crop_width: int = ORIENTATION_CROP_WIDTH,
    crop_height: int = ORIENTATION_CROP_HEIGHT,
) -> np.ndarray:
    """Warp the quad (TL, TR, BR, BL of the desired crop) to a crop_width x crop_height image."""
    destination_corners = np.array(
        [[0, 0], [crop_width - 1, 0], [crop_width - 1, crop_height - 1], [0, crop_height - 1]],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(corners.astype(np.float32), destination_corners)
    return cv2.warpPerspective(
        image, homography, (crop_width, crop_height), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE
    )


def apply_upside_down_decision(landscape_corners: np.ndarray, is_upside_down: bool) -> np.ndarray:
    """Return corners starting at the check's own top-left given the classifier's decision."""
    return np.roll(landscape_corners, -2, axis=0) if is_upside_down else landscape_corners
