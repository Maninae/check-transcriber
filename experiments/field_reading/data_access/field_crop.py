"""Cut a field crop out of a rectified check crop, given a box in check-crop pixels.

Uses the same margin rule synth v1 used for its `field_crop` PNGs (15% of box height, at least
6 px, clipped), so a reader sees the same framing whether the box came from ground truth
("oracle" localization) or from a localizer. Readers must go through this for end-to-end runs.
"""

import numpy as np

FIELD_CROP_MARGIN_FRACTION = 0.15
FIELD_CROP_MIN_MARGIN_PX = 6


def field_crop_window(box: list[float], check_width: int, check_height: int) -> list[int]:
    """Integer [x0, y0, x1, y1] window: the box plus the standard margin, clipped to the check."""
    x0, y0, x1, y1 = box
    margin = max(FIELD_CROP_MIN_MARGIN_PX, FIELD_CROP_MARGIN_FRACTION * (y1 - y0))
    return [int(max(0, np.floor(x0 - margin))), int(max(0, np.floor(y0 - margin))),
            int(min(check_width, np.ceil(x1 + margin))), int(min(check_height, np.ceil(y1 + margin)))]


def crop_field_from_check(check_image: np.ndarray, box: list[float]) -> np.ndarray:
    """The field crop (H, W[, C]) for `box` on a rectified check image array."""
    x0, y0, x1, y1 = field_crop_window(box, check_image.shape[1], check_image.shape[0])
    return check_image[y0:y1, x0:x1]
