"""Does the inside of a candidate quad look like a check? (the false-positive gate)

Edge evidence alone accepts any four-sided cell with real borders: a floor tile boxed by
grout, a strip of wood between two planks' grain lines, the gap between two checks.
What those lack is what every check has: printed text, on smooth, bright, near-neutral
paper. Measured over the quad shrunk by 10% (so the border itself is excluded):

- print fraction: share of pixels whose print residue exceeds a threshold (text, rules,
  MICR). Real checks on val: >= 0.066 at the 1st percentile; empty tiles ~0.
- interior texture: median local std of text-suppressed lightness (paper is smooth).
- interior paper score and chroma medians (paper is bright and near-neutral).
"""

from dataclasses import dataclass

import cv2
import numpy as np

from experiments.detection.classical.classical_detector_config import ClassicalDetectorConfig
from experiments.detection.classical.working_image_channels import WorkingImageChannels

INTERIOR_SHRINK_FACTOR = 0.9
INTERIOR_SAMPLING_STRIDE = 2  # medians over every 2nd pixel in each direction (4x cheaper)


@dataclass
class InteriorAppearance:
    """Median appearance statistics inside a quad."""

    print_fraction: float
    texture_median: float
    paper_score_median: float
    chroma_median: float


def measure_interior_appearance(
    corners: np.ndarray, channels: WorkingImageChannels, config: ClassicalDetectorConfig
) -> InteriorAppearance | None:
    """Statistics over the shrunken quad interior; None when it has no in-frame pixels."""
    centroid = corners.mean(axis=0)
    shrunk_corners = centroid + (corners - centroid) * INTERIOR_SHRINK_FACTOR
    image_height, image_width = channels.lightness.shape
    left = int(max(0, np.floor(shrunk_corners[:, 0].min())))
    top = int(max(0, np.floor(shrunk_corners[:, 1].min())))
    right = int(min(image_width, np.ceil(shrunk_corners[:, 0].max()) + 1))
    bottom = int(min(image_height, np.ceil(shrunk_corners[:, 1].max()) + 1))
    if right <= left or bottom <= top:
        return None
    interior_mask = np.zeros((bottom - top, right - left), dtype=np.uint8)
    cv2.fillConvexPoly(interior_mask, np.rint(shrunk_corners - [left, top]).astype(np.int32), 1)
    stride = INTERIOR_SAMPLING_STRIDE
    inside = interior_mask[::stride, ::stride].astype(bool)
    if not inside.any():
        return None

    def interior_values(signal_map: np.ndarray) -> np.ndarray:
        return signal_map[top:bottom:stride, left:right:stride][inside]

    return InteriorAppearance(
        print_fraction=float((interior_values(channels.print_residue) > config.print_residue_threshold).mean()),
        texture_median=float(np.median(interior_values(channels.texture_std))),
        paper_score_median=float(np.median(interior_values(channels.paper_score))),
        chroma_median=float(np.median(interior_values(channels.chroma))),
    )


def interior_looks_like_check(appearance: InteriorAppearance | None, config: ClassicalDetectorConfig) -> bool:
    """All four interior statistics inside the check range."""
    if appearance is None:
        return False
    return (
        appearance.print_fraction >= config.minimum_interior_print_fraction
        and appearance.texture_median <= config.maximum_interior_texture
        and appearance.paper_score_median >= config.minimum_interior_paper_score
        and appearance.chroma_median <= config.maximum_interior_chroma
    )
