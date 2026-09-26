"""Canny edges from lightness OR color: the edge map the Canny-based generators share.

On a white or cream bedsheet the lit sides of a check have no lightness step at all
(paper and sheet are equally bright); only the drop-shadow sides do. What survives is hue:
a beige, pink or blue check against a neutral sheet. Canny on the Lab a and b channels
(amplified by `chroma_edge_gain`, since paper tints are only a few units of chroma) finds
those sides; its union with lightness Canny gives closed check outlines.

Used by line extraction, edge-bounded Canny cells and hole-filled Canny masks.
"""

import cv2
import numpy as np

from experiments.detection.classical.classical_detector_config import ClassicalDetectorConfig
from experiments.detection.classical.preprocessing.working_image_channels import WorkingImageChannels

NEUTRAL_CHROMA_LEVEL = 128.0  # a/b are stored centered on 0; Canny needs uint8 around mid-gray


def build_combined_canny_edges(
    channels: WorkingImageChannels, low_threshold: float, high_threshold: float, config: ClassicalDetectorConfig
) -> np.ndarray:
    """uint8 0/255 Canny edges of lightness, OR'd with Canny of each amplified color channel."""
    lightness_uint8 = np.clip(channels.lightness, 0, 255).astype(np.uint8)
    edges = cv2.Canny(lightness_uint8, low_threshold, high_threshold, L2gradient=True)
    if config.use_chroma_edges:
        for color_channel in (channels.lab_a, channels.lab_b):
            amplified_uint8 = np.clip(NEUTRAL_CHROMA_LEVEL + config.chroma_edge_gain * color_channel, 0, 255).astype(np.uint8)
            edges |= cv2.Canny(amplified_uint8, low_threshold, high_threshold, L2gradient=True)
    return edges
