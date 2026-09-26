"""Turn the signal maps into binary masks and pull out check-sized connected regions.

Three mask families, each catching checks the others miss:

- smooth-paper masks (`texture_std` below a threshold, not too dark): work on busy
  fabrics, carpet and crochet, even when the fabric is as bright as the paper.
- paper-score Otsu mask (bright and neutral): works on wood, colored rugs and dark
  surfaces, where the background is smooth but saturated or darker.
- edge-bounded cells (connected components of NOT-edge pixels, edges from a gradient
  threshold or from Canny's hysteresis, which keeps faint borders on white bedsheets
  continuous): each check interior is a cell fenced off by its own border, which also
  separates touching checks whose seam shows as an edge even when both sides are
  equally bright paper.

A region is returned as its external contour in working-image pixels.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from experiments.detection.classical.classical_detector_config import (
    ClassicalDetectorConfig,
    odd_kernel_size,
)
from experiments.detection.classical.working_image_channels import WorkingImageChannels

DARK_FLOOR_PERCENTILE = 25.0  # smooth regions darker than this paper-score percentile are not paper


@dataclass
class CandidateRegion:
    """One connected region that may be a check."""

    contour: np.ndarray  # (N, 2) int32 external contour, working pixels
    region_area: float  # pixel count of the region
    source_name: str  # which mask produced it, for diagnostics and tuning


def build_candidate_masks(channels: WorkingImageChannels, config: ClassicalDetectorConfig) -> dict[str, np.ndarray]:
    """Every binary mask (uint8 0/255) whose regions are check candidates, keyed by name."""
    masks: dict[str, np.ndarray] = {}
    paper_score = channels.paper_score
    not_dark = paper_score > np.percentile(paper_score, DARK_FLOOR_PERCENTILE)
    for texture_threshold in config.smooth_texture_std_thresholds:
        smooth_mask = (channels.texture_std < texture_threshold) & not_dark
        masks[f"smooth_std{texture_threshold:g}"] = smooth_mask.astype(np.uint8) * 255

    paper_score_uint8 = np.clip(paper_score, 0, 255).astype(np.uint8)
    otsu_threshold, _ = cv2.threshold(paper_score_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    for offset in config.paper_score_otsu_offsets:
        masks[f"paper_otsu{offset:+g}"] = ((paper_score > otsu_threshold + offset).astype(np.uint8)) * 255

    dilation_size = 2 * config.edge_dilation_pixels + 1
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilation_size, dilation_size))
    for gradient_threshold in config.edge_gradient_thresholds:
        edge_mask = (channels.gradient_magnitude > gradient_threshold).astype(np.uint8) * 255
        edge_mask = cv2.dilate(edge_mask, dilation_kernel)
        masks[f"edge_cells{gradient_threshold:g}"] = cv2.bitwise_not(edge_mask)

    lightness_uint8 = np.clip(channels.lightness, 0, 255).astype(np.uint8)
    canny_dilation_size = 2 * config.canny_dilation_pixels + 1
    canny_dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (canny_dilation_size, canny_dilation_size))
    for low_threshold, high_threshold in config.canny_threshold_pairs:
        canny_edges = cv2.Canny(lightness_uint8, low_threshold, high_threshold, L2gradient=True)
        masks[f"canny_cells{low_threshold:g}"] = cv2.bitwise_not(cv2.dilate(canny_edges, canny_dilation_kernel))
    return masks


def extract_regions_from_mask(
    binary_mask: np.ndarray,
    source_name: str,
    channels: WorkingImageChannels,
    config: ClassicalDetectorConfig,
) -> list[CandidateRegion]:
    """Open the mask to cut thin bridges, then return every check-sized component's contour."""
    opening_size = odd_kernel_size(config.mask_opening_fraction, channels.long_side_pixels)
    opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opening_size, opening_size))
    opened_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, opening_kernel)

    minimum_area = config.minimum_area_fraction * channels.area_pixels * 0.5  # regions shrink under opening
    maximum_area = config.maximum_area_fraction * channels.area_pixels
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(opened_mask, connectivity=4)
    regions = []
    for component_index in range(1, component_count):
        left, top, width, height, area = stats[component_index]
        if area < minimum_area or area > maximum_area:
            continue
        component_mask = (labels[top : top + height, left : left + width] == component_index).astype(np.uint8)
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        largest_contour = max(contours, key=cv2.contourArea).reshape(-1, 2) + np.array([left, top])
        regions.append(CandidateRegion(largest_contour.astype(np.int32), float(area), source_name))
    return regions


def extract_all_candidate_regions(
    channels: WorkingImageChannels, config: ClassicalDetectorConfig
) -> list[CandidateRegion]:
    """Regions from every mask family, pooled (duplicates are resolved after fitting)."""
    regions = []
    for source_name, binary_mask in build_candidate_masks(channels, config).items():
        regions.extend(extract_regions_from_mask(binary_mask, source_name, channels, config))
    return regions
