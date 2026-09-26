"""Turn the signal maps into binary masks and pull out check-sized connected regions.

Three mask families, each catching checks the others miss:

- smooth-paper masks (`texture_std` below a threshold, not too dark): work on busy
  fabrics, carpet and crochet, even when the fabric is as bright as the paper.
- textured masks (the complement, `texture_std` above a threshold): on a plain white
  sheet the relation inverts, the check's fine security print is the textured thing.
- paper-score Otsu mask (bright and neutral): works on wood, colored rugs and dark
  surfaces, where the background is smooth but saturated or darker.
- edge-bounded cells (connected components of NOT-edge pixels, edges from a gradient
  threshold or from Canny's hysteresis, which keeps faint borders on white bedsheets
  continuous): each check interior is a cell fenced off by its own border, which also
  separates touching checks whose seam shows as an edge even when both sides are
  equally bright paper.
- hole-filled edge maps: the Canny edge map with every enclosed hole filled (external
  contours drawn solid). A check whose outer border forms a closed loop becomes one solid
  blob no matter how many print or shadow edges cross its interior, which is where
  edge-bounded cells fragment.

A region is returned as its external contour in working-image pixels.
"""

import cv2
import numpy as np

from experiments.detection.classical.adjacent_cell_merging import (
    find_adjacent_cell_pairs,
    merged_pair_regions,
)
from experiments.detection.classical.candidate_region import CandidateRegion
from experiments.detection.classical.classical_detector_config import (
    ClassicalDetectorConfig,
    odd_kernel_size,
)
from experiments.detection.classical.working_image_channels import WorkingImageChannels

DARK_FLOOR_PERCENTILE = 25.0  # smooth regions darker than this paper-score percentile are not paper


def build_candidate_masks(channels: WorkingImageChannels, config: ClassicalDetectorConfig) -> dict[str, np.ndarray]:
    """Every binary mask (uint8 0/255) whose regions are check candidates, keyed by name."""
    masks: dict[str, np.ndarray] = {}
    paper_score = channels.paper_score
    not_dark = paper_score > np.percentile(paper_score, DARK_FLOOR_PERCENTILE)
    for texture_threshold in config.smooth_texture_std_thresholds:
        smooth_mask = (channels.texture_std < texture_threshold) & not_dark
        masks[f"smooth_std{texture_threshold:g}"] = smooth_mask.astype(np.uint8) * 255

    for texture_threshold in config.textured_std_thresholds:
        masks[f"textured_std{texture_threshold:g}"] = ((channels.texture_std > texture_threshold).astype(np.uint8)) * 255

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
        dilated_edges = cv2.dilate(canny_edges, canny_dilation_kernel)
        masks[f"canny_cells{low_threshold:g}"] = cv2.bitwise_not(dilated_edges)
        if config.use_hole_filled_edges:
            masks[f"canny_filled{low_threshold:g}"] = fill_enclosed_holes(dilated_edges)
    return masks


def fill_enclosed_holes(binary_mask: np.ndarray) -> np.ndarray:
    """The mask with every region it fully encloses filled in (external contours drawn solid)."""
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(binary_mask)
    cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)
    return filled_mask


def extract_regions_from_mask(
    binary_mask: np.ndarray,
    source_name: str,
    channels: WorkingImageChannels,
    config: ClassicalDetectorConfig,
) -> list[CandidateRegion]:
    """Open the mask to cut thin bridges, then return every check-sized component's contour.

    For edge-bounded cell masks, unions of adjacent cells are added (`adjacent_cell_merging`).
    """
    opening_size = odd_kernel_size(config.mask_opening_fraction, channels.long_side_pixels)
    opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opening_size, opening_size))
    opened_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, opening_kernel)

    minimum_area = config.minimum_area_fraction * channels.area_pixels * 0.5  # regions shrink under opening
    maximum_area = config.maximum_area_fraction * channels.area_pixels
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(opened_mask, connectivity=4)
    regions = []
    if config.merge_adjacent_cells and "cells" in source_name:
        piece_areas = stats[:, cv2.CC_STAT_AREA]
        eligible_labels = np.flatnonzero(
            (piece_areas >= config.minimum_cell_piece_fraction * channels.area_pixels) & (piece_areas <= maximum_area)
        )
        eligible_labels = eligible_labels[eligible_labels > 0]
        separation_width = opening_size + 2 * max(config.edge_dilation_pixels, config.canny_dilation_pixels) + 2
        cell_pairs = find_adjacent_cell_pairs(
            labels, eligible_labels, separation_width, int(config.minimum_shared_boundary_fraction * channels.long_side_pixels)
        )
        regions.extend(merged_pair_regions(labels, stats, cell_pairs, separation_width, maximum_area, source_name))
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
