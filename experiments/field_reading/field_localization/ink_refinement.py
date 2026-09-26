"""Classical refinement: snap a layout-prior region to the text ink actually present on the check.

Pipeline (all OpenCV primitives available in OpenCV.js):
1. `compute_clean_ink_mask`: adaptive threshold -> ink; morphological opening with long horizontal /
   vertical kernels finds ruled lines and box borders, which are subtracted.
2. `refine_prior_to_ink_box`: inside the prior's search region, keep connected components whose height
   is plausible for field text (drops small printed labels such as "PAY TO THE ORDER OF", filler dashes
   and specks), group them into words by horizontal dilation, pick the group nearest the prior's
   expected centre, then absorb neighbouring groups on the same text line. No ink -> no box.

- Heights scale with the prior's `median_height`, so one parameter set serves every layout.
- Printed labels of field-text size (e.g. a large "$") can still be grabbed; the x-distance penalty
  to the prior's median box is the only defence.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from experiments.field_reading.field_localization.box_geometry import clip_box, denormalize_box


@dataclass(frozen=True)
class InkRefinementParams:
    """Tunable knobs; lengths are fractions of the prior's expected text-box height unless noted."""

    adaptive_block_size_px: int = 41
    adaptive_offset: int = 14
    horizontal_line_min_length_px: int = 90
    vertical_line_min_length_px: int = 45
    search_region_padding: float = 0.5
    min_component_height: float = 0.33
    max_component_height: float = 2.2
    min_component_area_px: int = 10
    word_gap: float = 0.6
    line_merge_gap: float = 2.5
    centre_distance_sigma: float = 1.2
    horizontal_offset_sigma_fraction_of_width: float = 0.08
    output_padding: float = 0.08
    # Seed word must sit this close to the expected centre (0..1); rejects e.g. the MICR line under an empty memo.
    min_seed_proximity: float = 0.5


def compute_clean_ink_mask(grayscale_check: np.ndarray, params: InkRefinementParams) -> np.ndarray:
    """uint8 {0, 1} mask of ink pixels with long ruled lines and box borders removed."""
    ink_mask = cv2.adaptiveThreshold(grayscale_check, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
                                     params.adaptive_block_size_px, params.adaptive_offset)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (params.horizontal_line_min_length_px, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, params.vertical_line_min_length_px))
    ruled_line_mask = cv2.morphologyEx(ink_mask, cv2.MORPH_OPEN, horizontal_kernel) | \
        cv2.morphologyEx(ink_mask, cv2.MORPH_OPEN, vertical_kernel)
    ruled_line_mask = cv2.dilate(ruled_line_mask, np.ones((3, 3), np.uint8))
    return ink_mask & (1 - ruled_line_mask)


def plausible_text_components(region_ink: np.ndarray, expected_height_px: float,
                              params: InkRefinementParams) -> np.ndarray:
    """(N, 5) array of [x, y, w, h, area] for components whose size fits field text."""
    _, _, component_stats, _ = cv2.connectedComponentsWithStats(region_ink, connectivity=8)
    component_stats = component_stats[1:]
    heights = component_stats[:, cv2.CC_STAT_HEIGHT]
    keep = (heights >= params.min_component_height * expected_height_px) & \
           (heights <= params.max_component_height * expected_height_px) & \
           (component_stats[:, cv2.CC_STAT_AREA] >= params.min_component_area_px)
    return component_stats[keep]


def group_components_into_words(component_stats: np.ndarray, region_shape: tuple[int, int],
                                expected_height_px: float, params: InkRefinementParams) -> list[dict]:
    """Group components whose boxes are within `word_gap` of each other horizontally.

    Returns [{"box": [x0, y0, x1, y1] (region pixels), "ink_area": int}].
    """
    box_canvas = np.zeros(region_shape, np.uint8)
    for x, y, width, height, _ in component_stats:
        box_canvas[y:y + height, x:x + width] = 1
    gap_px = max(1, int(round(params.word_gap * expected_height_px)))
    dilated_canvas = cv2.dilate(box_canvas, cv2.getStructuringElement(cv2.MORPH_RECT, (gap_px, 1)))
    _, group_labels = cv2.connectedComponents(dilated_canvas, connectivity=8)
    word_groups: dict[int, dict] = {}
    for x, y, width, height, area in component_stats:
        label = int(group_labels[y + height // 2, x + width // 2])
        group = word_groups.setdefault(label, {"box": [x, y, x + width, y + height], "ink_area": 0})
        group["box"] = [min(group["box"][0], x), min(group["box"][1], y),
                        max(group["box"][2], x + width), max(group["box"][3], y + height)]
        group["ink_area"] += int(area)
    return list(word_groups.values())


def merge_same_line_words(seed_group: dict, word_groups: list[dict], expected_height_px: float,
                          params: InkRefinementParams) -> list[float]:
    """Grow the seed box by absorbing words that share its text line and sit within `line_merge_gap`."""
    merged_box = list(seed_group["box"])
    remaining_groups = [group for group in word_groups if group is not seed_group]
    max_gap_px = params.line_merge_gap * expected_height_px
    absorbed_any = True
    while absorbed_any:
        absorbed_any = False
        for group in list(remaining_groups):
            x0, y0, x1, y1 = group["box"]
            vertical_overlap = min(y1, merged_box[3]) - max(y0, merged_box[1])
            shares_line = vertical_overlap >= 0.5 * min(y1 - y0, merged_box[3] - merged_box[1])
            horizontal_gap = max(x0 - merged_box[2], merged_box[0] - x1, 0)
            if shares_line and horizontal_gap <= max_gap_px:
                merged_box = [min(merged_box[0], x0), min(merged_box[1], y0),
                              max(merged_box[2], x1), max(merged_box[3], y1)]
                remaining_groups.remove(group)
                absorbed_any = True
    return merged_box


def refine_prior_to_ink_box(clean_ink_mask: np.ndarray, field_prior: dict,
                            params: InkRefinementParams) -> tuple[list[float] | None, float]:
    """Snap one field prior to the best ink text line; returns (box in crop pixels or None, confidence)."""
    crop_height, crop_width = clean_ink_mask.shape
    expected_height_px = field_prior["median_height"] * crop_height
    padding_px = params.search_region_padding * expected_height_px
    region_x0, region_y0, region_x1, region_y1 = [int(round(value)) for value in clip_box(
        np.add(denormalize_box(field_prior["search_region"], crop_width, crop_height),
               [-padding_px, -padding_px, padding_px, padding_px]).tolist(), crop_width, crop_height)]
    region_ink = np.ascontiguousarray(clean_ink_mask[region_y0:region_y1, region_x0:region_x1])
    if region_ink.size == 0 or not region_ink.any():
        return None, 0.0
    component_stats = plausible_text_components(region_ink, expected_height_px, params)
    if len(component_stats) == 0:
        return None, 0.0
    word_groups = group_components_into_words(component_stats, region_ink.shape, expected_height_px, params)
    median_box_px = np.subtract(denormalize_box(field_prior["median_box"], crop_width, crop_height),
                                [region_x0, region_y0, region_x0, region_y0])
    expected_centre_y = (median_box_px[1] + median_box_px[3]) / 2
    horizontal_sigma_px = params.horizontal_offset_sigma_fraction_of_width * crop_width
    best_group, best_score, best_proximity = None, -1.0, 0.0
    for group in word_groups:
        x0, y0, x1, y1 = group["box"]
        vertical_offset = ((y0 + y1) / 2 - expected_centre_y) / (params.centre_distance_sigma * expected_height_px)
        horizontal_offset = max(median_box_px[0] - x1, x0 - median_box_px[2], 0) / horizontal_sigma_px
        proximity = float(np.exp(-0.5 * (vertical_offset ** 2 + horizontal_offset ** 2)))
        score = proximity * np.sqrt(group["ink_area"])
        if score > best_score:
            best_group, best_score, best_proximity = group, score, proximity
    if best_proximity < params.min_seed_proximity:
        return None, 0.0
    merged_box = merge_same_line_words(best_group, word_groups, expected_height_px, params)
    output_padding_px = params.output_padding * expected_height_px
    refined_box = clip_box([merged_box[0] + region_x0 - output_padding_px, merged_box[1] + region_y0 - output_padding_px,
                            merged_box[2] + region_x0 + output_padding_px, merged_box[3] + region_y0 + output_padding_px],
                           crop_width, crop_height)
    return refined_box, best_proximity
