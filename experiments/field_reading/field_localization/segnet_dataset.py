"""Torch dataset for the field-mask net: check crop -> (canvas image, soft per-field box masks).

- Crops are decoded at half resolution (`IMREAD_REDUCED_COLOR_2`, ~800 px wide), which is cheaper than
  a full decode and still above the 768 px canvas.
- Targets are exact fractional coverage of each output cell by the field box (soft edges), so the net
  learns sub-cell box edges. Absent / not-in-frame fields are all-zero channels.
- `render_check_to_canvas` is the single place defining crop -> canvas geometry; inference uses it too.
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from experiments.field_reading.field_localization.localization_config import BOX_TARGET_STATUSES
from experiments.field_reading.field_localization.segnet_augmentation import (apply_photometric_augmentation,
                                                                              random_content_homography,
                                                                              warp_box_to_axis_aligned)
from experiments.field_reading.field_localization.segnet_config import (CANVAS_HEIGHT_PX, CANVAS_WIDTH_PX,
                                                                        FIELD_CHANNEL_NAMES, OUTPUT_STRIDE)


def box_to_soft_mask(grid_box: list[float], grid_width: int, grid_height: int) -> np.ndarray:
    """(grid_height, grid_width) map of the fraction of each cell [i, i+1) covered by the box (grid units)."""
    cell_starts_x = np.arange(grid_width, dtype=np.float32)
    cell_starts_y = np.arange(grid_height, dtype=np.float32)
    coverage_x = np.clip(np.minimum(grid_box[2], cell_starts_x + 1) - np.maximum(grid_box[0], cell_starts_x), 0, 1)
    coverage_y = np.clip(np.minimum(grid_box[3], cell_starts_y + 1) - np.maximum(grid_box[1], cell_starts_y), 0, 1)
    return np.outer(coverage_y, coverage_x).astype(np.float32)


def render_check_to_canvas(check_crop_path: str, crop_size: tuple[int, int],
                           jitter_homography: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Decode a crop and place it on the fixed canvas; returns (canvas RGB uint8, crop->canvas homography)."""
    crop_width, crop_height = crop_size
    reduced_bgr = cv2.imread(check_crop_path, cv2.IMREAD_REDUCED_COLOR_2)
    reduced_height, reduced_width = reduced_bgr.shape[:2]
    crop_to_canvas = np.diag([CANVAS_WIDTH_PX / crop_width, CANVAS_WIDTH_PX / crop_width, 1.0])
    if jitter_homography is not None:
        crop_to_canvas = jitter_homography @ crop_to_canvas
    reduced_to_crop = np.diag([crop_width / reduced_width, crop_height / reduced_height, 1.0])
    canvas_bgr = cv2.warpPerspective(reduced_bgr, crop_to_canvas @ reduced_to_crop, (CANVAS_WIDTH_PX, CANVAS_HEIGHT_PX),
                                     flags=cv2.INTER_AREA if jitter_homography is None else cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB), crop_to_canvas


class FieldMaskDataset(Dataset):
    """Samples are data dicts: canvas_image (3, H, W) float [0, 1], field_masks (7, H/s, W/s), check_index_in_split."""

    def __init__(self, check_records: list[dict], augment: bool, base_seed: int = 0):
        self.check_records = check_records
        self.augment = augment
        self.base_seed = base_seed
        self.epoch_index = 0

    def __len__(self) -> int:
        return len(self.check_records)

    def __getitem__(self, sample_index: int) -> dict:
        check_record = self.check_records[sample_index]
        random_generator = np.random.default_rng((self.base_seed, self.epoch_index, sample_index))
        crop_width, crop_height = check_record["check_crop_size"]
        jitter_homography = None
        if self.augment:
            content_height = crop_height * CANVAS_WIDTH_PX / crop_width
            jitter_homography = random_content_homography(CANVAS_WIDTH_PX, content_height, CANVAS_WIDTH_PX,
                                                          random_generator)
        canvas_rgb, crop_to_canvas = render_check_to_canvas(check_record["check_crop_path"], (crop_width, crop_height),
                                                            jitter_homography)
        if self.augment:
            canvas_rgb = apply_photometric_augmentation(canvas_rgb, random_generator)
        grid_width, grid_height = CANVAS_WIDTH_PX // OUTPUT_STRIDE, CANVAS_HEIGHT_PX // OUTPUT_STRIDE
        field_masks = np.zeros((len(FIELD_CHANNEL_NAMES), grid_height, grid_width), np.float32)
        for channel_index, field_name in enumerate(FIELD_CHANNEL_NAMES):
            field_target = check_record["fields"].get(field_name)
            if field_target is None or field_target["box"] is None or field_target["status"] not in BOX_TARGET_STATUSES:
                continue
            canvas_box = warp_box_to_axis_aligned(field_target["box"], crop_to_canvas)
            field_masks[channel_index] = box_to_soft_mask([value / OUTPUT_STRIDE for value in canvas_box],
                                                          grid_width, grid_height)
        return {"canvas_image": torch.from_numpy(canvas_rgb).permute(2, 0, 1).float() / 255.0,
                "field_masks": torch.from_numpy(field_masks),
                "check_index_in_split": sample_index}
