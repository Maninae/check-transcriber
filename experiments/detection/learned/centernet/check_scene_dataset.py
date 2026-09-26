"""PyTorch dataset of v1 check scenes, read from the 1280-long-side copies on vega.

Each item: load the downscaled JPEG, scale the full-resolution GT corners by the
width ratio, augment (train) or letterbox (eval), encode CenterNet targets, and
return a data_dict of tensors. Decoding the 3-6 MP originals every epoch is too slow,
so the copies from `prepare_yolo_datasets.py` are the training source.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from experiments.detection.dataset.scene_annotations import SceneAnnotation
from experiments.detection.learned.centernet.centernet_config import (
    IMAGENET_MEAN_RGB,
    IMAGENET_STD_RGB,
)
from experiments.detection.learned.centernet.check_center_targets import encode_check_targets
from experiments.detection.learned.prepare_yolo_datasets import downscaled_copy_root
from experiments.detection.learned.centernet.scene_augmentation import (
    augment_scene_for_training,
    letterbox_scene_for_evaluation,
)

TARGET_KEYS = ("center_heatmap_target", "corner_offset_target", "corner_offset_weight", "corner_offset_normalizer")


def downscaled_image_path(scene: SceneAnnotation, images_root: Path | None = None) -> Path:
    """Path of the 1280-long-side copy of a scene image (per dataset unless `images_root` is given)."""
    if images_root is None:
        images_root = downscaled_copy_root(scene.dataset_name) / "obb" / "images"
    return images_root / scene.split_name / f"{scene.scene_id}.jpg"


def load_downscaled_scene(scene: SceneAnnotation, images_root: Path | None = None) -> dict:
    """data_dict with `source_image_rgb`, `check_corner_sets` (in that image's pixels) and the scale."""
    image_bgr = cv2.imread(str(downscaled_image_path(scene, images_root)), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"missing downscaled copy for {scene.scene_id} under {images_root}")
    full_resolution_to_copy_scale = image_bgr.shape[1] / scene.image_width
    return {
        "scene_id": scene.scene_id,
        "source_image_rgb": cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
        "check_corner_sets": [check.corners * full_resolution_to_copy_scale for check in scene.checks],
        "full_resolution_to_source_scale": full_resolution_to_copy_scale,
    }


def normalize_image_to_tensor(image_rgb: np.ndarray) -> torch.Tensor:
    """(S, S, 3) uint8 RGB -> (3, S, S) float tensor with ImageNet normalization."""
    image = image_rgb.astype(np.float32) / 255.0
    image = (image - np.array(IMAGENET_MEAN_RGB, dtype=np.float32)) / np.array(IMAGENET_STD_RGB, dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))


class CheckSceneCenterNetDataset(Dataset):
    """v1 scenes -> {`input_image`, CenterNet target maps, `scene_id`}."""

    def __init__(self, scenes: list[SceneAnnotation], input_size_pixels: int, augment: bool, seed: int = 0):
        self.scenes = scenes
        self.input_size_pixels = input_size_pixels
        self.augment = augment
        self.seed = seed
        self.epoch = 0  # bumped by the trainer so augmentation differs per epoch yet stays reproducible

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, index: int) -> dict:
        data_dict = load_downscaled_scene(self.scenes[index])
        if self.augment:
            rng = np.random.default_rng((self.seed, self.epoch, index))
            data_dict = augment_scene_for_training(data_dict, self.input_size_pixels, rng)
        else:
            data_dict = letterbox_scene_for_evaluation(data_dict, self.input_size_pixels)
        targets = encode_check_targets(data_dict["input_check_corner_sets"], self.input_size_pixels)
        item = {"input_image": normalize_image_to_tensor(data_dict["input_image_rgb"]), "scene_id": data_dict["scene_id"]}
        item.update({key: torch.from_numpy(targets[key]) for key in TARGET_KEYS})
        return item
