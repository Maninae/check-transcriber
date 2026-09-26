"""Run a trained CenterNet checkpoint on images and emit `DetectedCheck`s in full-res pixels.

Preprocessing mirrors training: INTER_AREA downscale to the 1280 long side the
training copies used, then letterbox to the model's input size. The corners come out
in the check's own TL,TR,BR,BL order, which is clockwise in the image, so
`orientation_known=True`.

Checkpoint format (written by `train_centernet.py`): a dict with `model_state_dict`,
`training_config` (the `CenterNetTrainingConfig` as a dict), `epoch`, `selection_metric`.
"""

from dataclasses import fields
from pathlib import Path

import cv2
import numpy as np
import torch

from experiments.detection.learned.centernet.centernet_config import CenterNetTrainingConfig
from experiments.detection.learned.centernet.centernet_decoding import (
    DEFAULT_MAX_DETECTIONS,
    DEFAULT_SCORE_THRESHOLD,
    decode_checks_in_input_pixels,
    map_input_corners_to_source,
)
from experiments.detection.learned.centernet.centernet_model import CenterNetCheckDetector
from experiments.detection.learned.centernet.check_scene_dataset import normalize_image_to_tensor
from experiments.detection.learned.centernet.scene_augmentation import letterbox_scene_for_evaluation
from experiments.detection.learned.prepare_yolo_datasets import RESIZED_LONG_SIDE_PIXELS
from experiments.detection.predictions.detected_check import DetectedCheck


def load_centernet_checkpoint(weights_path: Path, device: str) -> tuple[CenterNetCheckDetector, CenterNetTrainingConfig]:
    """Rebuild the model from a checkpoint (no pretrained download), in eval mode on `device`."""
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    known_field_names = {field.name for field in fields(CenterNetTrainingConfig)}
    training_config = CenterNetTrainingConfig(
        **{key: value for key, value in checkpoint["training_config"].items() if key in known_field_names}
    )
    model = CenterNetCheckDetector(training_config.backbone_name, training_config.neck_channels, pretrained_backbone=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval(), training_config


@torch.no_grad()
def predict_maps_for_batch(model: CenterNetCheckDetector, input_batch: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """(B, 3, S, S) -> (heatmap probabilities (B, G, G), corner offsets (B, 8, G, G)) as numpy."""
    outputs = model(input_batch)
    heatmaps = torch.sigmoid(outputs["center_heatmap_logits"])[:, 0]
    return heatmaps.float().cpu().numpy(), outputs["corner_offsets"].float().cpu().numpy()


def maps_to_detected_checks(
    heatmap: np.ndarray,
    corner_offsets: np.ndarray,
    source_to_input_affine: np.ndarray,
    source_to_full_resolution_scale: float,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    max_detections: int = DEFAULT_MAX_DETECTIONS,
) -> list[DetectedCheck]:
    """Decode one image's maps and express the quads in full-resolution pixels."""
    corners_in_input, scores = decode_checks_in_input_pixels(heatmap, corner_offsets, score_threshold, max_detections)
    corners_full_resolution = map_input_corners_to_source(corners_in_input, source_to_input_affine) * source_to_full_resolution_scale
    return [
        DetectedCheck(corners=corners, score=float(score), orientation_known=True)
        for corners, score in zip(corners_full_resolution, scores)
    ]


class CenterNetCheckPredictor:
    """Wraps a CenterNet checkpoint behind the `DetectedCheck` contract (one image at a time)."""

    def __init__(self, weights_path: Path, device: str = "cpu", score_threshold: float = DEFAULT_SCORE_THRESHOLD):
        self.model, self.training_config = load_centernet_checkpoint(weights_path, device)
        self.device = device
        self.score_threshold = score_threshold
        self.input_size_pixels = self.training_config.input_size_pixels

    def detect_checks(self, image_bgr: np.ndarray) -> list[DetectedCheck]:
        """Detect every check in one full-resolution BGR image."""
        long_side = max(image_bgr.shape[:2])
        resize_factor = min(1.0, RESIZED_LONG_SIDE_PIXELS / long_side)
        if resize_factor < 1.0:
            image_bgr = cv2.resize(image_bgr, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)
        data_dict = {"source_image_rgb": cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), "check_corner_sets": []}
        data_dict = letterbox_scene_for_evaluation(data_dict, self.input_size_pixels)
        input_batch = normalize_image_to_tensor(data_dict["input_image_rgb"])[None].to(self.device)
        heatmaps, corner_offsets = predict_maps_for_batch(self.model, input_batch)
        return maps_to_detected_checks(
            heatmaps[0], corner_offsets[0], data_dict["source_to_input_affine"], 1.0 / resize_factor, self.score_threshold
        )
