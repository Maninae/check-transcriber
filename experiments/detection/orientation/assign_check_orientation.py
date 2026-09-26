"""Give unordered detections (classical, OBB) the check's own TL/TR/BR/BL corner order.

Two steps per detection:
1. Roll the clockwise corners so the first side is a long side (landscape).
2. Warp to a small grayscale crop and ask the upside-down classifier; roll by two if
   it says upside down.
The result has `orientation_known=True` and the classifier's probability in
`extras["upside_down_probability"]`. Detections that already know their orientation
(the pose / CenterNet models) pass through unchanged.
"""

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import torch

from experiments.detection.orientation.rectify_check_crop import (
    apply_upside_down_decision,
    roll_corners_to_long_side_first,
    warp_quadrilateral_to_crop,
)
from experiments.detection.orientation.upside_down_classifier import UpsideDownCheckClassifier
from experiments.detection.predictions.detected_check import DetectedCheck

# Crops were built from 1280-long-side images; match that scale so text size agrees.
ORIENTATION_WORKING_LONG_SIDE_PIXELS = 1280


class CheckOrientationAssigner:
    """Loads the classifier once and orients lists of detections."""

    def __init__(self, classifier_weights_path: Path, device: str = "cpu"):
        self.device = device
        self.classifier = UpsideDownCheckClassifier()
        self.classifier.load_state_dict(torch.load(classifier_weights_path, map_location=device))
        self.classifier.to(device).eval()

    def orient_detected_checks(self, image_bgr: np.ndarray, detections: list[DetectedCheck]) -> list[DetectedCheck]:
        """Return detections with corners starting at each check's own top-left."""
        unordered_indices = [index for index, detection in enumerate(detections) if not detection.orientation_known]
        if not unordered_indices:
            return detections
        working_scale = min(1.0, ORIENTATION_WORKING_LONG_SIDE_PIXELS / max(image_bgr.shape[:2]))
        working_gray = cv2.cvtColor(
            cv2.resize(image_bgr, None, fx=working_scale, fy=working_scale, interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2GRAY,
        )
        landscape_corner_sets = [roll_corners_to_long_side_first(detections[index].corners) for index in unordered_indices]
        crops = np.stack([warp_quadrilateral_to_crop(working_gray, corners * working_scale) for corners in landscape_corner_sets])
        with torch.no_grad():
            crop_batch = torch.from_numpy(crops).float().unsqueeze(1).to(self.device) / 255.0
            upside_down_probabilities = torch.sigmoid(self.classifier(crop_batch)).cpu().numpy()
        oriented_detections = list(detections)
        for index, landscape_corners, probability in zip(unordered_indices, landscape_corner_sets, upside_down_probabilities):
            oriented_detections[index] = replace(
                detections[index],
                corners=apply_upside_down_decision(landscape_corners, bool(probability > 0.5)),
                orientation_known=True,
                extras={**detections[index].extras, "upside_down_probability": float(probability)},
            )
        return oriented_detections
