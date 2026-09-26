"""Run a trained Ultralytics YOLO model (OBB or 4-corner pose) and emit `DetectedCheck`s.

The image is downscaled with INTER_AREA to the same long side the training copies
used (`RESIZED_LONG_SIDE_PIXELS`), so inference sees the training distribution; the
model then letterboxes to its own `imgsz`. Output corners are scaled back to the
full-resolution image.

- OBB: four rectangle corners, reordered clockwise; `orientation_known=False`
  (a rotated box has no notion of the check's top).
- Pose: the four keypoints ARE the check's TL, TR, BR, BL, so `orientation_known=True`;
  per-corner confidences are kept in `extras["corner_confidences"]`.
"""

import numpy as np
import cv2
from ultralytics import YOLO

from experiments.detection.learned.prepare_yolo_datasets import RESIZED_LONG_SIDE_PIXELS
from experiments.detection.predictions.detected_check import DetectedCheck

DEFAULT_CONFIDENCE_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.5


def order_corners_clockwise(corners: np.ndarray) -> np.ndarray:
    """Sort 4 points clockwise in image coordinates (y down), starting nearest the top-left."""
    center = corners.mean(axis=0)
    angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
    clockwise_corners = corners[np.argsort(angles)]  # increasing angle = clockwise when y points down
    start_index = int(np.argmin(clockwise_corners.sum(axis=1)))
    return np.roll(clockwise_corners, -start_index, axis=0)


def resize_to_training_scale(image_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """Downscale to the training long side; returns (resized image, full-res / resized scale)."""
    long_side = max(image_bgr.shape[:2])
    if long_side <= RESIZED_LONG_SIDE_PIXELS:
        return image_bgr, 1.0
    resize_factor = RESIZED_LONG_SIDE_PIXELS / long_side
    resized_bgr = cv2.resize(image_bgr, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)
    return resized_bgr, 1.0 / resize_factor


class YoloCheckDetector:
    """Wraps a trained YOLO OBB or pose checkpoint behind the `DetectedCheck` contract."""

    def __init__(
        self,
        weights_path: str,
        image_size: int,
        device: str = "cpu",
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    ):
        self.model = YOLO(weights_path)
        self.task_name = self.model.task  # "obb" or "pose"
        self.image_size = image_size
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

    def detect_checks(self, image_bgr: np.ndarray) -> list[DetectedCheck]:
        """Detect every check in one full-resolution BGR image."""
        resized_bgr, scale_to_full_resolution = resize_to_training_scale(image_bgr)
        result = self.model.predict(
            resized_bgr,
            imgsz=self.image_size,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )[0]
        if self.task_name == "obb":
            return self.convert_obb_result(result, scale_to_full_resolution)
        if self.task_name == "pose":
            return self.convert_pose_result(result, scale_to_full_resolution)
        raise ValueError(f"unsupported YOLO task {self.task_name!r}")

    def convert_obb_result(self, result, scale_to_full_resolution: float) -> list[DetectedCheck]:
        """Rotated boxes -> clockwise quads without orientation."""
        corner_sets = result.obb.xyxyxyxy.cpu().numpy() * scale_to_full_resolution
        confidences = result.obb.conf.cpu().numpy()
        return [
            DetectedCheck(corners=order_corners_clockwise(corners.astype(np.float64)), score=float(confidence))
            for corners, confidence in zip(corner_sets, confidences)
        ]

    def convert_pose_result(self, result, scale_to_full_resolution: float) -> list[DetectedCheck]:
        """Ordered corner keypoints -> quads with orientation."""
        corner_sets = result.keypoints.xy.cpu().numpy() * scale_to_full_resolution
        corner_confidences = result.keypoints.conf.cpu().numpy()
        box_confidences = result.boxes.conf.cpu().numpy()
        return [
            DetectedCheck(
                corners=corners.astype(np.float64),
                score=float(box_confidence),
                orientation_known=True,
                extras={"corner_confidences": keypoint_confidences.tolist()},
            )
            for corners, keypoint_confidences, box_confidence in zip(
                corner_sets, corner_confidences, box_confidences
            )
        ]
