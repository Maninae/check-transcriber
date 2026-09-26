"""Replace a frame-filling learned box with the classical detector's quad fitted inside it.

Why: when one check fills most of the photo under a steep tilt, the check is a strong
trapezoid. A rotated box cannot represent it, and the YOLO box comes out loose (training
rarely showed checks that large). The classical detector fits edges directly and is tight
there, but on its own it misses some of these checks. So the learned detector decides
WHERE and HOW MANY, and classical decides the exact corners, only for large detections.

Per learned detection whose quad covers at least `minimum_frame_fraction` of the photo:
1. Crop the quad's bounding box grown by `crop_margin_fraction`, clipped to the photo.
2. Run the classical detector on the crop with limits relaxed for a check that fills
   the crop (`maximum_area_fraction`, `minimum_interior_angle_degrees`).
3. Keep the classical quad with the highest IoU to the learned quad if that IoU is at
   least `minimum_agreement_iou`; otherwise keep the learned quad.
Replaced detections get `extras["hybrid_replaced"] = True`; orientation is reassigned
downstream, so the corner order here is only clockwise.

Tune every threshold on close-up val (v1.1-closeup-val), never on eval.
"""

from dataclasses import dataclass, replace

import numpy as np
from shapely.geometry import Polygon

from experiments.detection.classical.classical_detector_config import ClassicalDetectorConfig
from experiments.detection.classical.detect_checks_classical import detect_checks_classical
from experiments.detection.predictions.detected_check import DetectedCheck


@dataclass
class HybridCloseUpConfig:
    """Thresholds for when and how the classical fit replaces a learned quad."""

    minimum_frame_fraction: float = 0.25  # learned quad area / photo area
    crop_margin_fraction: float = 0.15  # of the quad's bounding-box size, per side
    minimum_agreement_iou: float = 0.6
    relaxed_maximum_area_fraction: float = 0.97
    relaxed_minimum_interior_angle_degrees: float = 40.0


def quadrilateral_iou(first_corners: np.ndarray, second_corners: np.ndarray) -> float:
    """IoU of two quads (invalid ones repaired with buffer(0))."""
    first_polygon, second_polygon = Polygon(first_corners).buffer(0), Polygon(second_corners).buffer(0)
    union_area = first_polygon.union(second_polygon).area
    return 0.0 if union_area == 0 else first_polygon.intersection(second_polygon).area / union_area


def fit_classical_inside_detection(
    image_bgr: np.ndarray, detection: DetectedCheck, config: HybridCloseUpConfig, classical_config: ClassicalDetectorConfig
) -> DetectedCheck | None:
    """Best agreeing classical quad inside the detection's crop, in photo pixels, or None."""
    image_height, image_width = image_bgr.shape[:2]
    x_min, y_min = detection.corners.min(axis=0)
    x_max, y_max = detection.corners.max(axis=0)
    margin_x = (x_max - x_min) * config.crop_margin_fraction
    margin_y = (y_max - y_min) * config.crop_margin_fraction
    crop_left, crop_top = int(max(0, x_min - margin_x)), int(max(0, y_min - margin_y))
    crop_right, crop_bottom = int(min(image_width, x_max + margin_x)), int(min(image_height, y_max + margin_y))
    crop_origin = np.array([crop_left, crop_top], dtype=np.float64)
    classical_detections = detect_checks_classical(image_bgr[crop_top:crop_bottom, crop_left:crop_right], classical_config)
    best_detection, best_iou = None, config.minimum_agreement_iou
    for classical_detection in classical_detections:
        corners_in_photo = classical_detection.corners + crop_origin
        agreement_iou = quadrilateral_iou(corners_in_photo, detection.corners)
        if agreement_iou >= best_iou:
            best_iou = agreement_iou
            best_detection = replace(
                detection,
                corners=corners_in_photo,
                orientation_known=False,
                extras={**detection.extras, "hybrid_replaced": True, "hybrid_agreement_iou": agreement_iou},
            )
    return best_detection


def apply_hybrid_close_up_fitting(
    image_bgr: np.ndarray, detections: list[DetectedCheck], config: HybridCloseUpConfig | None = None
) -> list[DetectedCheck]:
    """Return detections with frame-filling ones replaced by an agreeing classical fit."""
    config = config or HybridCloseUpConfig()
    classical_config = replace(
        ClassicalDetectorConfig(),
        maximum_area_fraction=config.relaxed_maximum_area_fraction,
        minimum_interior_angle_degrees=config.relaxed_minimum_interior_angle_degrees,
    )
    photo_area = image_bgr.shape[0] * image_bgr.shape[1]
    hybrid_detections = []
    for detection in detections:
        frame_fraction = Polygon(detection.corners).buffer(0).area / photo_area
        replacement = None
        if frame_fraction >= config.minimum_frame_fraction:
            replacement = fit_classical_inside_detection(image_bgr, detection, config, classical_config)
        hybrid_detections.append(replacement or detection)
    return hybrid_detections
