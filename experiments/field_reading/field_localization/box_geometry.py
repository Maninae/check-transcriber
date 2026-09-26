"""Pure box arithmetic shared by priors, refinement, post-processing and metrics.

Boxes are [x0, y0, x1, y1] pixel-edge coordinates (x1 > x0 means non-empty).
"""

import numpy as np


def box_area(box: list[float]) -> float:
    """Area of a box; 0 for degenerate boxes."""
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def box_intersection_area(box_a: list[float], box_b: list[float]) -> float:
    """Area of the overlap of two boxes."""
    width = min(box_a[2], box_b[2]) - max(box_a[0], box_b[0])
    height = min(box_a[3], box_b[3]) - max(box_a[1], box_b[1])
    return max(0.0, width) * max(0.0, height)


def box_iou(box_a: list[float], box_b: list[float]) -> float:
    """Intersection over union; 0 when either box is empty."""
    intersection = box_intersection_area(box_a, box_b)
    union = box_area(box_a) + box_area(box_b) - intersection
    return intersection / union if union > 0 else 0.0


def box_coverage_of_target(predicted_box: list[float], target_box: list[float]) -> float:
    """Fraction of the target box's area inside the predicted box (1.0 = the text is fully covered)."""
    target_area = box_area(target_box)
    return box_intersection_area(predicted_box, target_box) / target_area if target_area > 0 else 0.0


def normalize_box(box: list[float], image_width: float, image_height: float) -> list[float]:
    """Pixel box -> [0, 1] fractions of the check crop."""
    return [box[0] / image_width, box[1] / image_height, box[2] / image_width, box[3] / image_height]


def denormalize_box(normalized_box: list[float], image_width: float, image_height: float) -> list[float]:
    """[0, 1] fractions -> pixel box in a crop of the given size."""
    return [normalized_box[0] * image_width, normalized_box[1] * image_height,
            normalized_box[2] * image_width, normalized_box[3] * image_height]


def clip_box(box: list[float], image_width: float, image_height: float) -> list[float]:
    """Clamp a box to the image rectangle."""
    return [float(np.clip(box[0], 0, image_width)), float(np.clip(box[1], 0, image_height)),
            float(np.clip(box[2], 0, image_width)), float(np.clip(box[3], 0, image_height))]
