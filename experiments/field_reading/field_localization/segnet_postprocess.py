"""Turn the net's per-field logit maps into at most one box per field, in check-crop pixels.

Per field channel: sigmoid -> bilinear upsample to canvas resolution -> threshold -> connected
components -> keep the component with the largest probability mass. The box is its pixel extent
mapped back to the 1600 px crop; confidence is the mean probability inside it. A channel whose peak
probability stays under `presence_threshold` yields no box (field absent or unreadable).
"""

import cv2
import numpy as np

from experiments.field_reading.field_localization.segnet_config import OUTPUT_STRIDE

MASK_THRESHOLD = 0.5
MIN_COMPONENT_PIXELS = 12


def sigmoid(logits: np.ndarray) -> np.ndarray:
    """Numerically safe logistic function."""
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))


def field_probability_to_box(stride_probability: np.ndarray, canvas_scale: float, presence_threshold: float,
                             crop_size: tuple[int, int]) -> tuple[list[float] | None, float]:
    """One field's (h, w) probability map at OUTPUT_STRIDE -> (box in crop pixels or None, confidence).

    Args:
        canvas_scale: canvas pixels per crop pixel (CANVAS_WIDTH_PX / crop width).
        crop_size: (width, height) of the rectified crop, used to clip.
    """
    if float(stride_probability.max()) < presence_threshold:
        return None, float(stride_probability.max())
    grid_height, grid_width = stride_probability.shape
    canvas_probability = cv2.resize(stride_probability.astype(np.float32),
                                    (grid_width * OUTPUT_STRIDE, grid_height * OUTPUT_STRIDE),
                                    interpolation=cv2.INTER_LINEAR)
    foreground = (canvas_probability >= MASK_THRESHOLD).astype(np.uint8)
    component_count, component_labels = cv2.connectedComponents(foreground, connectivity=8)
    if component_count <= 1:
        return None, float(stride_probability.max())
    probability_mass = np.bincount(component_labels.ravel(), weights=canvas_probability.ravel(),
                                   minlength=component_count)
    pixel_counts = np.bincount(component_labels.ravel(), minlength=component_count)
    probability_mass[0] = -1
    probability_mass[pixel_counts < MIN_COMPONENT_PIXELS] = -1
    best_label = int(np.argmax(probability_mass))
    if probability_mass[best_label] < 0:
        return None, float(stride_probability.max())
    rows, columns = np.nonzero(component_labels == best_label)
    crop_width, crop_height = crop_size
    box = [columns.min() / canvas_scale, rows.min() / canvas_scale,
           (columns.max() + 1) / canvas_scale, (rows.max() + 1) / canvas_scale]
    box = [min(max(box[0], 0), crop_width), min(max(box[1], 0), crop_height),
           min(max(box[2], 0), crop_width), min(max(box[3], 0), crop_height)]
    return box, float(probability_mass[best_label] / pixel_counts[best_label])


def logits_to_field_boxes(field_logits: np.ndarray, canvas_scale: float, presence_threshold: float,
                          crop_size: tuple[int, int]) -> list[tuple[list[float] | None, float]]:
    """(7, h, w) logits -> [(box | None, confidence)] in FIELD_CHANNEL_NAMES order."""
    field_probabilities = sigmoid(field_logits)
    return [field_probability_to_box(channel_probability, canvas_scale, presence_threshold, crop_size)
            for channel_probability in field_probabilities]
