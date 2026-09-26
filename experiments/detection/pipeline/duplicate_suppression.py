"""Drop duplicate detections of the same check (greedy quad-IoU suppression).

The CenterNet decoder is NMS-free by design (a check is a heatmap local maximum), but a
check that fills much of the photo can raise two peaks. On val this took single-check
precision to 61%; suppression at IoU 0.5 restores 100% with no recall loss on either val
set. It is a few lines in JS (convex polygon intersection), run on a handful of quads.
"""

from experiments.detection.pipeline.hybrid_close_up_fitting import quadrilateral_iou
from experiments.detection.predictions.detected_check import DetectedCheck

DUPLICATE_IOU_THRESHOLD = 0.5  # tuned on v1 val and close-up val


def suppress_duplicate_detections(
    detections: list[DetectedCheck], iou_threshold: float = DUPLICATE_IOU_THRESHOLD
) -> list[DetectedCheck]:
    """Keep detections in descending score order unless they overlap a kept one by >= iou_threshold."""
    kept_detections: list[DetectedCheck] = []
    for detection in sorted(detections, key=lambda candidate: -candidate.score):
        if all(quadrilateral_iou(detection.corners, kept.corners) < iou_threshold for kept in kept_detections):
            kept_detections.append(detection)
    return kept_detections
