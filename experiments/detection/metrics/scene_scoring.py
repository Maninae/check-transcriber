"""Score one scene: match predictions to GT checks and fill the per-check/per-scene records.

Flow per scene:
1. Drop predictions below the score threshold (they do not exist for any metric).
2. Clip every predicted quad, GT outline and GT corner quad to the image rectangle.
3. IoU matrix vs GT outlines; greedy matching at 0.1 (diagnostic) and at each
   reporting threshold.
4. For each GT check with a diagnostic match: IoU vs outline and vs corner quad,
   best-cyclic-shift corner error over the in-frame GT corners, orientation/axis.
"""

import numpy as np

from experiments.detection.dataset.scene_annotations import CheckAnnotation, SceneAnnotation
from experiments.detection.metrics.detection_matching import greedy_match_by_iou, pairwise_iou_matrix
from experiments.detection.metrics.metric_records import (
    DIAGNOSTIC_MATCHING_IOU_THRESHOLD,
    MATCHING_IOU_THRESHOLDS,
    OVERLAPPED_VISIBLE_FRACTION_THRESHOLD,
    PRIMARY_MATCHING_IOU_THRESHOLD,
    CheckScoreRecord,
    SceneScoreRecord,
    threshold_key,
)
from experiments.detection.metrics.quadrilateral_geometry import (
    UPSIDE_DOWN_CYCLIC_SHIFT,
    best_cyclic_corner_alignment,
    check_short_side_length,
    clipped_polygon_in_frame,
    corner_inside_frame_mask,
    polygon_iou,
    signed_area_image_coordinates,
)
from experiments.detection.predictions.detected_check import DetectedCheck

# (low, high) inclusive check-count ranges for the scene-size breakdown.
CHECK_COUNT_BUCKET_RANGES: tuple[tuple[int, int], ...] = ((1, 3), (4, 6), (7, 9), (10, 12))


def check_count_bucket_label(check_count: int) -> str:
    """Bucket label like "4-6" for a scene's GT check count ("0" / "13+" outside the ranges)."""
    for bucket_low, bucket_high in CHECK_COUNT_BUCKET_RANGES:
        if bucket_low <= check_count <= bucket_high:
            return f"{bucket_low}-{bucket_high}"
    return "0" if check_count < CHECK_COUNT_BUCKET_RANGES[0][0] else f"{CHECK_COUNT_BUCKET_RANGES[-1][1] + 1}+"


def new_check_record(scene_id: str, check: CheckAnnotation) -> CheckScoreRecord:
    """Record for a GT check with its attributes filled and no match yet."""
    return CheckScoreRecord(
        scene_id=scene_id,
        check_index=check.check_index,
        size_kind=check.size_kind,
        orientation_class=check.orientation_class,
        deformation_kinds=list(check.deformation_kinds),
        fully_in_frame=check.fully_in_frame,
        overlapped=check.visible_fraction < OVERLAPPED_VISIBLE_FRACTION_THRESHOLD,
        visible_fraction=check.visible_fraction,
    )


def fill_localization_fields(
    check_record: CheckScoreRecord,
    check: CheckAnnotation,
    prediction: DetectedCheck,
    iou_with_outline: float,
    scene: SceneAnnotation,
) -> None:
    """Fill IoU vs corner quad, corner error and orientation for one matched pair (in place)."""
    image_width, image_height = scene.image_width, scene.image_height
    check_record.iou_with_outline = iou_with_outline
    check_record.iou_with_corner_quad = polygon_iou(
        clipped_polygon_in_frame(prediction.corners, image_width, image_height),
        clipped_polygon_in_frame(check.corners, image_width, image_height),
    )
    check_record.prediction_orientation_known = prediction.orientation_known
    check_record.prediction_clockwise = signed_area_image_coordinates(prediction.corners) > 0.0
    in_frame_corner_mask = corner_inside_frame_mask(check.corners, image_width, image_height)
    check_record.corners_used_for_error = int(in_frame_corner_mask.sum())
    alignment = best_cyclic_corner_alignment(prediction.corners, check.corners, in_frame_corner_mask)
    if alignment is None:
        return
    best_shift, mean_error_px, max_error_px = alignment
    check_record.best_cyclic_shift = best_shift
    check_record.corner_error_mean_px = mean_error_px
    check_record.corner_error_max_px = max_error_px
    check_record.corner_error_mean_percent_short_side = 100.0 * mean_error_px / check_short_side_length(check.corners)
    check_record.axis_correct = best_shift in (0, UPSIDE_DOWN_CYCLIC_SHIFT)
    if prediction.orientation_known:
        check_record.orientation_correct = best_shift == 0


def score_scene(
    scene: SceneAnnotation,
    predictions: list[DetectedCheck] | None,
    score_threshold: float,
) -> tuple[SceneScoreRecord, list[CheckScoreRecord]]:
    """Score one scene's predictions; `predictions=None` means the scene was absent from the file."""
    kept_predictions = [
        prediction for prediction in (predictions or []) if prediction.score >= score_threshold
    ]
    image_width, image_height = scene.image_width, scene.image_height
    predicted_polygons = [
        clipped_polygon_in_frame(prediction.corners, image_width, image_height) for prediction in kept_predictions
    ]
    ground_truth_outline_polygons = [
        clipped_polygon_in_frame(check.outline, image_width, image_height) for check in scene.checks
    ]
    iou_matrix = pairwise_iou_matrix(predicted_polygons, ground_truth_outline_polygons)

    check_records = [new_check_record(scene.scene_id, check) for check in scene.checks]
    diagnostic_matching = greedy_match_by_iou(iou_matrix, DIAGNOSTIC_MATCHING_IOU_THRESHOLD)
    for ground_truth_index, prediction_index in diagnostic_matching.items():
        check_records[ground_truth_index].matched_prediction_index = prediction_index
        fill_localization_fields(
            check_records[ground_truth_index],
            scene.checks[ground_truth_index],
            kept_predictions[prediction_index],
            float(iou_matrix[prediction_index, ground_truth_index]),
            scene,
        )

    true_positives_by_threshold: dict[str, int] = {}
    false_positives_by_threshold: dict[str, int] = {}
    for iou_threshold in MATCHING_IOU_THRESHOLDS:
        matching = greedy_match_by_iou(iou_matrix, iou_threshold)
        for ground_truth_index in matching:
            check_records[ground_truth_index].matched_iou_threshold_keys.append(threshold_key(iou_threshold))
        true_positives_by_threshold[threshold_key(iou_threshold)] = len(matching)
        false_positives_by_threshold[threshold_key(iou_threshold)] = len(kept_predictions) - len(matching)

    primary_key = threshold_key(PRIMARY_MATCHING_IOU_THRESHOLD)
    scene_record = SceneScoreRecord(
        scene_id=scene.scene_id,
        background_id=scene.background_id,
        background_category=scene.background_category,
        background_source=scene.background_source,
        layout_mode=scene.layout_mode,
        cast_shadow_kind=scene.cast_shadow_kind,
        check_count_bucket=check_count_bucket_label(len(scene.checks)),
        ground_truth_count=len(scene.checks),
        predicted_count=len(kept_predictions),
        count_exact=len(kept_predictions) == len(scene.checks),
        true_positives_by_threshold=true_positives_by_threshold,
        false_positives_by_threshold=false_positives_by_threshold,
        scene_perfect=(
            true_positives_by_threshold[primary_key] == len(scene.checks)
            and false_positives_by_threshold[primary_key] == 0
        ),
        had_predictions_entry=predictions is not None,
    )
    return scene_record, check_records
