"""Tune the hybrid close-up stage on a validation set (close-up val, never eval).

    CHECK_DETECTION_DATASET_ROOT=<synth/v1.1-closeup-val> python -m \
        experiments.detection.pipeline.tune_hybrid_close_up \
        --predictions <yolo raw val predictions.json> --output-dir <dir>

Two levels, so the expensive part runs once per crop setting:
1. For each crop setting (margin, relaxed interior angle), run the classical detector
   once inside every learned box covering >= the smallest frame-fraction candidate and
   cache every classical quad (in photo pixels) with its IoU to the learned box.
2. For each (frame fraction, agreement IoU) pair, assemble hybrid detections from the
   cache and score them.

Objective: share of ground-truth checks found (IoU >= 0.5) with mean corner error
< 5 px ("usable for rectification"), before refinement. The best settings are then
re-scored with refinement applied to all checks vs only to non-replaced checks.
"""

import argparse
import itertools
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon

from experiments.detection.classical.classical_detector_config import ClassicalDetectorConfig
from experiments.detection.classical.detect_checks_classical import detect_checks_classical
from experiments.detection.dataset.scene_annotations import SceneAnnotation, load_split_scene_annotations
from experiments.detection.metrics.score_predictions import score_predictions_against_split
from experiments.detection.pipeline.hybrid_close_up_fitting import HybridCloseUpConfig, quadrilateral_iou
from experiments.detection.predictions.detected_check import DetectedCheck, load_predictions_file
from experiments.detection.refinement.quadrilateral_refinement import refine_detected_checks

logger = logging.getLogger(__name__)

CROP_MARGIN_CANDIDATES = (0.08, 0.2)
RELAXED_ANGLE_CANDIDATES = (35.0, 45.0)
RELAXED_MAXIMUM_AREA_FRACTION = 0.97
FRAME_FRACTION_CANDIDATES = (0.1, 0.2, 0.3, 0.45)
AGREEMENT_IOU_CANDIDATES = (0.5, 0.6, 0.7, 0.8)
USABLE_CORNER_ERROR_PIXELS = 5.0
TOP_SETTINGS_TO_REFINE = 3


def classical_candidates_for_scene(task: tuple) -> tuple[str, list]:
    """For one scene: per learned detection, a list of (classical corners, agreement IoU)."""
    scene, detections, crop_margin, relaxed_angle, smallest_frame_fraction = task
    image_bgr = cv2.imread(str(scene.image_path), cv2.IMREAD_COLOR)
    image_height, image_width = image_bgr.shape[:2]
    classical_config = replace(
        ClassicalDetectorConfig(),
        maximum_area_fraction=RELAXED_MAXIMUM_AREA_FRACTION,
        minimum_interior_angle_degrees=relaxed_angle,
    )
    candidates_per_detection = []
    for detection in detections:
        frame_fraction = Polygon(detection.corners).buffer(0).area / (image_height * image_width)
        if frame_fraction < smallest_frame_fraction:
            candidates_per_detection.append([])
            continue
        x_min, y_min = detection.corners.min(axis=0)
        x_max, y_max = detection.corners.max(axis=0)
        margin_x, margin_y = (x_max - x_min) * crop_margin, (y_max - y_min) * crop_margin
        left, top = int(max(0, x_min - margin_x)), int(max(0, y_min - margin_y))
        right, bottom = int(min(image_width, x_max + margin_x)), int(min(image_height, y_max + margin_y))
        origin = np.array([left, top], dtype=np.float64)
        candidates_per_detection.append([
            (classical.corners + origin, quadrilateral_iou(classical.corners + origin, detection.corners))
            for classical in detect_checks_classical(image_bgr[top:bottom, left:right], classical_config)
        ])
    return scene.scene_id, candidates_per_detection


def assemble_hybrid_predictions(
    scenes: list[SceneAnnotation], raw_predictions: dict, candidate_cache: dict, frame_fraction: float, agreement_iou: float
) -> dict[str, list[DetectedCheck]]:
    """Hybrid detections per scene from cached classical candidates."""
    hybrid_predictions = {}
    for scene in scenes:
        photo_area = scene.image_width * scene.image_height
        scene_detections = []
        for detection, candidates in zip(raw_predictions[scene.scene_id], candidate_cache[scene.scene_id]):
            covers_enough = Polygon(detection.corners).buffer(0).area / photo_area >= frame_fraction
            agreeing = [candidate for candidate in candidates if candidate[1] >= agreement_iou]
            if covers_enough and agreeing:
                best_corners = max(agreeing, key=lambda candidate: candidate[1])[0]
                scene_detections.append(replace(detection, corners=best_corners, extras={"hybrid_replaced": True}))
            else:
                scene_detections.append(detection)
        hybrid_predictions[scene.scene_id] = scene_detections
    return hybrid_predictions


def usable_check_fraction(metrics: dict, ground_truth_check_count: int) -> float:
    """Share of GT checks matched at IoU 0.5 whose mean corner error is under the usable bound."""
    usable_count = sum(
        1
        for record in metrics["per_check_records"]
        if record.get("corner_error_mean_px") is not None and record["corner_error_mean_px"] < USABLE_CORNER_ERROR_PIXELS
    )
    return usable_count / ground_truth_check_count


def summarize(metrics: dict, ground_truth_check_count: int) -> dict:
    """The few numbers the sweep ranks and reports."""
    corner_error = metrics["localization"]["corner_error_mean_px"]
    return {
        "usable_fraction": usable_check_fraction(metrics, ground_truth_check_count),
        "scene_perfect_rate": metrics["scenes"]["scene_perfect_rate"],
        "precision": metrics["detection"]["0.50"]["precision"],
        "recall": metrics["detection"]["0.50"]["recall"],
        "corner_median_px": corner_error["median"],
        "corner_p90_px": corner_error["p90"],
    }


def refine_predictions(scenes: list[SceneAnnotation], predictions: dict, skip_replaced: bool) -> dict:
    """Apply corner refinement to all detections, or only to those not replaced by the hybrid."""
    refined = {}
    for scene in scenes:
        image_bgr = cv2.imread(str(scene.image_path), cv2.IMREAD_COLOR)
        detections = predictions[scene.scene_id]
        refined_all = refine_detected_checks(image_bgr, detections)
        refined[scene.scene_id] = [
            original if skip_replaced and original.extras.get("hybrid_replaced") else refined_detection
            for original, refined_detection in zip(detections, refined_all)
        ]
    return refined


def main() -> None:
    """Run the two-level sweep and write results JSON."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None, help="first N scenes (keeps CPU cost bounded)")
    parser.add_argument("--crop-margins", default=",".join(map(str, CROP_MARGIN_CANDIDATES)))
    parser.add_argument("--relaxed-angles", default=",".join(map(str, RELAXED_ANGLE_CANDIDATES)))
    parser.add_argument("--frame-fractions", default=",".join(map(str, FRAME_FRACTION_CANDIDATES)))
    parser.add_argument("--agreement-ious", default=",".join(map(str, AGREEMENT_IOU_CANDIDATES)))
    arguments = parser.parse_args()
    crop_margins = [float(value) for value in arguments.crop_margins.split(",")]
    relaxed_angles = [float(value) for value in arguments.relaxed_angles.split(",")]
    frame_fractions = [float(value) for value in arguments.frame_fractions.split(",")]
    agreement_ious = [float(value) for value in arguments.agreement_ious.split(",")]
    scenes = load_split_scene_annotations(arguments.split, limit=arguments.limit)
    ground_truth_check_count = sum(len(scene.checks) for scene in scenes)
    _, raw_predictions = load_predictions_file(arguments.predictions)
    baseline = summarize(score_predictions_against_split(raw_predictions, scenes), ground_truth_check_count)
    logger.info("raw learned boxes: %s", baseline)
    results = []
    for crop_margin, relaxed_angle in itertools.product(crop_margins, relaxed_angles):
        tasks = [(scene, raw_predictions[scene.scene_id], crop_margin, relaxed_angle, min(frame_fractions)) for scene in scenes]
        with ProcessPoolExecutor(max_workers=arguments.workers) as executor:
            candidate_cache = dict(executor.map(classical_candidates_for_scene, tasks, chunksize=4))
        for frame_fraction, agreement_iou in itertools.product(frame_fractions, agreement_ious):
            hybrid_predictions = assemble_hybrid_predictions(scenes, raw_predictions, candidate_cache, frame_fraction, agreement_iou)
            settings = HybridCloseUpConfig(
                minimum_frame_fraction=frame_fraction,
                crop_margin_fraction=crop_margin,
                minimum_agreement_iou=agreement_iou,
                relaxed_maximum_area_fraction=RELAXED_MAXIMUM_AREA_FRACTION,
                relaxed_minimum_interior_angle_degrees=relaxed_angle,
            )
            summary = summarize(score_predictions_against_split(hybrid_predictions, scenes), ground_truth_check_count)
            results.append({"settings": asdict(settings), **summary, "predictions": hybrid_predictions})
            logger.info("%s -> usable %.3f perfect %.3f", asdict(settings), summary["usable_fraction"], summary["scene_perfect_rate"])
    results.sort(key=lambda result: (-result["usable_fraction"], -result["scene_perfect_rate"]))
    refined_rows = []
    for result in results[:TOP_SETTINGS_TO_REFINE]:
        for skip_replaced in (False, True):
            refined = refine_predictions(scenes, result["predictions"], skip_replaced)
            refined_rows.append({
                "settings": result["settings"],
                "refine_replaced": not skip_replaced,
                **summarize(score_predictions_against_split(refined, scenes), ground_truth_check_count),
            })
    refined_baseline = summarize(
        score_predictions_against_split(refine_predictions(scenes, raw_predictions, False), scenes), ground_truth_check_count
    )
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "raw": baseline,
        "raw_refined": refined_baseline,
        "sweep": [{key: value for key, value in result.items() if key != "predictions"} for result in results],
        "top_refined": sorted(refined_rows, key=lambda row: -row["usable_fraction"]),
    }
    (arguments.output_dir / "hybrid_sweep.json").write_text(json.dumps(payload, indent=2))
    logger.info("raw refined: %s", refined_baseline)
    for row in payload["top_refined"]:
        logger.info("refined %s", row)


if __name__ == "__main__":
    main()
