"""Per-scene evaluation worker and the before/after corner-error summaries.

One record per scored check: GT attributes (deformations, overlap, out-of-frame, surface),
the approximate and refined corners in GT order, their per-corner errors, and the time
refinement took. Summaries pool all four corners of every check for the error
percentiles and use each check's WORST corner for the "% of checks < 2 / 5 px" rates,
since one bad corner ruins the perspective warp.
"""

import time
import zlib

import cv2
import numpy as np
from shapely.geometry import Polygon

from experiments.detection.dataset.scene_annotations import SceneAnnotation
from experiments.detection.predictions.detected_check import DetectedCheck
from experiments.detection.refinement.detector_input_simulation import (
    align_corner_order_to_reference,
    compute_corner_errors,
    simulate_detector_corners,
)
from experiments.detection.refinement.quadrilateral_refinement import refine_check_quadrilateral
from experiments.detection.refinement.refinement_config import CornerRefinementConfig

OVERLAPPED_VISIBLE_FRACTION = 0.98
PREDICTION_MATCH_MINIMUM_IOU = 0.5
SUCCESS_THRESHOLDS_PIXELS = (2.0, 5.0)


def compute_quad_iou(first_corners: np.ndarray, second_corners: np.ndarray) -> float:
    """Polygon IoU of two quads (0 when either is degenerate)."""
    first_polygon, second_polygon = Polygon(first_corners).buffer(0), Polygon(second_corners).buffer(0)
    union_area = first_polygon.union(second_polygon).area
    return float(first_polygon.intersection(second_polygon).area / union_area) if union_area > 0 else 0.0


def match_predictions_to_ground_truth(scene: SceneAnnotation, detections: list[DetectedCheck]) -> dict[int, DetectedCheck]:
    """Greedy one-to-one IoU matching; returns {GT check position: detection}."""
    candidate_pairs = []
    for ground_truth_position, check in enumerate(scene.checks):
        for detection_position, detection in enumerate(detections):
            iou = compute_quad_iou(check.corners, detection.corners)
            if iou >= PREDICTION_MATCH_MINIMUM_IOU:
                candidate_pairs.append((iou, ground_truth_position, detection_position))
    matches: dict[int, DetectedCheck] = {}
    used_detections: set[int] = set()
    for _, ground_truth_position, detection_position in sorted(candidate_pairs, reverse=True):
        if ground_truth_position in matches or detection_position in used_detections:
            continue
        matches[ground_truth_position] = detections[detection_position]
        used_detections.add(detection_position)
    return matches


def evaluate_scene(
    scene: SceneAnnotation,
    perturbation_kind: str,
    perturbation_amount: float,
    config: CornerRefinementConfig,
    scene_detections: list[DetectedCheck] | None,
) -> list[dict]:
    """Refine every GT check's simulated (or matched real) quad in one scene; one record each."""
    cv2.setNumThreads(1)
    image_bgr = cv2.imread(str(scene.image_path), cv2.IMREAD_COLOR)
    random_generator = np.random.default_rng(zlib.crc32(scene.scene_id.encode()))
    matched_detections = match_predictions_to_ground_truth(scene, scene_detections) if scene_detections is not None else None
    records = []
    for ground_truth_position, check in enumerate(scene.checks):
        if matched_detections is None:
            approximate_corners = simulate_detector_corners(check.corners, perturbation_kind, perturbation_amount, random_generator)
        elif ground_truth_position in matched_detections:
            approximate_corners = matched_detections[ground_truth_position].corners
        else:
            continue
        start_time = time.perf_counter()
        refined_corners, diagnostics = refine_check_quadrilateral(image_bgr, approximate_corners, config)
        elapsed_milliseconds = 1000 * (time.perf_counter() - start_time)
        # Real predictions start at an arbitrary corner: score in GT order.
        approximate_in_gt_order = align_corner_order_to_reference(approximate_corners, check.corners)
        refined_in_gt_order = align_corner_order_to_reference(refined_corners, check.corners)
        records.append(
            {
                "scene_id": scene.scene_id,
                "check_index": check.check_index,
                "background_category": scene.background_category,
                "deformation_kinds": list(check.deformation_kinds),
                "overlapped": check.visible_fraction < OVERLAPPED_VISIBLE_FRACTION,
                "out_of_frame": not check.fully_in_frame,
                "size_kind": check.size_kind,
                "ground_truth_corners": check.corners.tolist(),
                "approximate_corners": approximate_in_gt_order.tolist(),
                "refined_corners": refined_in_gt_order.tolist(),
                "errors_before": compute_corner_errors(approximate_in_gt_order, check.corners).tolist(),
                "errors_after": compute_corner_errors(refined_in_gt_order, check.corners).tolist(),
                "milliseconds": elapsed_milliseconds,
                "quad_reverted": diagnostics["quad_reverted"],
            }
        )
    return records


def summarize_corner_errors(records: list[dict], error_key: str) -> dict:
    """Pooled corner-error percentiles plus worst-corner success rates for one side of the comparison."""
    if not records:
        return {"checks": 0}
    per_check_errors = np.array([record[error_key] for record in records])
    pooled = per_check_errors.reshape(-1)
    worst_corner = per_check_errors.max(axis=1)
    summary = {
        "checks": len(records),
        "mean": float(pooled.mean()),
        "median": float(np.median(pooled)),
        "p90": float(np.percentile(pooled, 90)),
        "p95": float(np.percentile(pooled, 95)),
    }
    for threshold in SUCCESS_THRESHOLDS_PIXELS:
        summary[f"checks_below_{threshold:g}px_percent"] = float(100 * np.mean(worst_corner < threshold))
    return summary


def group_records_for_breakdown(records: list[dict]) -> dict[str, list[dict]]:
    """Named subsets: all, each deformation kind (and flat), overlap, frame, surface, input-error bucket."""
    groups: dict[str, list[dict]] = {"all": records}
    for kind in ("flat", "corner_lift", "curl", "fold", "waves"):
        groups[f"deformation={kind}"] = [
            record for record in records if (kind in record["deformation_kinds"] or (kind == "flat" and not record["deformation_kinds"]))
        ]
    groups["overlapped"] = [record for record in records if record["overlapped"]]
    groups["not_overlapped"] = [record for record in records if not record["overlapped"]]
    groups["out_of_frame"] = [record for record in records if record["out_of_frame"]]
    for category in sorted({record["background_category"] for record in records}):
        groups[f"background={category}"] = [record for record in records if record["background_category"] == category]
    worst_input = {id(record): max(record["errors_before"]) for record in records}
    groups["input_worst<2px"] = [record for record in records if worst_input[id(record)] < 2]
    groups["input_worst_2-10px"] = [record for record in records if 2 <= worst_input[id(record)] < 10]
    groups["input_worst>=10px"] = [record for record in records if worst_input[id(record)] >= 10]
    return {name: subset for name, subset in groups.items() if subset}
