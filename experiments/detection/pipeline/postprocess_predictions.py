"""Apply corner refinement and/or orientation assignment to an existing predictions file.

    python -m experiments.detection.pipeline.postprocess_predictions \
        --predictions <in.json> --output <out.json> [--hybrid] [--refine] [--orient]

This decouples the stages: a detector runs once, and every combination
(raw, +hybrid, +refine, +orient, ...) is scored from the same detections. Order is
hybrid (classical fit replaces frame-filling learned boxes), then refine (corners get
exact), then orient (the classifier's crop is then aligned with the paper). `seconds_per_image` in the output header is the detector's
time plus the time spent here, excluding JPEG decode.
"""

import argparse
import dataclasses
import json
import logging
import time
from pathlib import Path

import cv2

from experiments.detection.dataset.scene_annotations import load_split_scene_annotations
from experiments.detection.pipeline.duplicate_suppression import suppress_duplicate_detections
from experiments.detection.pipeline.hybrid_close_up_fitting import HybridCloseUpConfig, apply_hybrid_close_up_fitting
from experiments.detection.orientation.assign_check_orientation import CheckOrientationAssigner
from experiments.detection.orientation.train_upside_down_classifier import CLASSIFIER_WEIGHTS_PATH
from experiments.detection.predictions.detected_check import load_predictions_file, save_predictions_file
from experiments.detection.refinement.quadrilateral_refinement import refine_detected_checks

logger = logging.getLogger(__name__)


def main() -> None:
    """Load predictions, post-process every scene, save a new predictions file."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refine", action="store_true")
    parser.add_argument("--orient", action="store_true")
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument("--suppress-duplicates", action="store_true", help="quad-IoU suppression first")
    parser.add_argument("--hybrid-config", default="{}", help="JSON overrides for HybridCloseUpConfig")
    parser.add_argument("--dataset", default=None, help="synth/<name> the predictions belong to (default: active)")
    parser.add_argument("--classifier-weights", type=Path, default=CLASSIFIER_WEIGHTS_PATH)
    arguments = parser.parse_args()
    if not (arguments.refine or arguments.orient or arguments.hybrid or arguments.suppress_duplicates):
        raise SystemExit("nothing to do: pass --hybrid, --refine and/or --orient")
    hybrid_config = HybridCloseUpConfig(**json.loads(arguments.hybrid_config))

    header, predictions_by_scene_id = load_predictions_file(arguments.predictions)
    scenes = [
        scene
        for scene in load_split_scene_annotations(header["split_name"], dataset_name=arguments.dataset)
        if scene.scene_id in predictions_by_scene_id
    ]
    orientation_assigner = CheckOrientationAssigner(arguments.classifier_weights) if arguments.orient else None
    postprocessed_by_scene_id = {}
    total_postprocess_seconds = 0.0
    for scene_number, scene in enumerate(scenes):
        image_bgr = cv2.imread(str(scene.image_path), cv2.IMREAD_COLOR)
        start_time = time.perf_counter()
        detections = predictions_by_scene_id[scene.scene_id]
        if arguments.suppress_duplicates:
            detections = suppress_duplicate_detections(detections)
        if arguments.hybrid:
            detections = apply_hybrid_close_up_fitting(image_bgr, detections, hybrid_config)
        if arguments.refine:
            detections = refine_detected_checks(image_bgr, detections)
        if orientation_assigner is not None:
            detections = orientation_assigner.orient_detected_checks(image_bgr, detections)
        total_postprocess_seconds += time.perf_counter() - start_time
        postprocessed_by_scene_id[scene.scene_id] = detections
        if scene_number % 100 == 0:
            logger.info("%d/%d scenes", scene_number, len(scenes))

    stage_suffix = ("+dedup" if arguments.suppress_duplicates else "") + ("+hybrid" if arguments.hybrid else "") + ("+refine" if arguments.refine else "") + ("+orient" if arguments.orient else "")
    detector_config = dict(header.get("config", {}))
    postprocess_seconds_per_image = total_postprocess_seconds / max(len(scenes), 1)
    detector_config["postprocess_seconds_per_image"] = postprocess_seconds_per_image
    detector_config["seconds_per_image"] = detector_config.get("seconds_per_image", 0.0) + postprocess_seconds_per_image
    detector_config["source_predictions"] = str(arguments.predictions)
    if arguments.hybrid:
        detector_config["hybrid_config"] = dataclasses.asdict(hybrid_config)
    save_predictions_file(
        arguments.output,
        detector_name=header["detector_name"] + stage_suffix,
        split_name=header["split_name"],
        predictions_by_scene_id=postprocessed_by_scene_id,
        detector_config=detector_config,
    )
    logger.info("wrote %s (%.1f ms/image post-processing)", arguments.output, 1000 * postprocess_seconds_per_image)


if __name__ == "__main__":
    main()
