"""Apply corner refinement and/or orientation assignment to an existing predictions file.

    python -m experiments.detection.pipeline.postprocess_predictions \
        --predictions <in.json> --output <out.json> [--refine] [--orient]

This decouples the stages: a detector runs once, and every combination
(raw, +refine, +orient, +refine+orient) is scored from the same detections. Order is
refine first (corners get exact), then orient (the crop the classifier sees is then
aligned with the paper). `seconds_per_image` in the output header is the detector's
time plus the time spent here, excluding JPEG decode.
"""

import argparse
import logging
import time
from pathlib import Path

import cv2

from experiments.detection.dataset.scene_annotations import load_split_scene_annotations
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
    parser.add_argument("--classifier-weights", type=Path, default=CLASSIFIER_WEIGHTS_PATH)
    arguments = parser.parse_args()
    if not (arguments.refine or arguments.orient):
        raise SystemExit("nothing to do: pass --refine and/or --orient")

    header, predictions_by_scene_id = load_predictions_file(arguments.predictions)
    scenes = [
        scene
        for scene in load_split_scene_annotations(header["split_name"])
        if scene.scene_id in predictions_by_scene_id
    ]
    orientation_assigner = CheckOrientationAssigner(arguments.classifier_weights) if arguments.orient else None
    postprocessed_by_scene_id = {}
    total_postprocess_seconds = 0.0
    for scene_number, scene in enumerate(scenes):
        image_bgr = cv2.imread(str(scene.image_path), cv2.IMREAD_COLOR)
        start_time = time.perf_counter()
        detections = predictions_by_scene_id[scene.scene_id]
        if arguments.refine:
            detections = refine_detected_checks(image_bgr, detections)
        if orientation_assigner is not None:
            detections = orientation_assigner.orient_detected_checks(image_bgr, detections)
        total_postprocess_seconds += time.perf_counter() - start_time
        postprocessed_by_scene_id[scene.scene_id] = detections
        if scene_number % 100 == 0:
            logger.info("%d/%d scenes", scene_number, len(scenes))

    stage_suffix = ("+refine" if arguments.refine else "") + ("+orient" if arguments.orient else "")
    detector_config = dict(header.get("config", {}))
    postprocess_seconds_per_image = total_postprocess_seconds / max(len(scenes), 1)
    detector_config["postprocess_seconds_per_image"] = postprocess_seconds_per_image
    detector_config["seconds_per_image"] = detector_config.get("seconds_per_image", 0.0) + postprocess_seconds_per_image
    detector_config["source_predictions"] = str(arguments.predictions)
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
