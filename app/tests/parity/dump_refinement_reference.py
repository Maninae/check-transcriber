"""Run the Python reference corner refinement on fixed inputs, for the JS parity runner.

    PYTHONPATH=. <venv>/bin/python app/tests/parity/dump_refinement_reference.py --input-name yolo_val
    PYTHONPATH=. <venv>/bin/python app/tests/parity/dump_refinement_reference.py --input-name classical_val
    PYTHONPATH=. <venv>/bin/python app/tests/parity/dump_refinement_reference.py --input-name classical_eval

Run from the repository root. For every scene of the input set it writes, into one JSON
(`<output>/reference__<input-name>.json`):

- `input_quads`: the detector corners exactly as the JS runner will read them.
- `python_refined`: `refine_detected_checks` output corners (same order), plus a few
  diagnostics (quad reverted, narrow-band acceptance, local corner fits, paper colour).
- `ground_truth`: per detection, the IoU-matched GT corners and their in-frame mask
  (None when unmatched), so both runtimes are scored against the same GT pairing.
- `python_milliseconds`: wall time of the whole scene's refinement, cv2 single-threaded.

Scene pixels are dumped separately with `dump_scene_pixels.py` into the same directory.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from experiments.detection.metrics.quadrilateral_geometry import corner_inside_frame_mask
from experiments.detection.dataset.scene_annotations import load_scene_annotation
from experiments.detection.predictions.detected_check import DetectedCheck
from experiments.detection.refinement.quadrilateral_refinement import refine_detected_checks
from experiments.detection.refinement.refinement_evaluation_records import match_predictions_to_ground_truth

DATASET_ROOT = Path("/Volumes/vega/datasets/check-transcriber")
SYNTHETIC_DATASET_ROOT = DATASET_ROOT / "synth/v1"
DEFAULT_OUTPUT_DIRECTORY = DATASET_ROOT / "tools/app-parity/refinement"
VAL_SCENE_STRIDE = 12
VAL_SCENE_COUNT = 60
EVAL_SCENE_IDS = ["eval_000012", "eval_000270", "eval_000098", "eval_000193", "eval_000633", "eval_000107", "eval_000059"]
INPUT_SETS = {
    "yolo_val": ("val", DATASET_ROOT / "experiments/detection/predictions/val__yolo26n-obb.json"),
    "classical_val": ("val", DATASET_ROOT / "experiments/detection/classical/final/val_predictions.json"),
    "classical_eval": ("eval", DATASET_ROOT / "experiments/detection/predictions/eval__classical.json"),
}


def select_scene_ids(split_name: str, predictions: dict) -> list[str]:
    """The fixed parity scene list: every 12th val scene (first 60), or the named eval scenes."""
    if split_name == "eval":
        return EVAL_SCENE_IDS
    all_val_ids = sorted(path.stem for path in (SYNTHETIC_DATASET_ROOT / "val/annotations").glob("*.json"))
    return [scene_id for scene_id in all_val_ids[::VAL_SCENE_STRIDE] if scene_id in predictions][:VAL_SCENE_COUNT]


def reference_for_scene(split_name: str, scene_id: str, detection_jsons: list[dict]) -> dict:
    """Refine one scene's detections with the Python reference and pair them with GT."""
    image_bgr = cv2.imread(str(SYNTHETIC_DATASET_ROOT / split_name / "images" / f"{scene_id}.jpg"), cv2.IMREAD_COLOR)
    detections = [DetectedCheck.from_json_dict(detection_json) for detection_json in detection_jsons]
    start_time = time.perf_counter()
    refined_checks = refine_detected_checks(image_bgr, detections)
    elapsed_milliseconds = 1000 * (time.perf_counter() - start_time)
    scene = load_scene_annotation(SYNTHETIC_DATASET_ROOT / split_name / "annotations" / f"{scene_id}.json", split_name)
    matched = match_predictions_to_ground_truth(scene, detections)
    ground_truth_by_detection = {id(detection): scene.checks[position] for position, detection in matched.items()}
    image_height, image_width = image_bgr.shape[:2]
    ground_truth = []
    for detection in detections:
        check = ground_truth_by_detection.get(id(detection))
        ground_truth.append(None if check is None else {
            "corners": check.corners.tolist(),
            "in_frame_mask": corner_inside_frame_mask(check.corners, image_width, image_height).tolist(),
        })
    return {
        "input_quads": [detection.corners.tolist() for detection in detections],
        "python_refined": [check.corners.tolist() for check in refined_checks],
        "python_diagnostics": [
            {
                "quad_reverted": check.extras["refinement"]["quad_reverted"],
                "corners_reverted_for_distance": check.extras["refinement"]["corners_reverted_for_distance"],
                "paper_colour": check.extras["refinement"]["paper_colour"],
                "sides_accepted_in_narrow_band": check.extras["refinement"]["passes"][0]["sides_accepted_in_narrow_band"],
                "local_corner_fits": check.extras["refinement"]["passes"][-1]["local_corner_fits"],
            }
            for check in refined_checks
        ],
        "ground_truth": ground_truth,
        "python_milliseconds": elapsed_milliseconds,
    }


def main() -> None:
    """Dump the reference for one input set."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input-name", choices=sorted(INPUT_SETS), required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--list-scenes", action="store_true", help="print the scene ids and exit")
    arguments = parser.parse_args()
    cv2.setNumThreads(1)
    split_name, predictions_path = INPUT_SETS[arguments.input_name]
    predictions = json.loads(predictions_path.read_text())["predictions"]
    scene_ids = select_scene_ids(split_name, predictions)
    if arguments.list_scenes:
        print(" ".join(scene_ids))
        return
    scenes = {}
    for scene_id in scene_ids:
        scenes[scene_id] = reference_for_scene(split_name, scene_id, predictions.get(scene_id, []))
        print(scene_id, len(scenes[scene_id]["input_quads"]), f"{scenes[scene_id]['python_milliseconds']:.0f} ms", flush=True)
    arguments.output.mkdir(parents=True, exist_ok=True)
    output_path = arguments.output / f"reference__{arguments.input_name}.json"
    output_path.write_text(json.dumps({"input_name": arguments.input_name, "split": split_name, "scenes": scenes}))
    print("wrote", output_path)


if __name__ == "__main__":
    main()
