"""Browser detection regression: the shipped JS pipeline vs the Python baseline on eval.

    <venv>/bin/python app/tests/test_detection_regression.py [--scenes 40]
    <venv>/bin/python app/tests/test_detection_regression.py --dataset closeup

Two scene sets:
- `v1` (default, gated): the wide-framing v1 eval split, compared with the Python pipeline.
- `closeup` (report only): 15 "close" + 5 "single" scenes from the v1.1 close-up eval set,
  the framing the operator actually shoots (5-6 checks filling the frame, or one check).
  No Python predictions exist for it, so it is scored against ground truth only, per regime.

Serves app/, pastes each eval scene into the page exactly as Ctrl+V would, waits for the
count step, clicks Continue so the orientation classifier runs, and reads the refined,
oriented quads out of `window.__checkTranscriberDebug`. Those become a predictions dict
scored IN-PROCESS by experiments/detection/metrics (the same definitions as the Python
harness: per-check outline IoU, greedy matching, corner error at the best cyclic shift,
orientation at shift 0). The Python "classical + refine + orient" predictions on the SAME
scenes are scored alongside, plus a per-scene JS-vs-Python diff, so any gap is explained
per scene rather than guessed at.

Scene choice is deterministic: the seven eval ids used in the detection showcase
overlays, then evenly spaced eval ids until the requested count.

Exit status 1 when the browser's scene-perfect rate trails Python's on the same scenes by
more than `MAXIMUM_SCENE_PERFECT_GAP`, or the corner median by more than
`MAXIMUM_CORNER_MEDIAN_GAP_PX`.
"""

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from browser_test_helpers import paste_image_file, serve_directory, wait_for_debug_state, wait_for_engines_ready  # noqa: E402
from experiments.detection.dataset.scene_annotations import load_scene_annotation, load_split_scene_annotations  # noqa: E402
from experiments.detection.metrics.quadrilateral_geometry import clipped_polygon_in_frame, polygon_iou  # noqa: E402
from experiments.detection.metrics.score_predictions import score_predictions_against_split  # noqa: E402
from experiments.detection.predictions.detected_check import DetectedCheck, load_predictions_file  # noqa: E402

SHOWCASE_EVAL_SCENE_IDS = ("eval_000012", "eval_000270", "eval_000098", "eval_000193", "eval_000633", "eval_000107", "eval_000059")
PYTHON_BASELINE_PREDICTIONS = Path(
    "/Volumes/vega/datasets/check-transcriber/experiments/detection/predictions/eval__classical+refine+orient.json"
)
CLOSEUP_EVAL_ROOT = Path("/Volumes/vega/datasets/check-transcriber/synth/v1.1-closeup-eval/eval")
CLOSEUP_SCENES_PER_REGIME = {"close": 15, "single": 5}
RESULTS_DIRECTORY = Path("/tmp/check-transcriber-m2")
ENGINE_READY_TIMEOUT_MS = 180_000
PER_SCENE_TIMEOUT_MS = 90_000
MAXIMUM_SCENE_PERFECT_GAP = 0.05
MAXIMUM_CORNER_MEDIAN_GAP_PX = 0.3
PAIRING_IOU_FOR_DIFF = 0.5


def choose_scene_ids(all_scene_ids: list[str], scene_count: int) -> list[str]:
    """The showcase ids first, then evenly spaced ids from the sorted split."""
    chosen = [scene_id for scene_id in SHOWCASE_EVAL_SCENE_IDS if scene_id in all_scene_ids]
    remaining = [scene_id for scene_id in sorted(all_scene_ids) if scene_id not in chosen]
    extra_needed = scene_count - len(chosen)
    if extra_needed > 0:
        step = len(remaining) / extra_needed
        chosen += [remaining[int(index * step)] for index in range(extra_needed)]
    return chosen[:scene_count]


def framing_regime_of(annotation_path: Path) -> str:
    """The v1.1 framing regime ("close", "single"; missing means "wide")."""
    effects = json.loads(annotation_path.read_text()).get("effects") or {}
    return (effects.get("framing") or {}).get("framing_regime", "wide")


def choose_closeup_scenes() -> dict:
    """regime -> evenly spaced scenes of that regime, image paths pointed at the v1.1 set."""
    paths_by_regime = {regime: [] for regime in CLOSEUP_SCENES_PER_REGIME}
    for annotation_path in sorted((CLOSEUP_EVAL_ROOT / "annotations").glob("*.json")):
        regime = framing_regime_of(annotation_path)
        if regime in paths_by_regime:
            paths_by_regime[regime].append(annotation_path)
    scenes_by_regime = {}
    for regime, wanted in CLOSEUP_SCENES_PER_REGIME.items():
        paths = paths_by_regime[regime]
        chosen = [paths[int(index * len(paths) / wanted)] for index in range(wanted)]
        scenes_by_regime[regime] = [
            dataclasses.replace(scene, image_path=CLOSEUP_EVAL_ROOT / "images" / scene.image_path.name)
            for scene in (load_scene_annotation(path, "eval") for path in chosen)
        ]
    return scenes_by_regime


def run_closeup_report() -> None:
    """Browser vs ground truth on the close-up set, per regime; report only."""
    scenes_by_regime = choose_closeup_scenes()
    all_scenes = [scene for scenes in scenes_by_regime.values() for scene in scenes]
    print(f"browser on {len(all_scenes)} close-up scenes ({', '.join(f'{len(v)} {k}' for k, v in scenes_by_regime.items())})", flush=True)
    browser_results = run_browser_on_scenes(all_scenes)
    report = {}
    for regime, scenes in scenes_by_regime.items():
        predictions = {scene.scene_id: to_detected_checks(browser_results[scene.scene_id]["quads"]) for scene in scenes}
        report[regime] = summarize(score_predictions_against_split(predictions, scenes, include_records=False))
        report[regime]["extra_detections"] = sum(max(0, len(predictions[scene.scene_id]) - len(scene.checks)) for scene in scenes)
        report[regime]["missed_checks"] = sum(max(0, len(scene.checks) - len(predictions[scene.scene_id])) for scene in scenes)
    print(f"\n{'metric':<26}" + "".join(f"{regime:>12}" for regime in report))
    for key in report["close"]:
        print(f"{key:<26}" + "".join(f"{report[regime][key]:>12.4f}" if report[regime][key] is not None else f"{'-':>12}" for regime in report))
    results_path = RESULTS_DIRECTORY / "detection_regression__closeup.json"
    results_path.write_text(json.dumps({
        "scene_ids": {regime: [scene.scene_id for scene in scenes] for regime, scenes in scenes_by_regime.items()},
        "browser_by_regime": report,
        "browser_quads": {scene_id: result["quads"] for scene_id, result in browser_results.items()},
    }, indent=1))
    print(f"results: {results_path}")


def run_browser_on_scenes(scenes) -> dict:
    """scene_id -> {"quads": [[x, y] x4 ...] oriented, "seconds_to_count": float}."""
    results = {}
    with serve_directory() as base_url, sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_context(viewport={"width": 1400, "height": 900}).new_page()
        page.on("dialog", lambda dialog: dialog.accept())  # "Start over with a new photo?"
        page.on("pageerror", lambda error: print(f"  [pageerror] {error}", flush=True))
        page.goto(base_url)
        wait_for_engines_ready(page, ENGINE_READY_TIMEOUT_MS)
        for scene_number, scene in enumerate(scenes):
            paste_image_file(page, scene.image_path)
            state = wait_for_debug_state(
                page, "state.step === 'count' && state.timings.countStepShownAt > state.timings.photoArrivedAt"
                f" && state.timings.photoArrivedAt > {results.get('_last_arrival', 0)}", PER_SCENE_TIMEOUT_MS,
            )
            results["_last_arrival"] = state["timings"]["photoArrivedAt"]
            seconds_to_count = (state["timings"]["countStepShownAt"] - state["timings"]["photoArrivedAt"]) / 1000
            quads = state["detectedQuads"]
            if quads:
                page.click("#count-continue-button")
                state = wait_for_debug_state(page, "state.timings.gridCompleteAt > state.timings.continuedAt", PER_SCENE_TIMEOUT_MS)
                quads = state["orientedQuads"]
            results[scene.scene_id] = {"quads": quads, "seconds_to_count": seconds_to_count}
            print(f"  {scene_number + 1:>2}/{len(scenes)} {scene.scene_id}: {len(quads)} checks (GT {len(scene.checks)}), "
                  f"{seconds_to_count:.2f} s to count step", flush=True)
        browser.close()
    results.pop("_last_arrival", None)
    return results


def to_detected_checks(quads) -> list[DetectedCheck]:
    """Browser quads are oriented (start at the check's own top-left)."""
    return [DetectedCheck(corners=np.asarray(quad, dtype=np.float64), orientation_known=True) for quad in quads]


def best_roll_corner_distance(first_corners: np.ndarray, second_corners: np.ndarray) -> float:
    """Mean corner distance at the best cyclic shift (orientation-agnostic)."""
    return min(float(np.linalg.norm(np.roll(first_corners, shift, axis=0) - second_corners, axis=1).mean()) for shift in range(4))


def diff_scene_against_python(browser_checks, python_checks) -> dict:
    """Count match and, for checks paired by IoU, the JS-vs-Python corner distance."""
    distances = []
    unused_python = list(range(len(python_checks)))
    for browser_check in browser_checks:
        browser_polygon = clipped_polygon_in_frame(browser_check.corners, 10**6, 10**6)
        scored = [(polygon_iou(browser_polygon, clipped_polygon_in_frame(python_checks[index].corners, 10**6, 10**6)), index) for index in unused_python]
        if not scored:
            continue
        best_iou, best_index = max(scored)
        if best_iou >= PAIRING_IOU_FOR_DIFF:
            unused_python.remove(best_index)
            distances.append(best_roll_corner_distance(browser_check.corners, python_checks[best_index].corners))
    return {"count_equal": len(browser_checks) == len(python_checks), "corner_distances_px": distances}


def summarize(metrics: dict) -> dict:
    """The handful of numbers the report compares."""
    localization = metrics["localization"]
    return {
        "scenes_all_correct": metrics["scenes"]["scene_perfect_rate"],
        "count_correct": metrics["scenes"]["count_accuracy"],
        "precision_at_0.5": metrics["detection"]["0.50"]["precision"],
        "recall_at_0.5": metrics["detection"]["0.50"]["recall"],
        "recall_at_0.9": metrics["detection"]["0.90"]["recall"],
        "iou_mean": localization["iou_with_outline"]["mean"],
        "corner_error_median_px": localization["corner_error_mean_px"]["median"],
        "corner_error_p90_px": localization["corner_error_mean_px"]["p90"],
        "orientation_accuracy": localization["orientation_accuracy"],
    }


def main() -> None:
    """Run, score, compare, write the results JSON, set the exit status."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenes", type=int, default=40)
    parser.add_argument("--dataset", choices=("v1", "closeup"), default="v1")
    arguments = parser.parse_args()
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    if arguments.dataset == "closeup":
        run_closeup_report()
        return
    all_scenes = {scene.scene_id: scene for scene in load_split_scene_annotations("eval")}
    scenes = [all_scenes[scene_id] for scene_id in choose_scene_ids(list(all_scenes), arguments.scenes)]
    print(f"browser regression on {len(scenes)} eval scenes", flush=True)

    started = time.perf_counter()
    browser_results = run_browser_on_scenes(scenes)
    browser_predictions = {scene_id: to_detected_checks(result["quads"]) for scene_id, result in browser_results.items()}
    _, python_all_predictions = load_predictions_file(PYTHON_BASELINE_PREDICTIONS)
    python_predictions = {scene.scene_id: python_all_predictions.get(scene.scene_id, []) for scene in scenes}

    browser_summary = summarize(score_predictions_against_split(browser_predictions, scenes, include_records=False))
    python_summary = summarize(score_predictions_against_split(python_predictions, scenes, include_records=False))
    scene_diffs = {scene.scene_id: diff_scene_against_python(browser_predictions[scene.scene_id], python_predictions[scene.scene_id]) for scene in scenes}
    all_distances = [distance for diff in scene_diffs.values() for distance in diff["corner_distances_px"]]
    seconds_to_count = [result["seconds_to_count"] for result in browser_results.values()]
    parity = {
        "scenes_with_equal_count": sum(diff["count_equal"] for diff in scene_diffs.values()) / len(scenes),
        "scenes_with_different_count": [scene_id for scene_id, diff in scene_diffs.items() if not diff["count_equal"]],
        "js_vs_python_corner_px_median": float(np.median(all_distances)) if all_distances else None,
        "js_vs_python_corner_px_p95": float(np.percentile(all_distances, 95)) if all_distances else None,
        "js_vs_python_corner_px_max": float(np.max(all_distances)) if all_distances else None,
        "paired_checks": len(all_distances),
    }

    print(f"\n{'metric':<26}{'browser (JS)':>14}{'Python':>12}")
    for key in browser_summary:
        print(f"{key:<26}{browser_summary[key]:>14.4f}{python_summary[key]:>12.4f}")
    print(f"\nJS vs Python per scene: {json.dumps(parity, indent=2)}")
    print(f"browser seconds paste->count step on these 4-6 MP scenes: median {np.median(seconds_to_count):.2f}, max {np.max(seconds_to_count):.2f}")
    print(f"total wall time {time.perf_counter() - started:.0f} s")

    results_path = RESULTS_DIRECTORY / "detection_regression__results.json"
    results_path.write_text(json.dumps({
        "scene_ids": [scene.scene_id for scene in scenes], "browser": browser_summary, "python_same_scenes": python_summary,
        "parity": parity, "seconds_to_count_step": seconds_to_count,
        "browser_quads": {scene_id: result["quads"] for scene_id, result in browser_results.items()},
    }, indent=1))
    print(f"results: {results_path}")

    scene_perfect_gap = python_summary["scenes_all_correct"] - browser_summary["scenes_all_correct"]
    corner_gap = browser_summary["corner_error_median_px"] - python_summary["corner_error_median_px"]
    if scene_perfect_gap > MAXIMUM_SCENE_PERFECT_GAP or corner_gap > MAXIMUM_CORNER_MEDIAN_GAP_PX:
        print(f"FAIL: scene-perfect gap {scene_perfect_gap:.3f}, corner median gap {corner_gap:.3f} px")
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
