"""Time the two spec budgets on a 12-megapixel photo with six checks, in headless Chromium.

    /Volumes/vega/datasets/check-transcriber/venv/bin/python app/tests/test_performance_budget.py [--runs 3]

Spec section 5: under 3 s from paste to the count step, under 10 s from Continue to a
fully populated review grid, for a 12 MP photo of six checks. The photo is the first
eval scene with exactly six checks, all fully in frame, upscaled to 4000 x 3000 and saved
as a JPEG (so decode cost is included, as for a real paste). Timings come from the page
itself (`performance.now()` at the paste event, at the count step render, at Continue,
at the last crop), via window.__checkTranscriberDebug. The first run includes JIT
warm-up and is reported separately.

This Mac is not the target laptop: report the numbers, and the note in app/CLAUDE.md
records how they compare to a mid-range Windows machine.
"""

import argparse
import statistics
import sys
from pathlib import Path

import cv2
from playwright.sync_api import sync_playwright

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from browser_test_helpers import paste_image_file, serve_directory, wait_for_debug_state, wait_for_engines_ready  # noqa: E402
from experiments.detection.dataset.scene_annotations import load_split_scene_annotations  # noqa: E402

TWELVE_MEGAPIXEL_SIZE = (4000, 3000)
TARGET_CHECK_COUNT = 6
JPEG_QUALITY = 90
PHOTO_PATH = Path("/tmp/check-transcriber-m2/twelve_megapixel_six_checks.jpg")
PASTE_TO_COUNT_BUDGET_S = 3.0
CONTINUE_TO_GRID_BUDGET_S = 10.0
TIMEOUT_MS = 120_000


def build_twelve_megapixel_photo() -> str:
    """Writes PHOTO_PATH from the first all-in-frame six-check eval scene; returns its id."""
    for scene in load_split_scene_annotations("eval"):
        if len(scene.checks) == TARGET_CHECK_COUNT and all(check.fully_in_frame for check in scene.checks):
            image = cv2.imread(str(scene.image_path), cv2.IMREAD_COLOR)
            upscaled = cv2.resize(image, TWELVE_MEGAPIXEL_SIZE, interpolation=cv2.INTER_CUBIC)
            PHOTO_PATH.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(PHOTO_PATH), upscaled, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            return scene.scene_id
    raise SystemExit("no eval scene with six fully in-frame checks")


def time_one_run(page, previous_arrival: float) -> dict:
    """Pastes the photo, continues, returns the two durations in seconds."""
    paste_image_file(page, PHOTO_PATH)
    state = wait_for_debug_state(page, f"state.step === 'count' && state.timings.photoArrivedAt > {previous_arrival} && state.timings.countStepShownAt", TIMEOUT_MS)
    detected_count = len(state["detectedQuads"])
    page.click("#count-continue-button")
    state = wait_for_debug_state(page, "state.timings.gridCompleteAt > state.timings.continuedAt", TIMEOUT_MS)
    timings = state["timings"]
    return {
        "arrival": timings["photoArrivedAt"],
        "detected_count": detected_count,
        "paste_to_count_s": (timings["countStepShownAt"] - timings["photoArrivedAt"]) / 1000,
        "continue_to_grid_s": (timings["gridCompleteAt"] - timings["continuedAt"]) / 1000,
    }


def main() -> None:
    """Build the photo, run it `--runs` times after one warm-up, report against the budget."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=3)
    arguments = parser.parse_args()
    scene_id = build_twelve_megapixel_photo()
    print(f"12 MP photo from {scene_id}: {PHOTO_PATH} ({PHOTO_PATH.stat().st_size / 1e6:.1f} MB)")
    with serve_directory() as base_url, sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_context(viewport={"width": 1400, "height": 900}).new_page()
        page.on("dialog", lambda dialog: dialog.accept())
        page.goto(base_url)
        wait_for_engines_ready(page, TIMEOUT_MS)
        runs = []
        previous_arrival = 0
        for run_index in range(arguments.runs + 1):
            run = time_one_run(page, previous_arrival)
            previous_arrival = run["arrival"]
            label = "warm-up" if run_index == 0 else f"run {run_index}"
            print(f"{label}: {run['detected_count']} checks, paste->count {run['paste_to_count_s']:.2f} s, "
                  f"Continue->grid {run['continue_to_grid_s']:.2f} s", flush=True)
            if run_index > 0:
                runs.append(run)
        browser.close()
    paste_median = statistics.median(run["paste_to_count_s"] for run in runs)
    grid_median = statistics.median(run["continue_to_grid_s"] for run in runs)
    print(f"median paste->count {paste_median:.2f} s (budget {PASTE_TO_COUNT_BUDGET_S} s), "
          f"Continue->grid {grid_median:.2f} s (budget {CONTINUE_TO_GRID_BUDGET_S} s)")
    if paste_median > PASTE_TO_COUNT_BUDGET_S or grid_median > CONTINUE_TO_GRID_BUDGET_S:
        print("OVER BUDGET on this machine")
        sys.exit(1)
    print("within budget on this machine")


if __name__ == "__main__":
    main()
