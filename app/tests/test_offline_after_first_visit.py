"""Offline after the first visit: the service worker must serve every file the pipeline needs.

    python3 app/tests/test_offline_after_first_visit.py

First visit online: wait for the engines and for the service worker to control the page,
then reload once so the worker's runtime cache fills from a controlled page. Then cut the
network (Playwright offline mode) and reload: the engines must come up again, a pasted
photo must reach the count step with checks found, and Continue must crop every check and
read its fields (some field filled by the reader), with zero failed requests. This proves
the CDN libraries (OpenCV.js, onnxruntime-web's .mjs and .wasm, Tesseract), the classifier
and the four field-reading models, and every app module are all served from the caches,
including the pipeline worker's own fetches from its blob: bootstrap.
"""

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))

from browser_test_helpers import paste_image_file, serve_directory, wait_for_debug_state, wait_for_engines_ready  # noqa: E402

SCENE_IMAGE_PATH = Path("/Volumes/vega/datasets/check-transcriber/synth/v1/eval/images/eval_000107.jpg")
TIMEOUT_MS = 180_000


def main() -> None:
    """Online visit, then offline reload and a full detection."""
    with serve_directory() as base_url, sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context()
        page = context.new_page()
        page.goto(base_url)
        wait_for_engines_ready(page, TIMEOUT_MS)
        page.wait_for_function("() => navigator.serviceWorker.controller !== null", timeout=TIMEOUT_MS)
        page.reload()
        wait_for_engines_ready(page, TIMEOUT_MS)

        context.set_offline(True)
        failed_requests = []
        page.on("requestfailed", lambda request: failed_requests.append(f"{request.url} -- {request.failure}"))
        page.reload()
        wait_for_engines_ready(page, TIMEOUT_MS)
        paste_image_file(page, SCENE_IMAGE_PATH)
        state = wait_for_debug_state(page, "state.step === 'count'", TIMEOUT_MS)
        detected_count = len(state["detectedQuads"])
        page.click("#count-continue-button")
        state = wait_for_debug_state(page, "state.timings.gridCompleteAt > state.timings.continuedAt", TIMEOUT_MS)
        browser.close()

    filled_values = [field["value"] for row in state["rows"] for field in row["fields"].values() if field["gatedState"] in ("confident", "unsure")]
    print(f"offline: engines ready, count step reached with {detected_count} checks; "
          f"{state['croppedRowCount']} crops and {len(filled_values)} fields read by the field reader")
    if failed_requests:
        print("FAIL: requests failed while offline:")
        for failure in failed_requests:
            print(f"  {failure}")
        sys.exit(1)
    if not state["detectedQuads"]:
        print("FAIL: no checks detected offline")
        sys.exit(1)
    if state["croppedRowCount"] != detected_count or not filled_values:
        print("FAIL: offline Continue did not crop every check and fill fields")
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
