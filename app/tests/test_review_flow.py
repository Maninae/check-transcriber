"""The count step and review grid, driven like the operator would, with screenshots.

    /Volumes/vega/datasets/check-transcriber/venv/bin/python app/tests/test_review_flow.py

On one eval scene: remove a detection (click outline, Delete), add it back by dragging
a rough box (the worker re-fits it), drag a corner, Continue; then in the grid type into
fields, undo with Ctrl+Z, Tab and Enter between fields, copy a field, copy a row (TSV on
the clipboard, row ticked done), rotate a crop, toggle the MICR band, open the lightbox
(arrows, Escape), check the unload guard, Copy all rows, and Finish batch (images
released, "N checks recorded"). Screenshots of the count step, the grid and the lightbox
go to /tmp/check-transcriber-m2/ for a human (and the agent) to look at.
"""

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))

from browser_test_helpers import (  # noqa: E402
    describe_page_state,
    paste_image_file,
    serve_directory,
    wait_for_debug_state,
    wait_for_engines_ready,
)

SCENE_IMAGE_PATH = Path("/Volumes/vega/datasets/check-transcriber/synth/v1/eval/images/eval_000107.jpg")
SCREENSHOT_DIRECTORY = Path("/tmp/check-transcriber-m2")
TIMEOUT_MS = 120_000


class CheckList:
    """Collects pass/fail lines and reports them all at the end."""

    def __init__(self):
        self.failures = []

    def that(self, description: str, condition: bool) -> None:
        print(f"[{'PASS' if condition else 'FAIL'}] {description}", flush=True)
        if not condition:
            self.failures.append(description)


def photo_point_to_page(page, x: float, y: float) -> tuple[float, float]:
    """Full-resolution photo coordinates -> page coordinates over the count overlay."""
    return tuple(page.evaluate(
        """([x, y]) => { const svg = document.getElementById('count-overlay'); const rect = svg.getBoundingClientRect();
        const [, , width, height] = svg.getAttribute('viewBox').split(' ').map(Number);
        return [rect.left + x * rect.width / width, rect.top + y * rect.height / height]; }""", [x, y]))


def bring_photo_point_into_view(page, x: float, y: float) -> None:
    """Scrolls so the photo point sits mid-viewport (mouse events off-screen hit nothing)."""
    _, page_y = photo_point_to_page(page, x, y)
    page.evaluate("(offset) => window.scrollBy(0, offset)", page_y - page.viewport_size["height"] / 2)


def exercise_count_step(page, checks: CheckList) -> int:
    """Remove, re-add by drawing, drag a corner; returns the final check count."""
    state = wait_for_debug_state(page, "state.step === 'count'", TIMEOUT_MS)
    detected = state["currentQuads"]
    checks.that("count step shows the detected checks", len(detected) > 0)
    heading = page.locator("#count-step-heading").inner_text()
    checks.that(f"header says 'Found {len(detected)} checks' ({heading!r})", heading == f"Found {len(detected)} {'check' if len(detected) == 1 else 'checks'}")
    page.screenshot(path=str(SCREENSHOT_DIRECTORY / "count_step.png"), full_page=True)

    # The typical "missed one": a check well inside the frame, away from its neighbours.
    image_width, image_height = page.evaluate("() => document.getElementById('count-overlay').getAttribute('viewBox').split(' ').slice(2).map(Number)")
    margin = 0.03 * max(image_width, image_height)
    interior = [quad for quad in detected if all(margin < x < image_width - margin and margin < y < image_height - margin for x, y in quad)]
    removed = (interior or detected)[0]
    centre = [sum(x for x, _ in removed) / 4, sum(y for _, y in removed) / 4]
    bring_photo_point_into_view(page, *centre)
    page.mouse.click(*photo_point_to_page(page, *centre))
    page.keyboard.press("Delete")
    state = describe_page_state(page)
    checks.that("click + Delete removes a check", len(state["currentQuads"]) == len(detected) - 1)

    xs = [x for x, _ in removed]
    ys = [y for _, y in removed]
    width, height = max(xs) - min(xs), max(ys) - min(ys)
    page.click("#add-check-button")
    bring_photo_point_into_view(page, *centre)
    start = photo_point_to_page(page, min(xs) - 0.04 * width, min(ys) + 0.05 * height)
    end = photo_point_to_page(page, max(xs) + 0.03 * width, max(ys) - 0.04 * height)
    page.mouse.move(*start)
    page.mouse.down()
    page.mouse.move(*end, steps=8)
    page.mouse.up()
    page.wait_for_function("() => !document.querySelector('.count-outline--pending')", timeout=TIMEOUT_MS)
    state = describe_page_state(page)
    checks.that("dragging a box adds the check back", len(state["currentQuads"]) == len(detected))
    refit_errors = []
    for quad in state["currentQuads"]:
        distances = [min(((qx - rx) ** 2 + (qy - ry) ** 2) ** 0.5 for rx, ry in removed) for qx, qy in quad]
        refit_errors.append(sum(distances) / 4)
    best_refit_error = min(refit_errors)
    checks.that(f"the rough box was re-fitted to the check's edges (mean corner distance {best_refit_error:.1f} px)", best_refit_error < 8)

    # Grab the corner farthest from every other check's corners, so no neighbour's handle is on top.
    quads = state["currentQuads"]
    def isolation(quad_index, corner_index):
        x, y = quads[quad_index][corner_index]
        return min(((x - ox) ** 2 + (y - oy) ** 2) ** 0.5 for other_index, other in enumerate(quads) if other_index != quad_index for ox, oy in other)
    quad_index, corner_index = max(((q, c) for q in range(len(quads)) for c in range(4)), key=lambda pair: isolation(*pair))
    corner_before = quads[quad_index][corner_index]
    bring_photo_point_into_view(page, *corner_before)
    page.mouse.move(*photo_point_to_page(page, *corner_before))
    page.mouse.down()
    page.mouse.move(*photo_point_to_page(page, corner_before[0] + 30, corner_before[1] + 20), steps=5)
    page.mouse.up()
    quads_after = describe_page_state(page)["currentQuads"]
    corner_after = quads_after[quad_index][corner_index]
    others_unchanged = all(quads_after[q][c] == quads[q][c] for q in range(len(quads)) for c in range(4) if (q, c) != (quad_index, corner_index))
    checks.that("dragging a corner moves only that corner", abs(corner_after[0] - corner_before[0] - 30) < 3 and abs(corner_after[1] - corner_before[1] - 20) < 3 and others_unchanged)
    page.mouse.move(*photo_point_to_page(page, *corner_after))
    page.mouse.down()
    page.mouse.move(*photo_point_to_page(page, *corner_before), steps=5)
    page.mouse.up()
    return len(quads)


def exercise_review_grid(page, checks: CheckList, check_count: int) -> None:
    """Fields, undo, keyboard flow, copy, rotate, MICR toggle, lightbox, guard, finish."""
    page.click("#count-continue-button")
    wait_for_debug_state(page, "state.timings.gridCompleteAt > state.timings.continuedAt", TIMEOUT_MS)
    checks.that("one review row per confirmed check", page.locator(".review-row").count() == check_count)

    payer = page.locator("#field-1-payer")
    payer.click()
    payer.type("Lauren Copeland")
    page.wait_for_timeout(800)
    payer.type(" Jr")
    page.keyboard.press("Control+z")
    checks.that("Ctrl+Z undoes only the last typing burst", payer.input_value() == "Lauren Copeland")
    page.keyboard.press("Tab")
    checks.that("Tab moves to the next field", page.evaluate("() => document.activeElement.id") == "field-1-amount")
    page.keyboard.type("1830.00")
    page.keyboard.press("Enter")
    checks.that("Enter moves to the next field", page.evaluate("() => document.activeElement.id") == "field-1-date")
    page.keyboard.type("2025-04-28")
    page.fill("#field-1-checkNumber", "2683")
    page.fill("#field-1-memo", "Sept rent")

    page.locator(".review-row >> nth=0 >> .copy-icon-button >> nth=1").click()
    checks.that("the amount copy button copies the amount", page.evaluate("() => navigator.clipboard.readText()") == "1830.00")
    page.locator(".review-row >> nth=0 >> text=Copy row").click()
    page.wait_for_timeout(200)
    checks.that("Copy row puts Date, Payer, Amount, Check number, Memo on the clipboard, tab-separated",
                page.evaluate("() => navigator.clipboard.readText()") == "2025-04-28\tLauren Copeland\t1830.00\t2683\tSept rent")
    checks.that("Copy row ticks the row done", page.locator(".review-row >> nth=0").get_attribute("class").find("review-row--done") >= 0)

    crop_before = page.evaluate("() => document.querySelector('.review-crop-canvas').toDataURL().length")
    page.locator(".review-row >> nth=1 >> text=Rotate").click()
    page.locator(".review-row >> nth=1 >> text=Show bottom line").click()
    checks.that("the MICR toggle flips to 'Blur bottom line'", page.locator(".review-row >> nth=1 >> text=Blur bottom line").count() == 1)
    page.locator(".review-row >> nth=1 >> text=Blur bottom line").click()
    checks.that("row 1's crop is unchanged by row 2's controls", page.evaluate("() => document.querySelector('.review-crop-canvas').toDataURL().length") == crop_before)
    page.locator("#review-step").screenshot(path=str(SCREENSHOT_DIRECTORY / "review_grid.png"))

    page.locator(".review-row >> nth=0 >> .review-crop-canvas").click()
    checks.that("clicking a crop opens the lightbox", page.locator("#lightbox").is_visible())
    page.wait_for_timeout(200)
    page.screenshot(path=str(SCREENSHOT_DIRECTORY / "lightbox.png"))
    page.keyboard.press("ArrowRight")
    checks.that("right arrow steps to the next check", page.locator("#lightbox-caption").inner_text() == f"Check 2 of {check_count}")
    page.mouse.move(700, 450)
    page.mouse.wheel(0, -400)
    zoomed = page.evaluate("() => document.getElementById('lightbox-canvas').style.transform")
    checks.that(f"wheel zooms ({zoomed})", "scale(1)" not in zoomed)
    page.screenshot(path=str(SCREENSHOT_DIRECTORY / "lightbox_zoomed.png"))
    checks.that("the lightbox holds no input fields", page.locator("#lightbox input, #lightbox textarea").count() == 0)
    page.keyboard.press("Escape")
    checks.that("Escape closes the lightbox", not page.locator("#lightbox").is_visible())

    guard_fires = page.evaluate("() => { const event = new Event('beforeunload', { cancelable: true }); window.dispatchEvent(event); return event.defaultPrevented; }")
    checks.that("leaving with uncopied rows triggers the 'leave page?' guard", guard_fires)
    page.click("#copy-all-bottom-button")
    page.wait_for_timeout(200)
    copied_lines = page.evaluate("() => navigator.clipboard.readText()").split("\n")
    checks.that("Copy all rows copies one line per check in count order", len(copied_lines) == check_count and copied_lines[0].startswith("2025-04-28\tLauren Copeland"))
    checks.that("the toast confirms the copy", page.locator("#toast").inner_text() == f"{check_count} rows copied")
    guard_after_copy = page.evaluate("() => { const event = new Event('beforeunload', { cancelable: true }); window.dispatchEvent(event); return event.defaultPrevented; }")
    checks.that("no guard once every row is copied", not guard_after_copy)

    page.click("#finish-batch-button")
    checks.that("Finish batch returns to the empty drop zone", page.locator("#drop-zone-prompt").is_visible())
    checks.that(f"'{check_count} checks recorded' stays visible", page.locator("#batch-recorded-line").inner_text() == f"{check_count} checks recorded")
    checks.that("no check image remains in the page", page.evaluate("() => [...document.querySelectorAll('canvas')].every((canvas) => canvas.width === 0)"))


def main() -> None:
    """Run the flow and report."""
    SCREENSHOT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    checks = CheckList()
    with serve_directory() as base_url, sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context(viewport={"width": 1400, "height": 900})
        context.grant_permissions(["clipboard-read", "clipboard-write"])
        page = context.new_page()
        page.on("pageerror", lambda error: checks.that(f"no page error ({error})", False))
        page.goto(base_url)
        wait_for_engines_ready(page, TIMEOUT_MS)
        paste_image_file(page, SCENE_IMAGE_PATH)
        check_count = exercise_count_step(page, checks)
        exercise_review_grid(page, checks, check_count)
        browser.close()
    print(f"\nscreenshots: {SCREENSHOT_DIRECTORY}/count_step.png, review_grid.png, lightbox.png, lightbox_zoomed.png")
    if checks.failures:
        print(f"{len(checks.failures)} check(s) failed")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
