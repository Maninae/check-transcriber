"""The count step and review grid, driven like the operator would, with screenshots.

    /Volumes/vega/datasets/check-transcriber/venv/bin/python app/tests/test_review_flow.py

On one eval scene: remove a detection (click outline, Delete), add it back by dragging a
rough box (the worker re-fits it), drag a corner, Continue. Then in the grid, with field
reads fed through `window.__checkTranscriberDebug.setFieldReads` (the same path as the
worker's reads, so every state below is deterministic):
- milestone 3: type into fields, Ctrl+Z, Tab and Enter, copy a field, Copy row;
- milestone 4: the three field states, the magnifier, Tab/Shift+Tab over unsure and blank
  fields only, Enter to the next one, Ctrl+Z undoing a payer snap and an autocomplete pick,
  autocomplete after two letters, the duplicate warning, the email-date suggestion, the
  date display setting (copy stays ISO), copy-column order and inclusion, re-gating when
  the payee list changes, Rotate flipping read boxes;
- rotate, the MICR toggle, the lightbox, the unload guard, Copy all rows, Finish batch
  (names and check numbers remembered as text, no pixels left), Clear everything.
Screenshots go to /tmp/check-transcriber-m2/ (count step, grid, lightbox) and
/tmp/check-transcriber-m4/ (mixed field states, inline magnifier, settings panel).
"""

import datetime
import json
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
M4_SCREENSHOT_DIRECTORY = Path("/tmp/check-transcriber-m4")
TIMEOUT_MS = 120_000
STORAGE_PREFIX = "checkTranscriber.v1."
SEEDED_PAYER_NAMES = ["Lauren Copeland", "Marcus Bell", "Maria Lopez"]
SEEDED_CHECK_HISTORY = [{"checkNumber": "2683", "payer": "Lauren Copeland", "date": "2026-08-30", "batchDate": "2026-09-12"}]
TODAY = datetime.date.today()


def printed_read(text: str, confidence: float = 0.995, box=None, reader: str = "crnn_general") -> dict:
    """A raw read of printed text, in the shape the worker posts."""
    return {"text": text, "confidence": confidence, "handwrittenProbability": 0.05, "reader": reader, "box": box}


def handwritten_read(text: str, box=None) -> dict:
    """A handwritten field the printed reader could not trust (gates to blank)."""
    return {"text": text, "confidence": 0.4, "handwrittenProbability": 0.9, "reader": "crnn_general", "box": box}


# Row 1: confident payer, amount, check number; unsure memo; blank date and payee.
ROW_1_READS = {
    "payer_name": printed_read("Lauren Copeland", box=[60, 40, 700, 120]),
    "amount_numeric": printed_read("$***453.00", box=[1240, 270, 1540, 370], reader="crnn_amount"),
    "amount_words": printed_read("FOUR HUNDRED FIFTY-THREE AND 00/100", box=[150, 380, 1200, 440]),
    "date": handwritten_read("9/1/26", box=[1020, 150, 1530, 260]),
    "check_number": printed_read("2683", confidence=0.9, box=[1430, 45, 1545, 115]),
    "memo": printed_read("Sept rent", confidence=0.97, box=[170, 500, 800, 610]),
    "payee": None,
}
# Row 2: a snapped payer (one edit from a known name) and amounts that disagree.
ROW_2_READS = {
    "payer_name": printed_read("Lauren Copelsnd", box=[60, 40, 700, 120]),
    "amount_numeric": printed_read("$***1,830.00", box=[1240, 270, 1540, 370], reader="crnn_amount"),
    "amount_words": printed_read("ONE THOUSAND EIGHT HUNDRED AND 00/100"),
    "date": printed_read(f"{TODAY.month}/{TODAY.day}/{TODAY.year}", confidence=0.99, box=[1020, 150, 1530, 260]),
    "check_number": printed_read("1044", confidence=0.9),
    "memo": None,
    "payee": None,
}
# Row 3: handwritten payer (blank), courtesy amount only (unsure), a payee read with no payee list yet.
ROW_3_READS = {
    "payer_name": handwritten_read("Dana W."),
    "amount_numeric": printed_read("$***95.00", reader="crnn_amount"),
    "amount_words": None,
    "date": None,
    "check_number": printed_read("301", confidence=0.9),
    "memo": None,
    "payee": printed_read("Oak Street Coop", box=[250, 260, 1200, 390]),
}
# Row 4: nothing read, so every field is a blank Tab stop (the milestone-3 typing row).
ROW_4_READS = {key: None for key in ["payer_name", "payee", "amount_numeric", "amount_words", "date", "memo", "check_number"]}


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


def row_fields(page, row_index: int) -> dict:
    """The debug hook's text-only view of one row's fields."""
    return describe_page_state(page)["rows"][row_index]["fields"]


def focused_id(page) -> str:
    return page.evaluate("() => document.activeElement.id")


def clipboard_text(page) -> str:
    page.wait_for_timeout(150)
    return page.evaluate("() => navigator.clipboard.readText()")


def app_storage(page) -> dict:
    """Every localStorage key this app owns -> raw value."""
    return page.evaluate(f"() => Object.fromEntries(Object.keys(localStorage).filter((key) => key.startsWith('checkTranscriber.')).map((key) => [key, localStorage.getItem(key)]))")


def exercise_milestone_3_typing(page, checks: CheckList) -> None:
    """Typed fields on row 4 (every field blank): undo bursts, Tab, Enter, copy a field and a row."""
    payer = page.locator("#field-4-payer")
    payer.click()
    payer.type("Dana Whitfield")
    page.wait_for_timeout(800)
    payer.type(" Jr")
    page.keyboard.press("Control+z")
    checks.that("Ctrl+Z undoes only the last typing burst", payer.input_value() == "Dana Whitfield")
    page.keyboard.press("Tab")
    checks.that("Tab moves to the next field", focused_id(page) == "field-4-amount")
    page.keyboard.type("1830")
    page.keyboard.press("Enter")
    checks.that("Enter moves to the next field", focused_id(page) == "field-4-date")
    checks.that("a typed amount commits to two decimals", page.locator("#field-4-amount").input_value() == "1830.00")
    page.keyboard.type("2025-04-28")
    page.fill("#field-4-checkNumber", "2683")
    page.fill("#field-4-memo", "Sept rent")
    page.locator(".review-row >> nth=3 >> .copy-icon-button >> nth=1").click()
    checks.that("the amount copy button copies the amount", clipboard_text(page) == "1830.00")
    page.locator(".review-row >> nth=3 >> text=Copy row").click()
    checks.that("Copy row puts Date, Payer, Amount, Check number, Memo on the clipboard, tab-separated",
                clipboard_text(page) == "2025-04-28\tDana Whitfield\t1830.00\t2683\tSept rent")
    checks.that("Copy row ticks the row done", "review-row--done" in page.locator(".review-row >> nth=3").get_attribute("class"))


def exercise_field_states(page, checks: CheckList) -> None:
    """The three states, the amount note, a snapped payer, the magnifier, and the screenshots."""
    row_1 = row_fields(page, 0)
    checks.that("confident fields are plain: payer, amount 453.00, check number",
                [row_1[key]["state"] for key in ("payer", "amount", "checkNumber")] == ["confident"] * 3 and row_1["amount"]["value"] == "453.00")
    checks.that("memo read is unsure with its value", row_1["memo"]["state"] == "unsure" and row_1["memo"]["value"] == "Sept rent")
    checks.that("handwritten date is blank", row_1["date"]["state"] == "blank" and row_1["date"]["value"] == "")
    flagged = page.evaluate("() => ['payer','amount','date','memo'].map((key) => document.getElementById(`field-1-${key}`).classList.contains('review-field-input--flagged'))")
    checks.that("only unsure and blank fields wear the amber highlight", flagged == [False, False, True, True])
    checks.that("a blank field says 'read from the check'", page.get_attribute("#field-1-date", "placeholder") == "read from the check")
    row_2 = row_fields(page, 1)
    checks.that("a near-miss payer snaps to the known name and is unsure",
                row_2["payer"]["value"] == "Lauren Copeland" and row_2["payer"]["state"] == "unsure" and row_2["payer"]["snappedFrom"] == "Lauren Copelsnd")
    checks.that("the snapped payer's tooltip shows the raw read", page.get_attribute("#field-2-payer", "title") == 'Read as "Lauren Copelsnd"')
    checks.that("disagreeing amounts: courtesy value, unsure, with the note shown",
                row_2["amount"]["value"] == "1830.00" and row_2["amount"]["state"] == "unsure"
                and page.locator(".review-row >> nth=1 >> .review-field-note:visible").inner_text() == "the written amount reads differently")
    checks.that("a blank payee with no payee list", row_fields(page, 2)["payee"]["state"] == "blank")

    rows_top = page.locator(".review-row >> nth=0").bounding_box()
    page.evaluate("(top) => window.scrollTo(0, window.scrollY + top - 20)", rows_top["y"])
    first, second = page.locator(".review-row >> nth=0").bounding_box(), page.locator(".review-row >> nth=1").bounding_box()
    page.screenshot(path=str(M4_SCREENSHOT_DIRECTORY / "grid_mixed_states.png"),
                    clip={"x": first["x"], "y": first["y"], "width": first["width"], "height": second["y"] + second["height"] - first["y"]})

    page.locator("#field-1-date").click()
    magnifier_shown = page.locator(".review-row >> nth=0 >> .review-magnifier").is_visible()
    outline_shown = page.locator(".review-row >> nth=0 >> .review-crop-box-outline").is_visible()
    checks.that("focusing a blank field magnifies its region inline and outlines its box", magnifier_shown and outline_shown)
    page.locator(".review-row >> nth=0").screenshot(path=str(M4_SCREENSHOT_DIRECTORY / "inline_magnifier.png"))
    page.keyboard.press("Escape")
    checks.that("Escape closes the magnifier", not page.locator(".review-row >> nth=0 >> .review-magnifier").is_visible())
    page.locator("#field-1-payer").click()
    checks.that("a confident field opens no magnifier", page.locator(".review-magnifier:visible").count() == 0)


def exercise_keyboard_flow(page, checks: CheckList) -> None:
    """Tab/Shift+Tab over unsure and blank fields only, Enter to the next, Ctrl+Z on a snap."""
    page.locator("#field-1-date").click()
    visited = []
    for _ in range(3):
        page.keyboard.press("Tab")
        visited.append(focused_id(page))
    checks.that(f"Tab skips confident fields and crosses rows ({visited})", visited == ["field-1-memo", "field-1-payee", "field-2-payer"])
    checks.that("tabbing past an unsure field confirms it", row_fields(page, 0)["memo"]["state"] == "confirmed")
    checks.that("tabbing past an empty blank field leaves it blank", row_fields(page, 0)["payee"]["state"] == "blank")
    page.keyboard.press("Shift+Tab")
    checks.that("Shift+Tab goes back", focused_id(page) == "field-1-payee")
    page.locator("#field-2-amount").click()
    page.keyboard.press("Enter")
    checks.that("Enter moves to the next unsure/blank field, skipping confident ones", focused_id(page) == "field-2-memo")
    checks.that("Enter confirms the unsure amount", row_fields(page, 1)["amount"]["state"] == "confirmed")
    page.locator("#field-2-payer").click()
    page.keyboard.press("Control+z")
    checks.that("Ctrl+Z undoes the payer snap back to what was read", page.locator("#field-2-payer").input_value() == "Lauren Copelsnd")


def exercise_autocomplete_and_warnings(page, checks: CheckList) -> None:
    """Payer autocomplete, the duplicate warning and the email-date suggestion."""
    page.locator("#field-3-payer").click()
    page.keyboard.type("M")
    checks.that("one letter offers nothing", page.locator(".payer-suggestions:visible").count() == 0)
    page.keyboard.type("a")
    suggestions = page.locator(".review-row >> nth=2 >> .payer-suggestion").all_inner_texts()
    checks.that(f"two letters offer the remembered names ({suggestions})", suggestions == ["Marcus Bell", "Maria Lopez"])
    page.keyboard.press("ArrowDown")
    page.keyboard.press("Enter")
    checks.that("arrow + Enter picks a suggestion and stays in the field",
                page.locator("#field-3-payer").input_value() == "Marcus Bell" and focused_id(page) == "field-3-payer")
    page.keyboard.press("Control+z")
    checks.that("Ctrl+Z undoes the autocomplete pick", page.locator("#field-3-payer").input_value() == "Ma")
    page.keyboard.type("ria Lopez")
    page.keyboard.press("Escape")
    page.keyboard.press("Enter")
    checks.that("Enter with nothing highlighted moves on", focused_id(page) == "field-3-amount")

    warning = page.locator(".review-row >> nth=0 >> .review-duplicate-warning")
    checks.that(f"a past check number + payer shows the warning ({warning.inner_text()!r})", warning.inner_text() == "Seen before: batch on Sep 12.")
    page.fill("#field-1-checkNumber", "2684")
    checks.that("the warning updates live as the number is edited", not warning.is_visible())
    page.fill("#field-1-checkNumber", "2683")
    checks.that("and comes back", warning.is_visible())

    page.fill("#email-date-input", "9/12/2026")
    suggestion = page.locator(".review-row >> nth=0 >> .email-date-suggestion")
    checks.that(f"a blank date offers the email date ({suggestion.inner_text()!r})", suggestion.is_visible() and suggestion.inner_text() == "Use 2026-09-12")
    checks.that("a filled date does not", not page.locator(".review-row >> nth=1 >> .email-date-suggestion").is_visible())
    suggestion.click()
    checks.that("one click fills the date", row_fields(page, 0)["date"]["value"] == "2026-09-12" and not suggestion.is_visible())


def exercise_settings(page, checks: CheckList) -> None:
    """Date display, column order and inclusion, payee re-gating, and the settings screenshot."""
    page.click("#settings-button")
    checks.that("Settings opens inline (not a dialog)", page.locator("#settings-panel").is_visible() and page.locator("#settings-panel[role=dialog]").count() == 0)
    page.check("input[name=date-display-format][value='m/d/yyyy']")
    checks.that("the date shows as M/D/YYYY", page.locator("#field-1-date").input_value() == "9/12/2026")
    page.locator(".review-row >> nth=0 >> .copy-icon-button >> nth=2").click()
    checks.that("the date still copies as ISO", clipboard_text(page) == "2026-09-12")

    page.locator(".copy-column[data-column-key=amount]").drag_to(page.locator(".copy-column[data-column-key=date]"), target_position={"x": 20, "y": 3})
    order = page.evaluate("() => [...document.querySelectorAll('.copy-column')].map((item) => item.dataset.columnKey)")
    checks.that(f"dragging moves Amount to the top ({order})", order[:3] == ["amount", "date", "payer"])
    page.uncheck(".copy-column[data-column-key=memo] input")
    page.check(".copy-column[data-column-key=payee] input")
    page.fill("#add-payee-input", "Oak Street Co-op")
    page.press("#add-payee-input", "Enter")
    payee = row_fields(page, 2)["payee"]
    checks.that("adding a payee re-gates the untouched payee field", payee["value"] == "Oak Street Co-op" and payee["state"] == "unsure")
    page.locator("#settings-panel").screenshot(path=str(M4_SCREENSHOT_DIRECTORY / "settings_panel.png"))
    stored = json.loads(app_storage(page).get(STORAGE_PREFIX + "settings", "{}"))
    checks.that("the column settings are saved", [column["key"] for column in stored.get("copyColumns", [])][:2] == ["amount", "date"])

    page.locator(".review-row >> nth=0 >> text=Copy row").click()
    checks.that("Copy row follows the column settings (Amount, Date, Payer, Check number, Payee)",
                clipboard_text(page) == "453.00\t2026-09-12\tLauren Copeland\t2683\t")
    page.click("#copy-all-top-button")
    lines = clipboard_text(page).split("\n")
    checks.that("Copy all rows follows them too", lines[2] == "95.00\t\tMaria Lopez\t301\tOak Street Co-op")
    page.check("input[name=date-display-format][value='iso']")
    page.uncheck(".copy-column[data-column-key=payee] input")
    page.check(".copy-column[data-column-key=memo] input")
    page.locator(".copy-column[data-column-key=amount]").drag_to(page.locator(".copy-column[data-column-key=checkNumber]"), target_position={"x": 20, "y": 3})
    order = page.evaluate("() => [...document.querySelectorAll('.copy-column')].map((item) => item.dataset.columnKey)")
    checks.that(f"columns back to the default order ({order})", order == ["date", "payer", "amount", "checkNumber", "memo", "payee"])
    page.click("#settings-button")


def exercise_rotate_lightbox_and_finish(page, checks: CheckList, check_count: int) -> None:
    """Rotate (boxes flip), MICR toggle, lightbox, guard, Copy all rows, Finish batch, storage."""
    box_before = row_fields(page, 1)["payer"]["box"]
    crop_width, crop_height = describe_page_state(page)["rows"][1]["cropSize"]
    crop_before = page.evaluate("() => document.querySelector('.review-crop-canvas').toDataURL().length")
    page.locator(".review-row >> nth=1 >> text=Rotate").click()
    box_after = row_fields(page, 1)["payer"]["box"]
    checks.that("Rotate turns the read boxes with the crop",
                box_after == [crop_width - box_before[2], crop_height - box_before[3], crop_width - box_before[0], crop_height - box_before[1]])
    checks.that("Rotate keeps the operator's fields", row_fields(page, 1)["amount"]["value"] == "1830.00")
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

    page.evaluate("() => document.querySelectorAll('.review-row input[type=checkbox]').forEach((box) => { if (box.checked) box.click(); })")
    guard_fires = page.evaluate("() => { const event = new Event('beforeunload', { cancelable: true }); window.dispatchEvent(event); return event.defaultPrevented; }")
    checks.that("leaving with uncopied rows triggers the 'leave page?' guard", guard_fires)
    page.click("#copy-all-bottom-button")
    copied_lines = clipboard_text(page).split("\n")
    checks.that("Copy all rows copies one line per check in count order, default columns",
                len(copied_lines) == check_count and copied_lines[3] == "2025-04-28\tDana Whitfield\t1830.00\t2683\tSept rent")
    checks.that("the toast confirms the copy", page.locator("#toast").inner_text() == f"{check_count} rows copied")
    guard_after_copy = page.evaluate("() => { const event = new Event('beforeunload', { cancelable: true }); window.dispatchEvent(event); return event.defaultPrevented; }")
    checks.that("no guard once every row is copied", not guard_after_copy)

    page.click("#finish-batch-button")
    checks.that("Finish batch returns to the empty drop zone", page.locator("#drop-zone-prompt").is_visible())
    checks.that(f"'{check_count} checks recorded' stays visible", page.locator("#batch-recorded-line").inner_text() == f"{check_count} checks recorded")
    checks.that("no check image remains in the page", page.evaluate("() => [...document.querySelectorAll('canvas')].every((canvas) => canvas.width === 0)"))
    storage = app_storage(page)
    names = json.loads(storage.get(STORAGE_PREFIX + "knownPayerNames", "[]"))
    checks.that(f"Finish batch remembers the batch's payer names ({names})", "Dana Whitfield" in names and names.count("Lauren Copeland") == 1)
    history = json.loads(storage.get(STORAGE_PREFIX + "checkHistory", "[]"))
    today_iso = TODAY.isoformat()
    checks.that("and each check number with payer, date and today's batch date",
                {"checkNumber": "2683", "payer": "Dana Whitfield", "date": "2025-04-28", "batchDate": today_iso} in history
                and {"checkNumber": "301", "payer": "Maria Lopez", "batchDate": today_iso} in history)
    checks.that("storage holds text only (no image data, every value small)",
                all("data:image" not in value and len(value) < 20_000 for value in storage.values()))


def exercise_clear_everything(page, checks: CheckList) -> None:
    """"Clear everything this page remembers" wipes every key this app owns, after a confirm."""
    page.evaluate("() => localStorage.setItem('someOtherSite.value', 'untouched')")
    page.once("dialog", lambda dialog: dialog.accept())
    page.click("#settings-button")
    page.click("#clear-everything-button")
    checks.that("Clear everything empties the app's localStorage keys", app_storage(page) == {})
    checks.that("and leaves other keys alone", page.evaluate("() => localStorage.getItem('someOtherSite.value')") == "untouched")
    checks.that("the settings panel shows no remembered names", page.locator("#known-payers-list li").count() == 0)


def exercise_review_grid(page, checks: CheckList, check_count: int) -> None:
    """Every review-grid check, in an order where each step's inputs are known."""
    page.click("#count-continue-button")
    wait_for_debug_state(page, "state.timings.gridCompleteAt > state.timings.continuedAt", TIMEOUT_MS)
    checks.that("one review row per confirmed check", page.locator(".review-row").count() == check_count)
    for row_index, raw_reads in enumerate([ROW_1_READS, ROW_2_READS, ROW_3_READS, ROW_4_READS]):
        page.evaluate("([index, reads]) => window.__checkTranscriberDebug.setFieldReads(index, reads)", [row_index, raw_reads])
    exercise_milestone_3_typing(page, checks)
    exercise_field_states(page, checks)
    exercise_keyboard_flow(page, checks)
    exercise_autocomplete_and_warnings(page, checks)
    exercise_settings(page, checks)
    exercise_rotate_lightbox_and_finish(page, checks, check_count)
    exercise_clear_everything(page, checks)


def main() -> None:
    """Run the flow and report."""
    SCREENSHOT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    M4_SCREENSHOT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    checks = CheckList()
    with serve_directory() as base_url, sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context(viewport={"width": 1400, "height": 900})
        context.grant_permissions(["clipboard-read", "clipboard-write"])
        page = context.new_page()
        page.on("pageerror", lambda error: checks.that(f"no page error ({error})", False))
        page.goto(base_url)
        page.evaluate("([prefix, names, history]) => { localStorage.setItem(prefix + 'knownPayerNames', JSON.stringify(names)); localStorage.setItem(prefix + 'checkHistory', JSON.stringify(history)); }",
                      [STORAGE_PREFIX, SEEDED_PAYER_NAMES, SEEDED_CHECK_HISTORY])
        wait_for_engines_ready(page, TIMEOUT_MS)
        paste_image_file(page, SCENE_IMAGE_PATH)
        check_count = exercise_count_step(page, checks)
        exercise_review_grid(page, checks, check_count)
        browser.close()
    print(f"\nscreenshots: {SCREENSHOT_DIRECTORY}/count_step.png, review_grid.png, lightbox.png, lightbox_zoomed.png")
    print(f"             {M4_SCREENSHOT_DIRECTORY}/grid_mixed_states.png, inline_magnifier.png, settings_panel.png")
    if checks.failures:
        print(f"{len(checks.failures)} check(s) failed")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
