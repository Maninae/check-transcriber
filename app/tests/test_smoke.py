"""Milestone 1 smoke test: serves app/ exactly as GitHub Pages would (a static
directory over HTTP, with app/ as the site root), drives it in real Chromium, and
checks every milestone-1 requirement end to end. Run with `python3 tests/test_smoke.py`
from the app/ directory, or `python3 app/tests/test_smoke.py` from the repo root
(needs `playwright` installed and `playwright install chromium` run once).

This is deliberately a plain script with assertions, not a pytest suite: milestone 1
has one thing to verify (does the skeleton actually work in a browser), and a script
that prints what it checked as it goes is easier to read top to bottom than pytest's
collection/fixture machinery would be for a single run. If a later milestone grows a
real regression suite (per spec section 8's mock-check-set plan), promote this into
a proper pytest suite at that point.

Every wait below has a hard timeout, and console/pageerror/requestfailed are captured
and printed on failure — a milestone 1 run that hangs (see js/engine_loader.js's
"CALLBACK STYLE IS LOAD-BEARING" note for the exact browser quirk that used to cause
this) must fail loudly with a diagnosable log, never spin the runner forever.
"""

import http.server
import functools
import socket
import subprocess
import sys
import threading
import time
from contextlib import closing
from pathlib import Path

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

APP_ROOT = Path(__file__).parent.parent  # app/ — the actual GitHub Pages web root
FIXTURES_DIR = Path(__file__).parent / "fixtures"
SCREENSHOT_DIR = Path("/tmp")

ALLOWED_THIRD_PARTY_ORIGIN = "https://cdn.jsdelivr.net"
PAGE_LOAD_TIMEOUT_MS = 30_000
ENGINE_READY_TIMEOUT_MS = 120_000
SHORT_WAIT_TIMEOUT_MS = 10_000


def find_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def start_static_server(port: int) -> http.server.ThreadingHTTPServer:
    """Serves APP_ROOT over plain HTTP, the same way GitHub Pages serves the app/ subtree."""
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(APP_ROOT))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


class Check:
    """Tiny pass/fail tracker so the script can run every check and report all
    failures at once, instead of stopping at the first assertion like bare `assert`
    would — much cheaper to fix three failures from one run than one at a time."""

    def __init__(self):
        self.failures = []

    def that(self, description: str, condition: bool):
        status = "PASS" if condition else "FAIL"
        print(f"[{status}] {description}", flush=True)
        if not condition:
            self.failures.append(description)

    def finish(self):
        print()
        if self.failures:
            print(f"{len(self.failures)} check(s) failed:")
            for failure in self.failures:
                print(f"  - {failure}")
            sys.exit(1)
        print("All checks passed.")


class DiagnosticLog:
    """Collects console/pageerror/requestfailed events so a failure (especially a
    timeout) has an actual root cause attached instead of just "it didn't happen"."""

    def __init__(self, page):
        self.entries = []
        page.on("console", lambda msg: self.entries.append(f"[console:{msg.type}] {msg.text}"))
        page.on("pageerror", lambda err: self.entries.append(f"[pageerror] {err}"))
        page.on("requestfailed", lambda req: self.entries.append(f"[requestfailed] {req.url} -- {req.failure}"))

    def dump(self, reason: str):
        print(f"\n--- diagnostic log ({reason}) ---")
        for entry in self.entries:
            print(entry)
        print("--- end diagnostic log ---\n")


def wait_or_dump(page, diagnostic_log, selector: str, timeout_ms: int, reason: str, **kwargs):
    """wait_for_selector with a hard timeout that dumps the diagnostic log (instead of
    just raising) before re-raising, so a CI failure log is self-contained."""
    try:
        page.wait_for_selector(selector, timeout=timeout_ms, **kwargs)
    except PlaywrightTimeoutError:
        diagnostic_log.dump(reason)
        raise


def main() -> None:
    check = Check()
    port = find_free_port()
    server = start_static_server(port)
    base_url = f"http://127.0.0.1:{port}/"

    third_party_request_urls = []

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page()
            diagnostic_log = DiagnosticLog(page)

            def record_request(request):
                # blob:/data: URLs are locally-generated in-browser references (e.g. the
                # blob URL Tesseract.js instantiates its worker from — see the
                # "workerBlobURL" note in engine_loader.js's module docstring), never a
                # real network request to a remote server, so they don't count as a
                # third-party origin regardless of what scheme-specific text follows.
                if request.url.startswith(("blob:", "data:")):
                    return
                origin = request.url.split("/")[0] + "//" + request.url.split("/")[2]
                if origin != f"http://127.0.0.1:{port}" and origin != ALLOWED_THIRD_PARTY_ORIGIN:
                    third_party_request_urls.append(request.url)

            page.on("request", record_request)

            page.goto(base_url, timeout=PAGE_LOAD_TIMEOUT_MS)

            # --- Initial page: trust sentence, drop zone, nothing else demanding attention ---
            trust_sentence = page.locator(".trust-sentence")
            check.that(
                "trust sentence is visible on first load",
                trust_sentence.is_visible()
                and trust_sentence.inner_text() == "Photos stay on this computer. Nothing is uploaded.",
            )
            check.that("drop zone prompt is visible on first load", page.locator("#drop-zone-prompt").is_visible())
            check.that(
                "photo preview is hidden on first load", not page.locator("#photo-preview-container").is_visible()
            )

            page.screenshot(path=str(SCREENSHOT_DIR / "check-transcriber-initial.png"))
            print(f"screenshot: {SCREENSHOT_DIR / 'check-transcriber-initial.png'}")

            # --- Readiness line reaches the ready state (OpenCV + Tesseract both loaded) ---
            wait_or_dump(
                page, diagnostic_log, "#engine-status[hidden]", ENGINE_READY_TIMEOUT_MS,
                reason="engine status line never reached ready", state="attached",
            )
            check.that("engine status line hides itself once ready", page.locator("#engine-status").is_hidden())
            build_info_logged = any("OpenCV build info" in entry for entry in diagnostic_log.entries)
            check.that("OpenCV build info was logged (proves cv actually compiled)", build_info_logged)

            # --- Upload path: EXIF-oriented JPEG produces the swapped "Photo received" dimensions ---
            page.set_input_files("#file-input", str(FIXTURES_DIR / "oriented.jpg"))
            wait_or_dump(
                page, diagnostic_log, "#photo-preview-container:not([hidden])", SHORT_WAIT_TIMEOUT_MS,
                reason="photo preview never appeared after uploading oriented.jpg",
            )
            photo_received_line = page.locator("#photo-received-line")
            decoded_size = (photo_received_line.get_attribute("data-photo-width"), photo_received_line.get_attribute("data-photo-height"))
            check.that(
                f"oriented JPEG (400x300 stored, EXIF orientation 6) decodes rotated to 300 x 400 ({decoded_size})",
                photo_received_line.inner_text() == "Photo received" and decoded_size == ("300", "400"),
            )

            page.screenshot(path=str(SCREENSHOT_DIR / "check-transcriber-post-upload.png"))
            print(f"screenshot: {SCREENSHOT_DIR / 'check-transcriber-post-upload.png'}")

            # --- Start Over, then the HEIC path: paste-equivalent via file input, expect the hint ---
            page.on("dialog", lambda dialog: dialog.accept())
            page.click("#start-over-button")
            if page.locator("#replace-confirm-yes").is_visible():  # the inline "start over?" bar, when there is work to lose
                page.click("#replace-confirm-yes")
            wait_or_dump(
                page, diagnostic_log, "#drop-zone-prompt:not([hidden])", SHORT_WAIT_TIMEOUT_MS,
                reason="drop zone prompt never reappeared after Start Over",
            )

            page.set_input_files("#file-input", str(FIXTURES_DIR / "fake.heic"))
            wait_or_dump(
                page, diagnostic_log, "#input-hint-line:not([hidden])", SHORT_WAIT_TIMEOUT_MS,
                reason="HEIC hint never appeared",
            )
            hint_text = page.locator("#input-hint-line").inner_text()
            check.that(
                "fake HEIC (correct magic bytes) shows the HEIC hint",
                "HEIC" in hint_text and "Copy image" in hint_text,
            )

            browser.close()
    finally:
        server.shutdown()

    check.that(
        "no request went to any origin other than localhost and the CDN",
        len(third_party_request_urls) == 0,
    )
    if third_party_request_urls:
        print("  unexpected origins contacted:")
        for url in third_party_request_urls:
            print(f"    - {url}")
    print(f"third-party origins actually contacted: {sorted({u.split('/')[2] for u in third_party_request_urls}) or ['(none)']}, CDN allowed: {ALLOWED_THIRD_PARTY_ORIGIN}")

    shell_list_check = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "sync_service_worker_shell_list.py"), "--check"],
        capture_output=True, text=True,
    )
    check.that("sw.js precaches exactly the files app/ ships (offline after first visit)", shell_list_check.returncode == 0)

    check.finish()


if __name__ == "__main__":
    main()
