"""Shared Playwright plumbing for the milestone-2/3 browser tests.

- `serve_directory(root)`: a quiet static HTTP server (same as GitHub Pages: files as-is).
- `paste_image_file(page, path)`: fires a real `paste` event carrying the image as a
  clipboard File, the same path the operator's Ctrl+V takes through js/input_doors.js,
  and answers the inline "Replace these checks?" question if the page asks it.
- `wait_for_debug_state(page, predicate_js, timeout_ms)`: polls
  `window.__checkTranscriberDebug.describe()` (quads, step, timings; never pixels).
"""

import base64
import functools
import http.server
import mimetypes
import socket
import threading
from contextlib import closing, contextmanager
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent

PASTE_IMAGE_SCRIPT = """
async ({ base64Bytes, mimeType, fileName, answerReplace }) => {
  const bytes = Uint8Array.from(atob(base64Bytes), (character) => character.charCodeAt(0));
  const file = new File([bytes], fileName, { type: mimeType });
  const clipboardData = new DataTransfer();
  clipboardData.items.add(file);
  document.dispatchEvent(new ClipboardEvent("paste", { clipboardData, bubbles: true }));
  if (!answerReplace) return;
  // A paste over an open batch asks inline ("Replace these 6 checks with the new photo?");
  // answer Replace, as the old window.confirm handlers did. Stop waiting as soon as the page
  // shows it took the photo without asking (the progress panel), or refused it.
  const started = performance.now();
  while (performance.now() - started < 3000) {
    const bar = document.getElementById("replace-confirm");
    if (bar && !bar.hidden) { document.getElementById("replace-confirm-yes").click(); return; }
    const panel = document.getElementById("progress-panel");
    const hint = document.getElementById("input-hint");
    const toast = document.getElementById("toast");
    const panelWorking = panel && !panel.hidden && panel.dataset.tone === "working"; // not a lingering "All N read"
    if (panelWorking || (hint && !hint.hidden) || (toast && !toast.hidden && toast.dataset.tone === "info")) return;
    await new Promise((resolve) => setTimeout(resolve, 15));
  }
}
"""


class QuietRequestHandler(http.server.SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler without a log line per request."""

    def log_message(self, *arguments):
        pass


def find_free_port() -> int:
    """An unused localhost TCP port."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as probe_socket:
        probe_socket.bind(("127.0.0.1", 0))
        return probe_socket.getsockname()[1]


@contextmanager
def serve_directory(root_directory: Path = APP_ROOT):
    """Serves `root_directory` over HTTP on localhost; yields the base URL."""
    port = find_free_port()
    handler = functools.partial(QuietRequestHandler, directory=str(root_directory))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{port}/"
    finally:
        server.shutdown()


def paste_image_file(page, image_path: Path, answer_replace: bool = True) -> None:
    """Pastes `image_path` into the page as the operator's Ctrl+V would.

    With a batch open the page asks inline whether to replace it (js/photo_intake.js);
    `answer_replace` clicks Replace, the way the tests' dialog handlers accepted the old
    window.confirm. Pass False to leave the question open.
    """
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    page.evaluate(
        PASTE_IMAGE_SCRIPT,
        {"base64Bytes": base64.b64encode(image_path.read_bytes()).decode(), "mimeType": mime_type,
         "fileName": image_path.name, "answerReplace": answer_replace},
    )


def describe_page_state(page) -> dict:
    """The debug hook's snapshot of the batch."""
    return page.evaluate("() => window.__checkTranscriberDebug.describe()")


def wait_for_debug_state(page, predicate_js: str, timeout_ms: int) -> dict:
    """Waits until `predicate_js` (a JS expression over `state`) is true; returns the state."""
    page.wait_for_function(
        f"() => {{ const state = window.__checkTranscriberDebug && window.__checkTranscriberDebug.describe(); return state && ({predicate_js}); }}",
        timeout=timeout_ms,
        polling=50,
    )
    return describe_page_state(page)


def wait_for_engines_ready(page, timeout_ms: int) -> None:
    """The readiness line hides itself once OpenCV, onnxruntime and Tesseract are ready."""
    page.wait_for_selector("#engine-status[hidden]", state="attached", timeout=timeout_ms)
