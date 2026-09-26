# app/ — Check Transcriber static site

This directory is the entire deployed artifact: plain HTML/CSS/ES-module JS, no framework, no bundler, no build step. GitHub Pages serves this directory (via `.github/workflows/deploy-pages.yml` at the repo root) exactly as it sits on disk, so what you see here is what ships. Full product spec: `docs/SPEC.md` at the repo root.

## Module map

| Path | Owns |
|---|---|
| `index.html` | Page shell, the Content-Security-Policy meta tag, DOM for every step (drop zone, count step, review grid, lightbox). |
| `js/main.js` | Wiring: DOM lookups, input doors, engine start, builds the count step / review grid / lightbox / batch flow, the `window.__checkTranscriberDebug` test hook. |
| `js/batch_flow.js` | One batch start to finish: photo → worker detection → count step → worker crops → review grid → Finish batch; the "leave page?" guard; timings. |
| `js/pipeline/` | Everything that touches pixels, in a Web Worker (detection, corner refinement, orientation, rectification). **Read its CLAUDE.md.** |
| `js/count/` | The count step (spec 4.2): `count_step.js` state + pointer/keyboard, `count_overlay.js` SVG drawing, `count_header_text.js` header copy. |
| `js/review/` | The review grid (spec 4.3/4.5): `review_grid.js` row state, `review_row.js` DOM, `crop_rendering.js` rotation + MICR blur, `clipboard_rows.js` TSV, `field_undo.js` per-field Ctrl+Z, `field_definitions.js` fields and copy order, `lightbox.js` (spec 4.4). |
| `js/step_indicator.js`, `js/toast.js`, `js/status_line.js` | Small DOM leaves: the step line, the "6 rows copied" toast, the engine readiness line. |
| `js/input_doors.js` | The three ways a photo arrives — paste, drag-and-drop, click-to-browse — and nothing else. |
| `js/heic_detect.js` | HEIC/HEIF detection by extension and by container magic bytes. |
| `js/image_decode.js` | EXIF-oriented decode into a full-res canvas. The only module that touches a raw File/Blob. |
| `js/engine_loader.js` | Starts the pipeline worker (OpenCV.js + onnxruntime-web load inside it) and Tesseract.js, proves both ready. **Read its module docstring** — callback style is load-bearing. |
| `js/cdn_config.js` | Every third-party URL, pinned, plus first-visit transfer sizes. |
| `models/upside_down_classifier.onnx` | Our own 1.9 MB orientation classifier (exported by `experiments/detection/export/export_orientation_onnx.py`, parity-checked). |
| `sw.js` | Service worker: app-shell + CDN cache-first, offline after first visit. |
| `styles/` | One stylesheet per screen; `base.css` holds the palette, type and shared buttons. |

## Visual hierarchy

The photo (count step) and the crop (review grid) are the payload: the largest things on the page. Chrome stays small and gray. The accent green marks only the primary action (Continue, Copy all rows) and confident outlines; amber marks outlines worth a look. Corner handles appear on hover or selection only.

## Adding field reading (milestone 4)

The review grid already has the hooks: `ReviewGrid.setCrop` receives each upright 1600 px crop; `rotateCheck` is where a re-read goes; `field_undo.js`'s `setValue` makes a script-set value undoable (the payer snap). The Tesseract worker from `engine_loader.js` is the live instance to reuse (`engines.tesseractWorker`); do not load it twice. The MICR band must never become a field (spec section 7).

## CSP invariants — read before touching the meta tag in index.html

The Content-Security-Policy is `default-src 'none'` plus a short allow-list. Every directive is commented in place in `index.html` with why it exists; the two non-obvious ones:

- **`'unsafe-eval'` in `script-src` is required, not optional.** The pinned OpenCV.js build's Emscripten embind layer uses `new Function(...)` to build dynamic-call trampolines at load time. `'wasm-unsafe-eval'` alone is not enough — confirmed by testing the exact pinned build under both. If a future OpenCV.js version drops this requirement, you can tighten the CSP then, but verify against the real build first.
- **`data:` in `connect-src`** is for OpenCV.js's own WASM binary, which is embedded inline as a base64 `data:` URI and loaded via `fetch()` — not for anything this app's own code does.
- **The CSP covers the pipeline worker only because it starts from a blob: bootstrap** (see `js/pipeline/CLAUDE.md`). A worker loaded from its own URL would get no CSP on GitHub Pages.

Adding a second CDN origin means adding it to `js/cdn_config.js` first (with the same "fetched with curl and confirmed 200" discipline as the existing entries), then widening `script-src`/`connect-src` by exactly that one origin — never widen to a wildcard.

## Service worker invariants

`sw.js` is a classic (non-module) worker with two caches: the app shell (this app's own files and the classifier model) and the CDN cache (whatever gets fetched from `cdn.jsdelivr.net`, including the pipeline worker's `importScripts` and onnxruntime's `.mjs`/`.wasm`). Both are cache-first, versioned by `SW_VERSION`. If you change any file in `app/` and want returning visitors to get the new version, bump `SW_VERSION`. `APP_SHELL_PATHS` is generated: after adding or removing a file run `python3 tests/sync_service_worker_shell_list.py` (the smoke test fails while it is stale).

## Product invariants (from the spec — do not relax these while iterating)

- **Never say "AI" or "model" anywhere in the UI.** The team is wary of AI; the tool describes itself as a check scanner. This is a hard product constraint, not a style preference — grep the UI copy for these words before shipping any milestone.
- **Never extract, store, or display the MICR line** (the machine-printed routing/account numbers along the bottom edge of a check) beyond what's needed to find the check number. Every displayed crop (row and lightbox) blurs the bottom 23% by default (`js/review/crop_rendering.js`), and the top band too when the orientation call was close; the per-row "Show bottom line" toggle resets with the next batch.
- **Nothing leaves the laptop.** The CSP is the technical enforcement of this; the trust sentence on the page ("Photos stay on this computer. Nothing is uploaded.") must stay literally true. Any change that would make it false is out of scope for this project, full stop.
- **Photo pixels are memory-only.** Nothing writes the original File/Blob, the canvases, the worker's Mats or the crops to `localStorage`/`IndexedDB`, and no object URL of a photo is ever made (crops are canvases). Finish batch and Start over release the canvases (width 0) and the worker's copy. Only text will persist (milestone 4).

## Local development

```
cd app
python3 -m http.server 8000
```
Then open `http://localhost:8000/`. This is also exactly what `tests/test_smoke.py` does under the hood, so "does it work locally" and "does the test pass" are the same question.

## Testing

All plain scripts that print what they check and exit non-zero on failure. `VENV=/Volumes/vega/datasets/check-transcriber/venv/bin/python` (has playwright, numpy, cv2, shapely and the repo's `experiments` package).

| Script | Checks | Run |
|---|---|---|
| `tests/test_smoke.py` | Milestone-1 invariants (trust sentence, EXIF, HEIC hint, only localhost + CDN contacted) and the service-worker list | `python3 tests/test_smoke.py` |
| `tests/test_detection_regression.py` | 40 eval scenes pasted into the page, scored with the Python metrics harness next to the Python pipeline on the same scenes | `$VENV tests/test_detection_regression.py` |
| `tests/test_performance_budget.py` | Paste→count and Continue→grid on a 12 MP six-check photo vs the 3 s / 10 s budget | `$VENV tests/test_performance_budget.py` |
| `tests/test_offline_after_first_visit.py` | Offline reload still loads every engine and detects checks | `python3 tests/test_offline_after_first_visit.py` |
| `tests/test_review_flow.py` | Count-step edits, review grid, copy, undo, lightbox, Finish batch; saves screenshots to `/tmp/check-transcriber-m2/` | `$VENV tests/test_review_flow.py` |
| `tests/parity/` | Node vs Python, module by module, on cv2-decoded pixels | see `js/pipeline/CLAUDE.md` |

`tests/browser_test_helpers.py` holds the shared static server, synthetic paste and debug-hook polling. `tests/generate_fixtures.py` regenerates the two milestone-1 fixtures.
