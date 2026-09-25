# app/ — Check Transcriber static site

This directory is the entire deployed artifact: plain HTML/CSS/ES-module JS, no framework, no bundler, no build step. GitHub Pages serves this directory (via `.github/workflows/deploy-pages.yml` at the repo root) exactly as it sits on disk, so what you see here is what ships. Full product spec: `docs/SPEC.md` at the repo root.

## Module map

| File | Owns |
|---|---|
| `index.html` | Page shell, the Content-Security-Policy meta tag, DOM structure for both drop-zone states. |
| `js/main.js` | Wiring: grabs DOM elements, starts engine loading, hooks up the input doors, drives the decode-and-preview flow, owns the one piece of app state (`currentPhoto`). |
| `js/input_doors.js` | The three ways a photo arrives — paste, drag-and-drop, click-to-browse — and nothing else. Doesn't know about HEIC or decoding. |
| `js/heic_detect.js` | HEIC/HEIF detection by extension and by container magic bytes. Pure function, no DOM. |
| `js/image_decode.js` | EXIF-oriented decode into a full-res canvas and a downscaled working canvas. The only module that touches a raw File/Blob. |
| `js/engine_loader.js` | Loads OpenCV.js and Tesseract.js from the CDN, proves both are ready. **Read its module docstring before touching this file** — the callback-only style is a fix for a real browser hang, not incidental. |
| `js/status_line.js` | The one DOM-facing class for the engine-readiness line (loading / ready / failed+retry). |
| `js/cdn_config.js` | Every third-party URL this app fetches, in one place. The CSP in `index.html` is only as tight as this file is honest — if you add a CDN file, pin it here first, then widen the CSP by exactly one line. |
| `sw.js` | Service worker: app-shell + CDN cache-first, offline after first visit. Classic (non-module) worker, deliberately. |

## Adding the next milestone's stage (detection, rectification, OCR)

`js/main.js` holds the decoded photo in a module-level `currentPhoto` object (`{ fullResCanvas, workingCanvas }`), populated by `showDecodedPhoto()`. The comment directly above that object is the named hook: milestone 2's contour-finding stage (spec section 5, stage 2) reads `currentPhoto.workingCanvas`, and later stages read `currentPhoto.fullResCanvas` for full-resolution crops. Don't re-invent a second decode path — everything downstream of "a photo arrived" should consume `currentPhoto`.

The `cv` and `tesseractWorker` handles from `engine_loader.js`'s `onReady` callback are the live engine instances every later stage reuses — do not call `loadProcessingEngines` a second time or load the CDN scripts again; the module already memoizes this, but a fresh caller should still use the handles it's given rather than re-deriving them.

Per spec section 5, the actual pixel-crunching (contour finding, perspective warp, OCR) is supposed to run in a **Web Worker**, not the main thread — milestone 1 doesn't have a worker yet because there's no processing yet, but `worker-src 'self'` is already in the CSP for exactly this reason. When you add it, it's a same-origin worker script (not the CDN's), and OpenCV/Tesseract's own WASM works fine inside a worker context.

## CSP invariants — read before touching the meta tag in index.html

The Content-Security-Policy is `default-src 'none'` plus a short allow-list. Every directive is commented in place in `index.html` with why it exists; the two non-obvious ones:

- **`'unsafe-eval'` in `script-src` is required, not optional.** The pinned OpenCV.js build's Emscripten embind layer uses `new Function(...)` to build dynamic-call trampolines at load time. `'wasm-unsafe-eval'` alone is not enough — confirmed by testing the exact pinned build under both. If a future OpenCV.js version drops this requirement, you can tighten the CSP then, but verify against the real build first.
- **`data:` in `connect-src`** is for OpenCV.js's own WASM binary, which is embedded inline as a base64 `data:` URI and loaded via `fetch()` — not for anything this app's own code does.

Adding a second CDN origin means adding it to `js/cdn_config.js` first (with the same "fetched with curl and confirmed 200" discipline as the existing entries), then widening `script-src`/`connect-src` by exactly that one origin — never widen to a wildcard.

## Service worker invariants

`sw.js` is a classic (non-module) worker with two caches: the app shell (this app's own files) and the CDN cache (whatever gets fetched from `cdn.jsdelivr.net`). Both are cache-first, versioned by `SW_VERSION`. If you change any file in `app/` and want returning visitors to get the new version, bump `SW_VERSION` — that's what forces every client to fetch fresh copies on their next visit. The `APP_SHELL_PATHS` list in `sw.js` is a manually-kept-in-sync mirror of the files this app actually ships (it can't `import` `cdn_config.js`'s origin constant, since classic workers don't support ES module imports); if you add a new `.js`/`.css` file, add it there too.

## Product invariants (from the spec — do not relax these while iterating)

- **Never say "AI" or "model" anywhere in the UI.** The team is wary of AI; the tool describes itself as a check scanner. This is a hard product constraint, not a style preference — grep the UI copy for these words before shipping any milestone.
- **Never extract, store, or display the MICR line** (the machine-printed routing/account numbers along the bottom edge of a check) beyond what's needed to find the check number. Every displayed crop blurs that band by default. This isn't implemented yet (no crops exist until milestone 3) — it's recorded here so whoever builds the crop/review-grid stage doesn't have to rediscover it.
- **Nothing leaves the laptop.** The CSP is the technical enforcement of this; the trust sentence on the page ("Photos stay on this computer. Nothing is uploaded.") must stay literally true. Any change that would make it false is out of scope for this project, full stop.
- **Photo pixels are memory-only.** `image_decode.js` never writes the original File/Blob or the decoded canvases to `localStorage`/`IndexedDB`. Only text (confirmed payer names, settings, etc. — none of which exist yet in milestone 1) persists locally.

## Local development

```
cd app
python3 -m http.server 8000
```
Then open `http://localhost:8000/`. This is also exactly what `tests/test_smoke.py` does under the hood, so "does it work locally" and "does the test pass" are the same question.

## Testing

`tests/test_smoke.py` is a Playwright smoke test — see its own module docstring for what it checks and why it's a plain script rather than pytest. Run it from `app/`:

```
python3 tests/test_smoke.py
```

Needs `playwright` (`pip install playwright && playwright install chromium`). `tests/generate_fixtures.py` regenerates the two test fixtures (an EXIF-rotated JPEG, a fake-HEIC-by-magic-bytes file) if you ever need to recreate `tests/fixtures/`.
