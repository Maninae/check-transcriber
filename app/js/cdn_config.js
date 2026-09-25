/**
 * Every third-party URL this app ever fetches, in one place.
 *
 * This is the single source of truth for the app's one CDN origin (jsDelivr). The
 * Content-Security-Policy in index.html locks script/connect access to exactly this
 * origin, so a future edit that reaches for a second CDN or a random API will fail
 * closed with a console CSP violation instead of silently phoning home. Keep it that
 * way: if a later milestone needs another file, pin it here and widen the CSP by one
 * line, deliberately, rather than adding an ad hoc URL somewhere else in the code.
 *
 * Versions are pinned exactly (no "@latest", no unpinned "@major") so a CDN-side
 * release can't change behavior under us. Every URL here was fetched with `curl` and
 * confirmed to return HTTP 200 before being pinned (see the milestone 1 build report).
 */

// jsDelivr is the one CDN origin this app is allowed to talk to. Every URL below
// must be a subpath of this origin, or the CSP in index.html will block it.
export const CDN_ORIGIN = "https://cdn.jsdelivr.net";

// OpenCV compiled to WebAssembly (stages 2-4 of the processing pipeline, added in a
// later milestone). This is the official OpenCV.js build published at
// docs.opencv.org, republished verbatim to npm by the @techstark/opencv-js package.
// Single file: the WASM binary is embedded inline as base64, so there is no second
// network request for a .wasm file.
export const OPENCV_JS_URL =
  `${CDN_ORIGIN}/npm/@techstark/opencv-js@4.10.0-release.1/dist/opencv.js`;
export const OPENCV_JS_SIZE_MB = 10.4;

// Tesseract.js: the main-thread API and the Web Worker it spawns to run OCR off the
// UI thread. `TESSERACT_CORE_PATH` is deliberately a directory, not a file: Tesseract
// feature-detects SIMD support in the browser and appends the right filename itself
// (tesseract-core-simd-lstm.wasm.js and friends). `TESSERACT_LANG_PATH` is likewise a
// directory that Tesseract appends "/eng.traineddata.gz" to.
export const TESSERACT_JS_URL =
  `${CDN_ORIGIN}/npm/tesseract.js@7.0.0/dist/tesseract.min.js`;
export const TESSERACT_WORKER_URL =
  `${CDN_ORIGIN}/npm/tesseract.js@7.0.0/dist/worker.min.js`;
export const TESSERACT_CORE_PATH =
  `${CDN_ORIGIN}/npm/tesseract.js-core@7.0.0`;
export const TESSERACT_LANG_PATH =
  `${CDN_ORIGIN}/npm/@tesseract.js-data/eng@1.0.0/4.0.0_best_int`;
// English LSTM-only traineddata, gzipped, gzip is Tesseract's default and this is
// the size actually transferred over the wire.
export const TESSERACT_LANG_SIZE_MB = 2.95;
// The core WASM file Tesseract picks depends on the browser's SIMD support; both
// candidates are within a few hundred KB of each other, so one estimate covers both.
export const TESSERACT_CORE_SIZE_MB = 3.9;
export const TESSERACT_JS_SIZE_MB = 0.2; // tesseract.min.js + worker.min.js combined

// Total first-visit download across every CDN file, shown in the readiness line so
// Angie knows what "getting ready" means and that it only happens once.
export const TOTAL_DOWNLOAD_SIZE_MB = Math.round(
  OPENCV_JS_SIZE_MB + TESSERACT_JS_SIZE_MB + TESSERACT_CORE_SIZE_MB + TESSERACT_LANG_SIZE_MB,
);
