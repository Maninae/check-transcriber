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

// OpenCV compiled to WebAssembly (stages 2-4 of the processing pipeline: detection,
// corner refinement, rectification). This is the official OpenCV.js build published
// at docs.opencv.org, republished verbatim to npm by the @techstark/opencv-js package.
// Single file: the WASM binary is embedded inline as base64, so there is no second
// network request for a .wasm file. Pinned to 5.0.0 because it is the same OpenCV the
// Python detector was built and scored on (experiments/detection/), and because the
// 4.10 build lacks approxPolyN / intersectConvexConvex / sqrBoxFilter, which the
// detector needs. Loaded only inside the pipeline worker (js/pipeline/).
export const OPENCV_JS_URL =
  `${CDN_ORIGIN}/npm/@techstark/opencv-js@5.0.0-release.1/dist/opencv.js`;

// onnxruntime-web, WASM backend only (no WebGPU): runs the small upside-down check
// classifier shipped in models/. `ort.wasm.min.js` is the UMD entry that
// importScripts() can load in the classic pipeline worker; at session creation it
// dynamic-imports the .mjs glue and fetches the .wasm from ONNX_RUNTIME_WASM_DIRECTORY.
export const ONNX_RUNTIME_VERSION = "1.30.0";
export const ONNX_RUNTIME_JS_URL =
  `${CDN_ORIGIN}/npm/onnxruntime-web@${ONNX_RUNTIME_VERSION}/dist/ort.wasm.min.js`;
export const ONNX_RUNTIME_WASM_DIRECTORY =
  `${CDN_ORIGIN}/npm/onnxruntime-web@${ONNX_RUNTIME_VERSION}/dist/`;

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

// First-visit download sizes, as transferred over the wire (jsDelivr serves brotli;
// measured with `curl -H 'Accept-Encoding: br' <url> | wc -c`, Sep 2026). Shown in the
// readiness line so the operator knows what "getting ready" means and that it only
// happens once.
const FIRST_VISIT_TRANSFER_SIZES_MB = {
  openCvJs: 3.5,
  onnxRuntimeJsAndWasm: 3.2,
  orientationClassifier: 1.9, // models/upside_down_classifier.onnx, served from our own host
  tesseractJs: 0.2, // tesseract.min.js + worker.min.js
  tesseractCore: 3.9, // the SIMD or non-SIMD core, a few hundred KB apart
  tesseractEnglishData: 2.95, // English LSTM traineddata, gzipped
};
export const TOTAL_DOWNLOAD_SIZE_MB = Math.round(
  Object.values(FIRST_VISIT_TRANSFER_SIZES_MB).reduce((total, sizeMb) => total + sizeMb, 0),
);
