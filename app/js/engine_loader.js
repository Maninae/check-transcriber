/**
 * Loads OpenCV.js and Tesseract.js from the CDN pinned in cdn_config.js, and proves
 * both are actually usable rather than just "downloaded": OpenCV by reading back a
 * real build string from the compiled WASM module, Tesseract by spinning up a worker
 * with the English language data loaded. Milestone 1 stops here — nothing in this
 * app calls into either engine yet. The next milestone's detection stage (contour
 * finding on the working canvas from image_decode.js) is what will actually call
 * `cv.Mat`, `cv.findContours`, etc., and will reuse the `cv` object this module hands
 * back rather than loading OpenCV a second time.
 *
 * CALLBACK STYLE IS LOAD-BEARING, NOT A STYLE CHOICE — READ BEFORE "CLEANING THIS UP":
 * every function below takes `(onReady, onError)` callbacks instead of returning a
 * Promise you `await`. This is because `await`-ing (or `Promise.all`-ing-and-awaiting)
 * anything whose resolution traces back to the pinned OpenCV.js build's `cv` global
 * pegs Chromium's main thread at 100% CPU forever — confirmed by bisection: `cv.then
 * (resolve, reject)` resolves in well under a second, but `await cv` (or `await`ing
 * ANY promise, at ANY nesting depth, whose settlement depends on that same `cv.then`
 * call firing) never returns, even wrapped in a plain native Promise. `cv` is a
 * non-native thenable (it comes back from Emscripten's factory call, not a real
 * `Promise`), and something about how V8 schedules the continuation after an `await`
 * on it never drains. Plain `.then(onSuccess, onError)` callback chaining sidesteps
 * whatever that is entirely. Do not reintroduce `async`/`await` on this path without
 * re-verifying against a real browser first — see the milestone 1 build report for
 * the full bisection.
 */

import {
  OPENCV_JS_URL,
  TESSERACT_JS_URL,
  TESSERACT_WORKER_URL,
  TESSERACT_CORE_PATH,
  TESSERACT_LANG_PATH,
  TOTAL_DOWNLOAD_SIZE_MB,
} from "./cdn_config.js";

export const READINESS_MESSAGE =
  `Getting ready for the first time (about ${TOTAL_DOWNLOAD_SIZE_MB} MB, once).`;
export const FAILURE_MESSAGE =
  "Couldn't load the check-reading tools. Check your connection and try again.";

// Tesseract's OEM (OCR Engine Mode) constant for "LSTM neural net only, no legacy
// engine" — the smaller, faster model, matching Tesseract.js's own default and its
// documented browser usage example.
const TESSERACT_OEM_LSTM_ONLY = 1;

function loadScriptTag(url, onLoad, onError) {
  const script = document.createElement("script");
  script.src = url;
  script.onload = onLoad;
  script.onerror = () => onError(new Error(`failed to load ${url}`));
  document.head.appendChild(script);
}

function loadOpenCv(onReady, onError) {
  loadScriptTag(
    OPENCV_JS_URL,
    () => window.cv.then(onReady, onError),
    onError,
  );
}

function loadTesseractWorker(onReady, onError) {
  loadScriptTag(
    TESSERACT_JS_URL,
    () => {
      // corePath/langPath are directories, not files: Tesseract feature-detects
      // SIMD support and picks the matching core file and traineddata itself.
      window.Tesseract.createWorker("eng", TESSERACT_OEM_LSTM_ONLY, {
        corePath: TESSERACT_CORE_PATH,
        workerPath: TESSERACT_WORKER_URL,
        langPath: TESSERACT_LANG_PATH,
      }).then(onReady, onError);
    },
    onError,
  );
}

// Memoizes the in-flight/completed load so a Retry click (or any other caller)
// never re-inserts a second copy of the 10 MB OpenCV script tag while the first is
// still loading. Cleared on failure so Retry genuinely retries from scratch.
let pendingCallbacks = null;
let completedEngines = null;

function settleAllPending(engines, error) {
  const callbacks = pendingCallbacks;
  pendingCallbacks = null;
  if (!callbacks) {
    return;
  }
  for (const { onReady, onError } of callbacks) {
    if (error) {
      onError(error);
    } else {
      onReady(engines);
    }
  }
}

/**
 * Loads both engines in parallel and calls `onReady({ cv, tesseractWorker,
 * openCvBuildInfo })` once both are genuinely ready, or `onError(error)` if either
 * fails (offline, CDN outage, etc.) — the caller is expected to offer a Retry, per
 * the spec's first-visit requirements. `openCvBuildInfo` is only included so the
 * caller (and the Playwright test) has independent proof OpenCV really compiled,
 * beyond "the load succeeded."
 */
export function loadProcessingEngines(onReady, onError) {
  if (completedEngines) {
    onReady(completedEngines);
    return;
  }
  if (pendingCallbacks) {
    pendingCallbacks.push({ onReady, onError });
    return;
  }
  pendingCallbacks = [{ onReady, onError }];

  let cvResult = null;
  let tesseractWorkerResult = null;

  function checkBothReady() {
    if (cvResult && tesseractWorkerResult) {
      completedEngines = {
        cv: cvResult,
        tesseractWorker: tesseractWorkerResult,
        openCvBuildInfo: cvResult.getBuildInformation(),
      };
      settleAllPending(completedEngines, null);
    }
  }

  function fail(error) {
    settleAllPending(null, error);
  }

  loadOpenCv((cv) => { cvResult = cv; checkBothReady(); }, fail);
  loadTesseractWorker((worker) => { tesseractWorkerResult = worker; checkBothReady(); }, fail);
}
