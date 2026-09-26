/**
 * Starts the check-processing engines and proves they are usable, not just downloaded:
 * - the pipeline worker (js/pipeline/), which loads OpenCV.js and onnxruntime-web inside
 *   itself, compiles OpenCV, creates the orientation classifier session, and reports
 *   OpenCV's real build string back;
 * - Tesseract.js on the main thread, with the English data loaded (field reading arrives
 *   in milestone 4 and will reuse this worker).
 * The page never touches OpenCV directly; all pixel work happens in the pipeline worker.
 *
 * CALLBACK STYLE IS LOAD-BEARING, NOT A STYLE CHOICE — READ BEFORE "CLEANING THIS UP":
 * every function below takes `(onReady, onError)` callbacks instead of returning a
 * Promise you `await`. Awaiting (or `Promise.all`-ing) anything whose resolution traces
 * back to the pinned OpenCV.js build's `cv` global pegged Chromium at 100% CPU forever
 * in milestone 1 — `cv.then(resolve, reject)` resolves in well under a second, but
 * `await cv` never returns. `cv` is a non-native thenable (Emscripten's module object,
 * which itself has a `.then`), so resolving any Promise with it makes the Promise adopt
 * it again, indefinitely. OpenCV now lives in the pipeline worker, which follows the
 * same rule (see pipeline_worker.js); this loader keeps the callback style so the two
 * engines' readiness is still combined without a Promise anywhere near `cv`.
 */

import { PipelineClient } from "./pipeline/pipeline_client.js";
import {
  OPENCV_JS_URL,
  ONNX_RUNTIME_JS_URL,
  ONNX_RUNTIME_WASM_DIRECTORY,
  HANDWRITING_READER_URLS,
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

/** Starts the pipeline worker; `onReady({ pipelineClient, openCvBuildInfo })`. */
function startPipelineWorker(onReady, onError) {
  const pipelineClient = new PipelineClient({
    openCvUrl: OPENCV_JS_URL,
    onnxRuntimeJsUrl: ONNX_RUNTIME_JS_URL,
    onnxRuntimeWasmDirectory: ONNX_RUNTIME_WASM_DIRECTORY,
    handwritingReaderUrls: HANDWRITING_READER_URLS,
  });
  pipelineClient.start(
    ({ openCvBuildInfo }) => onReady({ pipelineClient, openCvBuildInfo }),
    (error) => {
      pipelineClient.terminate();
      onError(error);
    },
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
// never starts a second pipeline worker (and a second OpenCV download) while the
// first is still loading. Cleared on failure so Retry genuinely retries from scratch.
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
 * Loads both engines in parallel and calls `onReady({ pipelineClient, tesseractWorker,
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

  let pipelineResult = null;
  let tesseractWorkerResult = null;

  function checkBothReady() {
    if (pipelineResult && tesseractWorkerResult) {
      completedEngines = {
        pipelineClient: pipelineResult.pipelineClient,
        tesseractWorker: tesseractWorkerResult,
        openCvBuildInfo: pipelineResult.openCvBuildInfo,
      };
      settleAllPending(completedEngines, null);
    }
  }

  function fail(error) {
    // A half-started pipeline worker would otherwise leak on Retry.
    if (pipelineResult) pipelineResult.pipelineClient.terminate();
    pipelineResult = null;
    settleAllPending(null, error);
  }

  startPipelineWorker((pipeline) => { pipelineResult = pipeline; checkBothReady(); }, fail);
  loadTesseractWorker((worker) => { tesseractWorkerResult = worker; checkBothReady(); }, fail);
}
