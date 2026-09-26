/**
 * The pipeline Web Worker (spec section 5: "all of this runs in a Web Worker so the page
 * never freezes"). Owns OpenCV.js, onnxruntime-web, the classifier session and the
 * current photo's full-resolution pixels; the page only ever sees quads and crops.
 *
 * CLASSIC worker on purpose: OpenCV.js and onnxruntime-web ship as UMD scripts that load
 * with importScripts(), which module workers do not have. Our own stage code is ES
 * modules, pulled in with dynamic import(), which classic workers do support.
 *
 * Started from a blob: URL by pipeline_client.js (`self.PIPELINE_WORKER_URL` is set by
 * that bootstrap). A blob worker inherits the page's Content-Security-Policy; a worker
 * loaded straight from its .js URL would get its policy from HTTP headers instead, and
 * GitHub Pages sends none, so it would run with no CSP at all. Relative URLs don't
 * resolve against a blob: URL, which is why everything here is resolved against
 * PIPELINE_WORKER_URL.
 *
 * OpenCV rule (see js/engine_loader.js): `cv` is a non-native thenable. Only ever use
 * `self.cv.then(callback)`; never await it, return it from a .then callback, or resolve a
 * Promise with it.
 *
 * Protocol: page -> worker `{ type, requestId, ...payload }`; worker -> page
 * `{ type: "result" | "error" | "progress" | "check-ready", requestId, ... }`, plus one
 * `{ type: "ready" | "init-failed" }` after "init".
 */

// Deliberately NOT named `cv` / `ort`: a classic worker's top-level `let` is a global
// lexical binding, and ort.wasm.min.js declares a global `ort`, so a same-named `let`
// makes importScripts fail (Chrome reports it as a NetworkError).
let openCv = null;
let onnxRuntime = null;
let stages = null;
let classifierSession = null;
let currentPhoto = null; // { imageBgr, workingGray, workingScale }

function resolveAgainstWorker(relativePath) {
  return new URL(relativePath, self.PIPELINE_WORKER_URL).href;
}

function postResult(requestId, payload, transfer) {
  self.postMessage({ type: "result", requestId, ...payload }, transfer || []);
}

function initialize(message) {
  try {
    importScripts(message.openCvUrl, message.onnxRuntimeJsUrl);
  } catch (error) {
    self.postMessage({ type: "init-failed", message: `could not load a processing library: ${error}` });
    return;
  }
  onnxRuntime = self.ort;
  onnxRuntime.env.wasm.wasmPaths = message.onnxRuntimeWasmDirectory;
  // Threads need cross-origin isolation (COOP/COEP headers), which GitHub Pages cannot set.
  onnxRuntime.env.wasm.numThreads = 1;
  const fail = (error) => self.postMessage({ type: "init-failed", message: String(error) });
  self.cv.then((readyCv) => {
    openCv = readyCv;
    import(resolveAgainstWorker("./worker_stages.js")).then((stageModule) => {
      stages = stageModule;
      stages.createUpsideDownClassifierSession(onnxRuntime, resolveAgainstWorker(message.classifierModelPath)).then((session) => {
        classifierSession = session;
        self.postMessage({ type: "ready", openCvBuildInfo: openCv.getBuildInformation() });
      }, fail);
    }, fail);
  }, fail);
}

function releasePhoto() {
  if (!currentPhoto) return;
  currentPhoto.imageBgr.delete();
  currentPhoto.workingGray.delete();
  currentPhoto = null;
}

function loadPhoto(message) {
  releasePhoto();
  const imageRgba = new openCv.Mat(message.height, message.width, openCv.CV_8UC4);
  const imageBgr = new openCv.Mat();
  try {
    imageRgba.data.set(new Uint8Array(message.rgbaBuffer));
    openCv.cvtColor(imageRgba, imageBgr, openCv.COLOR_RGBA2BGR);
  } finally {
    imageRgba.delete();
  }
  const { workingGray, workingScale } = stages.buildOrientationWorkingGray(openCv, imageBgr);
  currentPhoto = { imageBgr, workingGray, workingScale };
  postResult(message.requestId, {});
}

function requirePhoto() {
  if (!currentPhoto) throw new Error("no photo is loaded in the pipeline worker");
  return currentPhoto;
}

function detectChecks(message) {
  const onProgress = (text) => self.postMessage({ type: "progress", requestId: message.requestId, text });
  const detections = stages.detectChecksInPhoto(openCv, requirePhoto().imageBgr, onProgress);
  postResult(message.requestId, { detections });
}

function refitDrawnRectangle(message) {
  const refit = stages.refitDrawnRectangle(openCv, requirePhoto().imageBgr, message.drawnCorners, message.otherCornerSets);
  postResult(message.requestId, refit);
}

/** Orients and rectifies the confirmed quads one at a time, streaming each crop back. */
function orientAndRectifyChecks(message) {
  const photo = requirePhoto();
  const processCheck = (checkIndex) => {
    if (checkIndex >= message.cornerSets.length) {
      postResult(message.requestId, {});
      return;
    }
    self.postMessage({ type: "progress", requestId: message.requestId, text: `Straightening check ${checkIndex + 1} of ${message.cornerSets.length}` });
    stages.orientAndRectifyCheck(openCv, onnxRuntime, classifierSession, photo, message.cornerSets[checkIndex]).then((checkResult) => {
      const rgbaBuffer = checkResult.rgbaPixels.buffer;
      self.postMessage({
        type: "check-ready", requestId: message.requestId, checkIndex,
        orientedCorners: checkResult.orientedCorners, upsideDownProbability: checkResult.upsideDownProbability,
        width: checkResult.width, height: checkResult.height, rgbaBuffer,
      }, [rgbaBuffer]);
      processCheck(checkIndex + 1);
    }).catch((error) => self.postMessage({ type: "error", requestId: message.requestId, message: String(error) }));
  };
  processCheck(0);
}

const REQUEST_HANDLERS = {
  "load-photo": loadPhoto,
  "detect-checks": detectChecks,
  "refit-drawn-rectangle": refitDrawnRectangle,
  "orient-and-rectify-checks": orientAndRectifyChecks,
  "release-photo": (message) => { releasePhoto(); postResult(message.requestId, {}); },
};

self.onmessage = (event) => {
  const message = event.data;
  if (message.type === "init") {
    initialize(message);
    return;
  }
  const handler = REQUEST_HANDLERS[message.type];
  try {
    if (!handler) throw new Error(`unknown pipeline request type: ${message.type}`);
    handler(message);
  } catch (error) {
    self.postMessage({ type: "error", requestId: message.requestId, message: error && error.stack ? error.stack : String(error) });
  }
};
