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
 * `{ type: "result" | "error" | "progress" | "check-ready" | "field-reads-ready", requestId, ... }`, plus one
 * `{ type: "ready" | "init-failed" }` after "init".
 */

// Deliberately NOT named `cv` / `ort`: a classic worker's top-level `let` is a global
// lexical binding, and ort.wasm.min.js declares a global `ort`, so a same-named `let`
// makes importScripts fail (Chrome reports it as a NetworkError).
let openCv = null;
let onnxRuntime = null;
let stages = null;
let classifierSession = null;
let fieldModels = null; // { segnetSession, crnnSessions, styleSession, handwritingReader }
let handwritingReaderUrls = null;
let handwritingReaderLoad = null; // in-flight Promise while the opt-in reader downloads
// The operator's latest on/off choice. A download that resolves after the switch went off
// (or after "Clear everything") must not turn the reader back on.
let handwritingReaderWanted = false;
// Bumped by every new batch and by release-photo: a read chain that sees a newer value stops.
let batchGeneration = 0;
let currentPhoto = null; // { imageBgr, workingGray, workingScale }
// Upright crops of the current batch, kept (memory only) so Rotate can re-read a check:
// checkIndex -> { width, height, rgbaPixels: Uint8ClampedArray }. Dropped with the photo.
let checkCrops = new Map();

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
  handwritingReaderUrls = message.handwritingReaderUrls;
  // Threads need cross-origin isolation (COOP/COEP headers), which GitHub Pages cannot set.
  onnxRuntime.env.wasm.numThreads = 1;
  const fail = (error) => self.postMessage({ type: "init-failed", message: String(error) });
  self.cv.then((readyCv) => {
    openCv = readyCv;
    import(resolveAgainstWorker("./worker_stages.js")).then((stageModule) => {
      stages = stageModule;
      Promise.all([
        stages.createUpsideDownClassifierSession(onnxRuntime, resolveAgainstWorker(message.classifierModelPath)),
        stages.createFieldModelSessions(onnxRuntime, resolveAgainstWorker),
      ]).then(([session, models]) => {
        classifierSession = session;
        fieldModels = models;
        self.postMessage({ type: "ready", openCvBuildInfo: openCv.getBuildInformation() });
      }, fail);
    }, fail);
  }, fail);
}

function releasePhoto() {
  batchGeneration += 1;
  checkCrops = new Map();
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

/**
 * Runs `readWithCropMat(cropRgba)` on a crop given as RGBA pixels (optionally turned 180
 * degrees first). The cv.Mat lives only for the duration of the returned Promise.
 */
function withCropMat(width, height, rgbaPixels, rotatedHalfTurn, readWithCropMat) {
  const cropRgba = new openCv.Mat(height, width, openCv.CV_8UC4);
  cropRgba.data.set(rgbaPixels);
  if (rotatedHalfTurn) openCv.rotate(cropRgba, cropRgba, openCv.ROTATE_180);
  const release = () => cropRgba.delete();
  let readPromise;
  try {
    readPromise = readWithCropMat(cropRgba);
  } catch (error) {
    release();
    throw error;
  }
  return readPromise.then((result) => { release(); return result; }, (error) => { release(); throw error; });
}

/** Pass 1 on a crop: `{ rawReads, timingsMs }` from the default readers. */
function readPrintedFields(crop, rotatedHalfTurn) {
  return withCropMat(crop.width, crop.height, crop.rgbaPixels, rotatedHalfTurn,
    (cropRgba) => stages.readCheckFields(openCv, onnxRuntime, fieldModels, cropRgba));
}

/** Pass 2 on a crop: TrOCR over pass 1's handwritten fields (handwriting reader on only). */
function readHandwritingFields(crop, rotatedHalfTurn, printedRawReads) {
  const reader = fieldModels.handwritingReader;
  return withCropMat(crop.width, crop.height, crop.rgbaPixels, rotatedHalfTurn,
    (cropRgba) => stages.readHandwrittenFields(onnxRuntime, reader, cropRgba, printedRawReads));
}

/** Both passes, as one result (Rotate re-reads, tests). */
function readAllFields(crop, rotatedHalfTurn) {
  return readPrintedFields(crop, rotatedHalfTurn).then((printed) => {
    if (!fieldModels.handwritingReader || !stages.hasHandwrittenFields(printed.rawReads)) return printed;
    return readHandwritingFields(crop, rotatedHalfTurn, printed.rawReads).then((handwriting) => ({
      rawReads: handwriting.rawReads, timingsMs: { ...printed.timingsMs, ...handwriting.timingsMs },
    }));
  });
}

/**
 * Orients and rectifies every confirmed quad, streaming each crop back as soon as it is
 * straight (the grid fills with crops first); then pass 1 reads every check with the
 * default readers; then, only if the handwriting reader is on, pass 2 re-reads the
 * handwritten fields check by check. Each pass streams `field-reads-ready` per check, so a
 * check can report twice (the grid re-gates the fields the operator has not touched).
 */
function orientAndRectifyChecks(message) {
  const photo = requirePhoto();
  batchGeneration += 1;
  const generation = batchGeneration;
  const checkCount = message.cornerSets.length;
  const printedReadsByCheck = new Map();
  // Start over, Finish batch or a new Continue released this batch: resolve quietly and stop.
  const isStale = () => {
    if (generation === batchGeneration) return false;
    postResult(message.requestId, { cancelled: true });
    return true;
  };
  const postProgress = (text) => self.postMessage({ type: "progress", requestId: message.requestId, text });
  const postError = (error) => {
    if (generation !== batchGeneration) return;
    self.postMessage({ type: "error", requestId: message.requestId, message: error && error.stack ? error.stack : String(error) });
  };
  const postReads = (checkIndex, result, pass) => self.postMessage({
    type: "field-reads-ready", requestId: message.requestId, checkIndex, pass, rawReads: result.rawReads, timingsMs: result.timingsMs,
  });
  const readHandwritingCheck = (checkIndex) => {
    if (isStale()) return;
    if (checkIndex >= checkCount || !fieldModels.handwritingReader) {
      postResult(message.requestId, {});
      return;
    }
    const printedRawReads = printedReadsByCheck.get(checkIndex);
    if (!stages.hasHandwrittenFields(printedRawReads)) {
      readHandwritingCheck(checkIndex + 1);
      return;
    }
    postProgress(`Reading handwriting on check ${checkIndex + 1} of ${checkCount}`);
    readHandwritingFields(checkCrops.get(checkIndex), false, printedRawReads).then((result) => {
      if (isStale()) return;
      postReads(checkIndex, result, "handwriting");
      readHandwritingCheck(checkIndex + 1);
    }).catch(postError);
  };
  const readCheck = (checkIndex) => {
    if (isStale()) return;
    if (checkIndex >= checkCount) {
      readHandwritingCheck(0);
      return;
    }
    postProgress(`Reading check ${checkIndex + 1} of ${checkCount}`);
    readPrintedFields(checkCrops.get(checkIndex), false).then((result) => {
      if (isStale()) return;
      printedReadsByCheck.set(checkIndex, result.rawReads);
      postReads(checkIndex, result, "printed");
      readCheck(checkIndex + 1);
    }).catch(postError);
  };
  const processCheck = (checkIndex) => {
    if (isStale()) return;
    if (checkIndex >= checkCount) {
      readCheck(0);
      return;
    }
    postProgress(`Straightening check ${checkIndex + 1} of ${checkCount}`);
    stages.orientAndRectifyCheck(openCv, onnxRuntime, classifierSession, photo, message.cornerSets[checkIndex]).then((checkResult) => {
      if (isStale()) return;
      checkCrops.set(checkIndex, { width: checkResult.width, height: checkResult.height, rgbaPixels: new Uint8ClampedArray(checkResult.rgbaPixels) });
      const rgbaBuffer = checkResult.rgbaPixels.buffer;
      self.postMessage({
        type: "check-ready", requestId: message.requestId, checkIndex,
        orientedCorners: checkResult.orientedCorners, upsideDownProbability: checkResult.upsideDownProbability,
        width: checkResult.width, height: checkResult.height, rgbaBuffer,
      }, [rgbaBuffer]);
      processCheck(checkIndex + 1);
    }).catch(postError);
  };
  processCheck(0);
}

/** Rotate on a row: re-read (both passes) that check's kept crop in the row's orientation. */
function rereadCheckFields(message) {
  const crop = checkCrops.get(message.checkIndex);
  if (!crop) throw new Error(`no crop is kept for check ${message.checkIndex}`);
  readAllFields(crop, message.rotatedHalfTurn).then(
    (result) => postResult(message.requestId, result),
    (error) => self.postMessage({ type: "error", requestId: message.requestId, message: String(error) }),
  );
}

/** Tests and the parity harness: read an arbitrary crop handed over as RGBA pixels (both passes). */
function readFieldsOfCrop(message) {
  const crop = { width: message.width, height: message.height, rgbaPixels: new Uint8Array(message.rgbaBuffer) };
  readAllFields(crop, false).then(
    (result) => postResult(message.requestId, result),
    (error) => self.postMessage({ type: "error", requestId: message.requestId, message: error && error.stack ? error.stack : String(error) }),
  );
}

/** The opt-in handwriting reader: download once (service-worker cached), or drop it. */
function setHandwritingReader(message) {
  const reply = () => postResult(message.requestId, { enabled: Boolean(fieldModels.handwritingReader) });
  const fail = (error) => self.postMessage({ type: "error", requestId: message.requestId, message: String(error) });
  handwritingReaderWanted = message.enabled;
  if (!message.enabled) {
    fieldModels.handwritingReader = null;
    reply();
    return;
  }
  if (fieldModels.handwritingReader) {
    reply();
    return;
  }
  const onProgress = (text) => self.postMessage({ type: "progress", requestId: message.requestId, text });
  if (!handwritingReaderLoad) {
    handwritingReaderLoad = stages.loadHandwritingReader(onnxRuntime, handwritingReaderUrls, onProgress);
    const clearLoad = () => { handwritingReaderLoad = null; };
    handwritingReaderLoad.then(clearLoad, clearLoad);
  }
  handwritingReaderLoad.then((reader) => {
    // Switched off (or cleared) while downloading: keep it off; the files stay cached.
    if (handwritingReaderWanted) fieldModels.handwritingReader = reader;
    reply();
  }, fail);
}

const REQUEST_HANDLERS = {
  "load-photo": loadPhoto,
  "detect-checks": detectChecks,
  "refit-drawn-rectangle": refitDrawnRectangle,
  "orient-and-rectify-checks": orientAndRectifyChecks,
  "reread-check-fields": rereadCheckFields,
  "read-fields-of-crop": readFieldsOfCrop,
  "set-handwriting-reader": setHandwritingReader,
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
