/**
 * Main-thread handle on the pipeline worker (pipeline_worker.js). Starts it from a blob:
 * bootstrap so the page's Content-Security-Policy governs the worker too (see the worker's
 * module docstring), and turns its message protocol into Promises.
 *
 * These Promises settle on worker messages only; no OpenCV object ever crosses to the
 * main thread, so awaiting them is safe (the OpenCV thenable rule is the worker's concern).
 */

const WORKER_SCRIPT_URL = new URL("./pipeline_worker.js", import.meta.url).href;
const CLASSIFIER_MODEL_PATH = "../../models/upside_down_classifier.onnx"; // relative to the worker script

export class PipelineClient {
  /** `libraryUrls`: `{ openCvUrl, onnxRuntimeJsUrl, onnxRuntimeWasmDirectory }` from cdn_config.js. */
  constructor(libraryUrls) {
    this.libraryUrls = libraryUrls;
    this.worker = null;
    this.nextRequestId = 1;
    this.pendingRequests = new Map(); // requestId -> { resolve, reject, onProgress, onCheckReady }
  }

  /** Starts the worker; calls `onReady({ openCvBuildInfo })` or `onError(error)`. */
  start(onReady, onError) {
    const bootstrapSource = `self.PIPELINE_WORKER_URL = ${JSON.stringify(WORKER_SCRIPT_URL)};\nimportScripts(self.PIPELINE_WORKER_URL);\n`;
    const bootstrapUrl = URL.createObjectURL(new Blob([bootstrapSource], { type: "text/javascript" }));
    this.worker = new Worker(bootstrapUrl);
    // Revoked only once the worker has reported in, so the revoke can't race its own fetch.
    const revokeBootstrap = () => URL.revokeObjectURL(bootstrapUrl);
    const onReadyAndRevoke = (details) => { revokeBootstrap(); onReady(details); };
    const onErrorAndRevoke = (error) => { revokeBootstrap(); onError(error); };
    this.worker.onmessage = (event) => this.handleMessage(event.data, onReadyAndRevoke, onErrorAndRevoke);
    this.worker.onerror = (event) => onErrorAndRevoke(new Error(`pipeline worker failed: ${event.message}`));
    this.worker.postMessage({ type: "init", ...this.libraryUrls, classifierModelPath: CLASSIFIER_MODEL_PATH });
  }

  /** Stops the worker (used by Retry after a failed start). */
  terminate() {
    if (this.worker) this.worker.terminate();
    this.worker = null;
    for (const pending of this.pendingRequests.values()) pending.reject(new Error("pipeline worker stopped"));
    this.pendingRequests.clear();
  }

  handleMessage(message, onReady, onError) {
    if (message.type === "ready") return onReady({ openCvBuildInfo: message.openCvBuildInfo });
    if (message.type === "init-failed") return onError(new Error(message.message));
    const pending = this.pendingRequests.get(message.requestId);
    if (!pending) return undefined;
    if (message.type === "progress") return pending.onProgress?.(message.text);
    if (message.type === "check-ready") return pending.onCheckReady?.(message);
    this.pendingRequests.delete(message.requestId);
    if (message.type === "error") return pending.reject(new Error(message.message));
    return pending.resolve(message);
  }

  request(type, payload = {}, { transfer = [], onProgress, onCheckReady } = {}) {
    const requestId = this.nextRequestId;
    this.nextRequestId += 1;
    return new Promise((resolve, reject) => {
      this.pendingRequests.set(requestId, { resolve, reject, onProgress, onCheckReady });
      this.worker.postMessage({ type, requestId, ...payload }, transfer);
    });
  }

  /** Copies the full-resolution photo into the worker (the page keeps its own canvas). */
  loadPhoto(fullResCanvas) {
    const { width, height } = fullResCanvas;
    const imageData = fullResCanvas.getContext("2d").getImageData(0, 0, width, height);
    const rgbaBuffer = imageData.data.buffer;
    return this.request("load-photo", { width, height, rgbaBuffer }, { transfer: [rgbaBuffer] });
  }

  /** Resolves to `[{ corners, confident }]` in reading order. */
  detectChecks(onProgress) {
    return this.request("detect-checks", {}, { onProgress }).then((message) => message.detections);
  }

  /** Resolves to `{ corners, refitSource }` for a rough rectangle the operator drew. */
  refitDrawnRectangle(drawnCorners, otherCornerSets) {
    return this.request("refit-drawn-rectangle", { drawnCorners, otherCornerSets });
  }

  /** Orients and crops each quad; `onCheckReady(message)` fires per check as it finishes. */
  orientAndRectifyChecks(cornerSets, onCheckReady, onProgress) {
    return this.request("orient-and-rectify-checks", { cornerSets }, { onCheckReady, onProgress });
  }

  /** Frees the worker's copy of the photo (Finish batch, Start over). */
  releasePhoto() {
    return this.worker ? this.request("release-photo") : Promise.resolve();
  }
}
