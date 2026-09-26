/**
 * Node-side helpers for Python-vs-JS parity runs of app/js/pipeline/ modules.
 *
 * The pipeline modules take `cv` as an argument and never touch DOM or worker globals,
 * so the exact same files run here and in the browser worker. This loads the same
 * pinned OpenCV.js build the app fetches (a local copy on vega, so runs are offline).
 *
 * - `loadOpenCvForNode(onReady)`: callback style, same reason as app/js/engine_loader.js
 *   (`cv` is a non-native thenable).
 * - `readBgrSceneMat(cv, directory, sceneId)`: a CV_8UC3 BGR Mat from dump_scene_pixels.py.
 */

import fs from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";

const require = createRequire(import.meta.url);
export const OPENCV_JS_LOCAL_COPY =
  "/Volumes/vega/datasets/check-transcriber/tools/app-parity/opencv-5.0.0-release.1.js";

/** Loads OpenCV.js and calls `onReady(cv)` once its WASM is initialized. */
export function loadOpenCvForNode(onReady) {
  const cvModule = require(OPENCV_JS_LOCAL_COPY);
  if (typeof cvModule.then === "function") {
    cvModule.then((cv) => onReady(cv));
  } else {
    cvModule.onRuntimeInitialized = () => onReady(cvModule);
  }
}

/** Reads `<directory>/<sceneId>.bgr` + `.json` into a new BGR Mat (caller deletes it). */
export function readBgrSceneMat(cv, directory, sceneId) {
  const { width, height } = JSON.parse(fs.readFileSync(path.join(directory, `${sceneId}.json`), "utf8"));
  const pixelBytes = fs.readFileSync(path.join(directory, `${sceneId}.bgr`));
  const imageBgr = new cv.Mat(height, width, cv.CV_8UC3);
  imageBgr.data.set(pixelBytes);
  return imageBgr;
}
