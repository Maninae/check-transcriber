/**
 * Runs the upside-down check classifier (models/upside_down_classifier.onnx, trained in
 * experiments/detection/orientation/, exported by experiments/detection/export/
 * export_orientation_onnx.py) with onnxruntime-web on its WASM backend.
 *
 * Input `crop`: (1, 1, 96, 224) float32 in [0, 1]. Output `upside_down_logit`: (1,),
 * > 0 means upside down. These are plain native Promises (onnxruntime-web's own), not
 * OpenCV.js's thenable, so chaining them is safe (see engine_loader.js on why `cv` is not).
 */

import { ORIENTATION_CROP_HEIGHT, ORIENTATION_CROP_WIDTH } from "./check_orientation.js";

const CLASSIFIER_INPUT_NAME = "crop";
const CLASSIFIER_OUTPUT_NAME = "upside_down_logit";

/** Creates the session from the model URL; resolves to the session. */
export function createUpsideDownClassifierSession(ort, modelUrl) {
  return ort.InferenceSession.create(modelUrl, { executionProviders: ["wasm"] });
}

/** Resolves to the classifier logit for one crop (> 0 means upside down). */
export function classifyUpsideDownLogit(ort, session, cropTensorData) {
  const inputTensor = new ort.Tensor("float32", cropTensorData, [1, 1, ORIENTATION_CROP_HEIGHT, ORIENTATION_CROP_WIDTH]);
  return session.run({ [CLASSIFIER_INPUT_NAME]: inputTensor }).then((outputs) => outputs[CLASSIFIER_OUTPUT_NAME].data[0]);
}

/** Logistic function, so the caller can report an upside-down probability. */
export function logitToProbability(logit) {
  return 1 / (1 + Math.exp(-logit));
}
