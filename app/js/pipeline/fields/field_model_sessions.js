/**
 * Opens the four default field-reading models (segnet, two CRNNs, style classifier) from
 * app/models/ with onnxruntime-web's WASM backend. fp32 on purpose: int8 moved CRNN
 * confidences by up to 1.6 nats in the field-reading export check, which breaks the gate.
 */

import { FIELD_MODEL_PATHS } from "./field_reading_config.js";

// A fresh options object per session: onnxruntime-web writes into it (`extra`), so a shared
// or frozen object breaks session creation.
const createSessionOptions = () => ({ executionProviders: ["wasm"] });

/** `resolveUrl(relativePath)` resolves against the worker script. Resolves to the sessions object readCheckFields takes. */
export function createFieldModelSessions(ort, resolveUrl) {
  const open = (relativePath) => ort.InferenceSession.create(resolveUrl(relativePath), createSessionOptions());
  return Promise.all([
    open(FIELD_MODEL_PATHS.segnet),
    open(FIELD_MODEL_PATHS.printedTextRecognizer),
    open(FIELD_MODEL_PATHS.amountRecognizer),
    open(FIELD_MODEL_PATHS.handwritingStyleClassifier),
  ]).then(([segnetSession, printedTextSession, amountSession, styleSession]) => ({
    segnetSession,
    crnnSessions: { crnn_general: printedTextSession, crnn_amount: amountSession },
    styleSession,
    handwritingReader: null,
  }));
}
