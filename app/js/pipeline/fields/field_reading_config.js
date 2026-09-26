/**
 * Constants of the field reader (spec section 5, stages 5-6), all mirrored from
 * experiments/field_reading/, which trained and scored these models. Changing any number
 * here without changing the Python first breaks parity (app/tests/parity/field_reading_*).
 */

// Model files shipped in app/models/ (relative to the worker script, js/pipeline/).
export const FIELD_MODEL_PATHS = Object.freeze({
  segnet: "../../models/segnet_mobilenetv3l_768.onnx",
  printedTextRecognizer: "../../models/crnn_general_h32.onnx",
  amountRecognizer: "../../models/crnn_amount_h32.onnx",
  handwritingStyleClassifier: "../../models/style_classifier_h32.onnx",
});

// The seven fields in the segnet's channel order (experiments/field_reading/config.py
// CheckField). The MICR band has no channel: it is never located, cropped or read.
export const FIELD_NAMES = Object.freeze([
  "payer_name",
  "payee",
  "amount_numeric",
  "amount_words",
  "date",
  "memo",
  "check_number",
]);

// segnet_config.py / segnet_postprocess.py / segnet_mobilenetv3l_768__gate.json.
export const SEGNET = Object.freeze({
  canvasWidth: 768,
  canvasHeight: 352,
  outputStride: 2,
  maskThreshold: 0.5,
  minimumComponentPixels: 12,
  logitClip: 30,
  minimumBoxConfidence: 0.8, // the val-selected gate: fewer boxes on absent fields, same IoU
  inputName: "canvas_rgb",
  outputName: "field_logits",
});

// data_access/field_crop.py: every reader sees the box plus this margin.
export const FIELD_CROP_MARGIN_FRACTION = 0.15;
export const FIELD_CROP_MINIMUM_MARGIN_PX = 6;
export const FIELD_CROP_MINIMUM_SIDE_PX = 2; // learned/field_crop_sources.py MIN_CROP_SIDE_PX

// learned/line_image_preprocessing.py + the ONNX sidecar JSONs (crnn_*_h32.json).
export const LINE_IMAGE = Object.freeze({
  inputHeight: 32,
  minimumWidth: 48,
  maximumWidth: 800,
  widthDownsample: 4,
  padValue: 1.0,
  // Lines are right-padded to a multiple of this width, as in training (MPS_WIDTH_MULTIPLE).
  padWidthMultiple: 128,
});

export const PRINTED_TEXT_CHARSET = " #$&*+,-./0123456789=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
export const AMOUNT_CHARSET = " $*,-./0123456789=xXon";
export const CRNN_INPUT_NAME = "line_images";
export const CRNN_OUTPUT_NAME = "log_probabilities";

// learned/handwriting_style_classifier.py: padded like the CRNN input, with the valid width
// passed as a fraction so the padding is masked out of the pooling.
export const STYLE_CLASSIFIER = Object.freeze({
  inputName: "line_images",
  validWidthInputName: "valid_width_fraction",
  outputName: "handwritten_logit",
});

// Which recognizer reads each field on the printed path (learned/reading_methods.py
// `crnn_amount_route`: the amount CRNN for the courtesy box, the general CRNN elsewhere).
export const PRINTED_READER_BY_FIELD = Object.freeze({
  payer_name: "crnn_general",
  payee: "crnn_general",
  amount_numeric: "crnn_amount",
  amount_words: "crnn_general",
  date: "crnn_general",
  memo: "crnn_general",
  check_number: "crnn_general",
});
