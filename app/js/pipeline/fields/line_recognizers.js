/**
 * Stage 6 for printed text (spec section 5): the two CRNN line recognizers and the
 * printed-vs-handwritten style classifier, all on the same grey line image
 * (experiments/field_reading/learned/line_image_preprocessing.py): luma grey, bilinear
 * resize to height 32 keeping aspect (width clamped to [48, 800]), (x / 255 - 0.5) / 0.5.
 *
 * Two deliberate details, both measured on 300 eval crops (Python, same models):
 * - Every line is right-padded with white (1.0) to a multiple of 128 px before the CRNN, and
 *   only the valid timesteps are decoded. The CRNN (a bidirectional LSTM) was trained and
 *   scored on 128-px-padded batches; unpadded reads lost 3.4 points overall (legal line
 *   50.7% -> 41.6%). Padding each line alone keeps reads independent of other fields.
 *   The style classifier pads the same way and masks the padding out itself.
 * - The resize is INTER_LINEAR_EXACT, not INTER_LINEAR: the research ran cv2 on an ARM Mac,
 *   whose INTER_LINEAR goes through a vendor HAL (KleidiCV/Carotene) that no browser build
 *   reproduces. INTER_LINEAR_EXACT is bit-identical in OpenCV.js and cv2 and is closer to
 *   that HAL's output than OpenCV.js's own INTER_LINEAR (accuracy 57.0% vs 57.3% unpadded).
 */

import {
  AMOUNT_CHARSET,
  CRNN_INPUT_NAME,
  CRNN_OUTPUT_NAME,
  LINE_IMAGE,
  PRINTED_TEXT_CHARSET,
  STYLE_CLASSIFIER,
} from "./field_reading_config.js";
import { decodeGreedyCtc } from "./ctc_decoding.js";

const PIXEL_VALUE_SCALE = 255;
const CHARSET_BY_READER = Object.freeze({ crnn_general: PRINTED_TEXT_CHARSET, crnn_amount: AMOUNT_CHARSET });

/** Python 3's round(): halves go to the even neighbour. */
export function roundHalfToEven(value) {
  const floor = Math.floor(value);
  const fraction = value - floor;
  if (fraction < 0.5) return floor;
  if (fraction > 0.5) return floor + 1;
  return floor % 2 === 0 ? floor : floor + 1;
}

/**
 * The grey line image for one field window of an RGBA cv.Mat check crop.
 * Returns `{ data: Float32Array(32 * width), width }`.
 */
export function buildLineImage(cv, cropRgba, window) {
  const [x0, y0, x1, y1] = window;
  const fieldRegion = cropRgba.roi(new cv.Rect(x0, y0, x1 - x0, y1 - y0));
  const grey = new cv.Mat();
  const resized = new cv.Mat();
  try {
    cv.cvtColor(fieldRegion, grey, cv.COLOR_RGBA2GRAY);
    const targetWidth = Math.min(LINE_IMAGE.maximumWidth, Math.max(LINE_IMAGE.minimumWidth,
      roundHalfToEven((grey.cols * LINE_IMAGE.inputHeight) / Math.max(1, grey.rows))));
    cv.resize(grey, resized, new cv.Size(targetWidth, LINE_IMAGE.inputHeight), 0, 0, cv.INTER_LINEAR_EXACT);
    const data = new Float32Array(targetWidth * LINE_IMAGE.inputHeight);
    for (let index = 0; index < data.length; index += 1) {
      data[index] = (Math.fround(resized.data[index] / PIXEL_VALUE_SCALE) - 0.5) / 0.5;
    }
    return { data, width: targetWidth };
  } finally {
    fieldRegion.delete();
    grey.delete();
    resized.delete();
  }
}

/** The line right-padded with white to a multiple of `widthMultiple` px: `{ data, paddedWidth }`. */
function padLineImage(lineImage, widthMultiple) {
  const paddedWidth = Math.ceil(lineImage.width / widthMultiple) * widthMultiple;
  const data = new Float32Array(LINE_IMAGE.inputHeight * paddedWidth).fill(LINE_IMAGE.padValue);
  for (let row = 0; row < LINE_IMAGE.inputHeight; row += 1) {
    data.set(lineImage.data.subarray(row * lineImage.width, (row + 1) * lineImage.width), row * paddedWidth);
  }
  return { data, paddedWidth };
}

/** Resolves to `{ text, confidence }` from one CRNN (`readerId` "crnn_general" | "crnn_amount"). */
export function readLineWithCrnn(ort, session, readerId, lineImage) {
  const { data, paddedWidth } = padLineImage(lineImage, LINE_IMAGE.padWidthMultiple);
  const inputTensor = new ort.Tensor("float32", data, [1, 1, LINE_IMAGE.inputHeight, paddedWidth]);
  return session.run({ [CRNN_INPUT_NAME]: inputTensor }).then((outputs) => {
    const logProbabilities = outputs[CRNN_OUTPUT_NAME];
    const classCount = logProbabilities.dims[2];
    const validTimestepCount = Math.floor(lineImage.width / LINE_IMAGE.widthDownsample);
    return decodeGreedyCtc(logProbabilities.data, validTimestepCount, classCount, CHARSET_BY_READER[readerId]);
  });
}

/** Resolves to p(handwritten) for one line image. */
export function classifyHandwritingProbability(ort, session, lineImage) {
  const { data, paddedWidth } = padLineImage(lineImage, LINE_IMAGE.padWidthMultiple);
  const feeds = {
    [STYLE_CLASSIFIER.inputName]: new ort.Tensor("float32", data, [1, 1, LINE_IMAGE.inputHeight, paddedWidth]),
    [STYLE_CLASSIFIER.validWidthInputName]: new ort.Tensor("float32", new Float32Array([lineImage.width / paddedWidth]), [1]),
  };
  return session.run(feeds).then((outputs) => 1 / (1 + Math.exp(-outputs[STYLE_CLASSIFIER.outputName].data[0])));
}
