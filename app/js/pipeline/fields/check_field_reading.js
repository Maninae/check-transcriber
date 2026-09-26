/**
 * Reads every field of one upright check crop (spec section 5, stages 5-6) and returns the
 * raw reads that js/fields/field_gating.js turns into confident / unsure / blank.
 *
 Two passes, so the grid fills at the default readers' speed even with the (slow, opt-in)
 * handwriting reader on:
 * 1. `readCheckFields`: segnet boxes (field_localization.js) -> per field with a box: the
 *    field window (box + margin), its grey line image, p(handwritten) from the style
 *    classifier, and the CRNN read (the amount CRNN for the courtesy box, the general CRNN
 *    elsewhere). The gate blanks handwritten fields read this way.
 * 2. `readHandwrittenFields` (handwriting reader on only): TrOCR re-reads the fields pass 1
 *    called handwritten, as the field-reading README recommends.
 * The legal line is read (it feeds the amount cross-check) but the page never shows it. The
 * MICR band has no box and is never read.
 *
 * OpenCV is used synchronously; the only Promises are onnxruntime-web's (native), so
 * chaining them is safe (see pipeline/CLAUDE.md on `cv`).
 */

import { FIELD_CROP_MINIMUM_SIDE_PX, FIELD_NAMES, PRINTED_READER_BY_FIELD } from "./field_reading_config.js";
import { fieldCropWindow } from "./field_crop_window.js";
import { locateFieldBoxes } from "./field_localization.js";
import { buildLineImage, classifyHandwritingProbability, readLineWithCrnn } from "./line_recognizers.js";
import { readHandwrittenCrop } from "./handwriting_reader.js";

const HANDWRITTEN_PROBABILITY_THRESHOLD = 0.5; // mirrors js/fields/field_gating_config.js
const TROCR_READER_ID = "trocr";

function copyWindowRgbaPixels(cropRgba, window) {
  const [x0, y0, x1, y1] = window;
  const width = x1 - x0;
  const height = y1 - y0;
  const pixels = new Uint8Array(width * height * 4);
  const source = cropRgba.data;
  for (let row = 0; row < height; row += 1) {
    const sourceOffset = ((y0 + row) * cropRgba.cols + x0) * 4;
    pixels.set(source.subarray(sourceOffset, sourceOffset + width * 4), row * width * 4);
  }
  return { pixels, width, height };
}

/**
 * Pass 1: every field with the default readers (the CRNNs), never the handwriting reader.
 * `models`: `{ segnetSession, crnnSessions: { crnn_general, crnn_amount }, styleSession }`.
 * `cropRgba` is an RGBA cv.Mat (not deleted here).
 * Resolves to `{ rawReads, timingsMs: { localization, reading } }`, rawReads keyed by field
 * name, each `{ text, confidence, handwrittenProbability, reader, box } | null`.
 */
export function readCheckFields(cv, ort, models, cropRgba) {
  const startedAt = performance.now();
  const cropRgb = new cv.Mat();
  cv.cvtColor(cropRgba, cropRgb, cv.COLOR_RGBA2RGB);
  let locatePromise;
  try {
    locatePromise = locateFieldBoxes(cv, ort, models.segnetSession, cropRgb);
  } finally {
    cropRgb.delete();
  }
  return locatePromise.then((boxesByField) => {
    const localizedAt = performance.now();
    const rawReads = {};
    const readField = (fieldIndex) => {
      if (fieldIndex >= FIELD_NAMES.length) return Promise.resolve();
      const fieldName = FIELD_NAMES[fieldIndex];
      const window = fieldWindowOrNull(boxesByField[fieldName].box, cropRgba);
      if (!window) {
        rawReads[fieldName] = null;
        return readField(fieldIndex + 1);
      }
      const lineImage = buildLineImage(cv, cropRgba, window);
      const printedReaderId = PRINTED_READER_BY_FIELD[fieldName];
      return classifyHandwritingProbability(ort, models.styleSession, lineImage)
        .then((handwrittenProbability) => readLineWithCrnn(ort, models.crnnSessions[printedReaderId], printedReaderId, lineImage)
          .then((read) => {
            rawReads[fieldName] = { ...read, reader: printedReaderId, handwrittenProbability, box: boxesByField[fieldName].box };
            return readField(fieldIndex + 1);
          }));
    };
    return readField(0).then(() => ({
      rawReads,
      timingsMs: { localization: localizedAt - startedAt, reading: performance.now() - localizedAt },
    }));
  });
}

function fieldWindowOrNull(box, cropRgba) {
  if (!box) return null;
  const window = fieldCropWindow(box, cropRgba.cols, cropRgba.rows);
  return Math.min(window[2] - window[0], window[3] - window[1]) < FIELD_CROP_MINIMUM_SIDE_PX ? null : window;
}

/** Whether pass 2 has anything to do for these reads. */
export function hasHandwrittenFields(rawReads) {
  return Object.values(rawReads).some((read) => read && read.handwrittenProbability > HANDWRITTEN_PROBABILITY_THRESHOLD);
}

/**
 * Pass 2 (only with the opt-in handwriting reader): re-reads the fields pass 1 found
 * handwritten with TrOCR, reusing pass 1's boxes and style calls. The courtesy amount keeps
 * the higher-confidence of the amount CRNN and TrOCR (README section 5); every other
 * handwritten field takes TrOCR's read. Resolves to `{ rawReads, timingsMs: { handwriting } }`
 * (a new object; printed fields are copied through unchanged).
 */
export function readHandwrittenFields(ort, handwritingReader, cropRgba, printedRawReads) {
  const startedAt = performance.now();
  const rawReads = { ...printedRawReads };
  const handwrittenFieldNames = FIELD_NAMES.filter((fieldName) => {
    const read = printedRawReads[fieldName];
    return read && read.handwrittenProbability > HANDWRITTEN_PROBABILITY_THRESHOLD;
  });
  const readNext = (position) => {
    if (position >= handwrittenFieldNames.length) {
      return Promise.resolve({ rawReads, timingsMs: { handwriting: performance.now() - startedAt } });
    }
    const fieldName = handwrittenFieldNames[position];
    const printedRead = printedRawReads[fieldName];
    const { pixels, width, height } = copyWindowRgbaPixels(cropRgba, fieldCropWindow(printedRead.box, cropRgba.cols, cropRgba.rows));
    return readHandwrittenCrop(ort, handwritingReader, pixels, width, height).then((handwritingRead) => {
      const keepPrinted = fieldName === "amount_numeric" && !(handwritingRead.confidence > printedRead.confidence);
      rawReads[fieldName] = keepPrinted ? printedRead : { ...printedRead, ...handwritingRead, reader: TROCR_READER_ID };
      return readNext(position + 1);
    });
  };
  return readNext(0);
}
