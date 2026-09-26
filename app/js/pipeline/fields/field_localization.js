/**
 * Stage 5 (spec section 5): find the seven field boxes on an upright check crop with the
 * segmentation net (models/segnet_mobilenetv3l_768.onnx), a port of experiments/
 * field_reading/field_localization/ (`render_check_to_canvas`, `segnet_postprocess`).
 *
 * 1. Canvas: the crop is halved (INTER_AREA, standing in for the half-resolution JPEG decode
 *    the Python trained on), then warped onto a 768 x 352 zero canvas at 768 / crop width,
 *    top-left aligned, exactly like the Python's crop -> canvas homography.
 * 2. The net returns 7 logit maps at stride 2. Per field: sigmoid, bilinear upsample to the
 *    canvas, threshold 0.5, 8-connected components, keep the component with the largest
 *    probability mass (>= 12 px). Box = its pixel extent mapped back to the crop;
 *    confidence = mean probability inside it; boxes under the val-selected gate are dropped.
 *
 * Boxes are pixel-edge [x0, y0, x1, y1] in the crop. Synchronous OpenCV around one ORT call.
 */

import { FIELD_NAMES, SEGNET } from "./field_reading_config.js";

const PIXEL_VALUE_SCALE = 255;

/** Float32Array (1, 3, 352, 768) CHW RGB in [0, 1] for an RGB cv.Mat crop. */
export function buildSegnetCanvasTensorData(cv, cropRgb) {
  const cropWidth = cropRgb.cols;
  const cropHeight = cropRgb.rows;
  const reducedWidth = Math.ceil(cropWidth / 2);
  const reducedHeight = Math.ceil(cropHeight / 2);
  const reduced = new cv.Mat();
  const canvas = new cv.Mat();
  const canvasScale = SEGNET.canvasWidth / cropWidth;
  // crop_to_canvas @ reduced_to_crop from segnet_dataset.render_check_to_canvas.
  const reducedToCanvas = cv.matFromArray(3, 3, cv.CV_64F, [
    canvasScale * (cropWidth / reducedWidth), 0, 0,
    0, canvasScale * (cropHeight / reducedHeight), 0,
    0, 0, 1,
  ]);
  try {
    cv.resize(cropRgb, reduced, new cv.Size(reducedWidth, reducedHeight), 0, 0, cv.INTER_AREA);
    cv.warpPerspective(reduced, canvas, reducedToCanvas, new cv.Size(SEGNET.canvasWidth, SEGNET.canvasHeight),
      cv.INTER_AREA, cv.BORDER_CONSTANT, new cv.Scalar(0, 0, 0, 0));
    const planeSize = SEGNET.canvasWidth * SEGNET.canvasHeight;
    const tensorData = new Float32Array(3 * planeSize);
    const pixels = canvas.data;
    for (let index = 0; index < planeSize; index += 1) {
      tensorData[index] = pixels[index * 3] / PIXEL_VALUE_SCALE;
      tensorData[planeSize + index] = pixels[index * 3 + 1] / PIXEL_VALUE_SCALE;
      tensorData[2 * planeSize + index] = pixels[index * 3 + 2] / PIXEL_VALUE_SCALE;
    }
    return tensorData;
  } finally {
    reduced.delete();
    canvas.delete();
    reducedToCanvas.delete();
  }
}

/** One field's stride-2 logits -> `{ box, confidence }` (box null when absent). Mirrors `field_probability_to_box`. */
function fieldLogitsToBox(cv, logits, offset, gridWidth, gridHeight, canvasScale, cropWidth, cropHeight) {
  const gridSize = gridWidth * gridHeight;
  const strideProbability = new Float32Array(gridSize);
  let peakProbability = -Infinity;
  for (let index = 0; index < gridSize; index += 1) {
    const clipped = Math.min(SEGNET.logitClip, Math.max(-SEGNET.logitClip, logits[offset + index]));
    const probability = Math.fround(1 / (1 + Math.exp(-clipped)));
    strideProbability[index] = probability;
    if (probability > peakProbability) peakProbability = probability;
  }
  if (peakProbability < SEGNET.maskThreshold) return { box: null, confidence: peakProbability };
  const strideMat = cv.matFromArray(gridHeight, gridWidth, cv.CV_32F, strideProbability);
  const canvasProbability = new cv.Mat();
  const labels = new cv.Mat();
  let foreground = null;
  try {
    cv.resize(strideMat, canvasProbability, new cv.Size(gridWidth * SEGNET.outputStride, gridHeight * SEGNET.outputStride), 0, 0, cv.INTER_LINEAR);
    const probabilities = canvasProbability.data32F;
    const foregroundBytes = new Uint8Array(probabilities.length);
    for (let index = 0; index < probabilities.length; index += 1) foregroundBytes[index] = probabilities[index] >= SEGNET.maskThreshold ? 1 : 0;
    foreground = cv.matFromArray(canvasProbability.rows, canvasProbability.cols, cv.CV_8U, foregroundBytes);
    const componentCount = cv.connectedComponents(foreground, labels, 8, cv.CV_32S);
    if (componentCount <= 1) return { box: null, confidence: peakProbability };
    const labelData = labels.data32S;
    const probabilityMass = new Float64Array(componentCount);
    const pixelCounts = new Int32Array(componentCount);
    for (let index = 0; index < labelData.length; index += 1) {
      probabilityMass[labelData[index]] += probabilities[index];
      pixelCounts[labelData[index]] += 1;
    }
    probabilityMass[0] = -1;
    let bestLabel = 0;
    for (let label = 0; label < componentCount; label += 1) {
      if (pixelCounts[label] < SEGNET.minimumComponentPixels) probabilityMass[label] = -1;
      if (probabilityMass[label] > probabilityMass[bestLabel]) bestLabel = label;
    }
    if (probabilityMass[bestLabel] < 0) return { box: null, confidence: peakProbability };
    const canvasWidth = canvasProbability.cols;
    let minimumColumn = Infinity;
    let maximumColumn = -Infinity;
    let minimumRow = Infinity;
    let maximumRow = -Infinity;
    for (let index = 0; index < labelData.length; index += 1) {
      if (labelData[index] !== bestLabel) continue;
      const row = Math.floor(index / canvasWidth);
      const column = index - row * canvasWidth;
      if (column < minimumColumn) minimumColumn = column;
      if (column > maximumColumn) maximumColumn = column;
      if (row < minimumRow) minimumRow = row;
      if (row > maximumRow) maximumRow = row;
    }
    const clamp = (value, limit) => Math.min(Math.max(value, 0), limit);
    const box = [
      clamp(minimumColumn / canvasScale, cropWidth),
      clamp(minimumRow / canvasScale, cropHeight),
      clamp((maximumColumn + 1) / canvasScale, cropWidth),
      clamp((maximumRow + 1) / canvasScale, cropHeight),
    ];
    return { box, confidence: probabilityMass[bestLabel] / pixelCounts[bestLabel] };
  } finally {
    strideMat.delete();
    canvasProbability.delete();
    if (foreground) foreground.delete();
    labels.delete();
  }
}

/**
 * Resolves to `{ field name: { box, confidence } }` for the seven fields; `box` is null when
 * the field is absent or under the confidence gate. `cropRgb` is an RGB cv.Mat (not deleted).
 */
export function locateFieldBoxes(cv, ort, segnetSession, cropRgb) {
  const tensorData = buildSegnetCanvasTensorData(cv, cropRgb);
  const cropWidth = cropRgb.cols;
  const cropHeight = cropRgb.rows;
  const inputTensor = new ort.Tensor("float32", tensorData, [1, 3, SEGNET.canvasHeight, SEGNET.canvasWidth]);
  return segnetSession.run({ [SEGNET.inputName]: inputTensor }).then((outputs) => {
    const logitsTensor = outputs[SEGNET.outputName];
    const [, channelCount, gridHeight, gridWidth] = logitsTensor.dims;
    const canvasScale = SEGNET.canvasWidth / cropWidth;
    const boxesByField = {};
    for (let channel = 0; channel < channelCount; channel += 1) {
      const located = fieldLogitsToBox(cv, logitsTensor.data, channel * gridHeight * gridWidth, gridWidth, gridHeight, canvasScale, cropWidth, cropHeight);
      const passesGate = located.box !== null && located.confidence >= SEGNET.minimumBoxConfidence;
      boxesByField[FIELD_NAMES[channel]] = { box: passesGate ? located.box : null, confidence: located.confidence };
    }
    return boxesByField;
  });
}
