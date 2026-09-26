/**
 * Every binary mask whose connected regions are check candidates (Python
 * `candidate_regions.build_candidate_masks`), produced one at a time.
 *
 * Mask families, each catching checks the others miss:
 * - smooth-paper (`textureStd` below a threshold, not too dark): busy fabrics, carpet.
 * - textured (`textureStd` above a threshold): plain white sheets, where the check's
 *   security print is the textured thing.
 * - paper-score Otsu (bright and neutral): wood, colored rugs, dark surfaces.
 * - edge-bounded cells (NOT of dilated gradient / Canny edges): each check interior is a
 *   cell fenced by its own border, which also separates touching checks.
 * - hole-filled Canny (external contours drawn solid): a check whose border is a closed
 *   loop becomes one blob however many print edges cross it.
 * Masks are handed to `onMask(name, maskMat)` in the Python dict order and deleted after
 * the callback returns, so only one full-size mask is alive at a time.
 */

import { buildCombinedCannyEdges } from "../preprocessing/combined_edge_map.js";
import { withMats } from "../numeric/mat_helpers.js";
import { formatGeneral, formatGeneralSigned, percentileLinear } from "../numeric/numpy_compatibility.js";

const DARK_FLOOR_PERCENTILE = 25.0; // smooth regions darker than this paper-score percentile are not paper
const MASK_ON = 255;
const fround = Math.fround;

/** A new CV_8U mask Mat: 255 where values[i] > threshold (or < threshold when `isBelow`). */
function thresholdMask(cv, width, height, values, threshold, isBelow) {
  const maskMat = new cv.Mat(height, width, cv.CV_8U);
  const maskBytes = maskMat.data;
  const pixelCount = width * height;
  if (isBelow) {
    for (let index = 0; index < pixelCount; index += 1) maskBytes[index] = values[index] < threshold ? MASK_ON : 0;
  } else {
    for (let index = 0; index < pixelCount; index += 1) maskBytes[index] = values[index] > threshold ? MASK_ON : 0;
  }
  return maskMat;
}

/** Hand `maskMat` to the callback, then delete it. */
function emitMask(onMask, name, maskMat) {
  try {
    onMask(name, maskMat);
  } finally {
    maskMat.delete();
  }
}

/** The mask with every region it fully encloses filled (external contours drawn solid). */
export function fillEnclosedHoles(cv, binaryMaskMat) {
  const filledMask = cv.Mat.zeros(binaryMaskMat.rows, binaryMaskMat.cols, cv.CV_8U);
  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();
  try {
    cv.findContours(binaryMaskMat, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
    cv.drawContours(filledMask, contours, -1, new cv.Scalar(MASK_ON), cv.FILLED);
    return filledMask;
  } finally {
    contours.delete();
    hierarchy.delete();
  }
}

/** Smooth-paper, textured and paper-score Otsu masks. */
function emitAppearanceMasks(cv, channels, config, onMask) {
  const { width, height } = channels;
  const paperScore = channels.paperScore.values;
  const textureStd = channels.textureStd.values;
  const darkFloor = percentileLinear(paperScore, DARK_FLOOR_PERCENTILE);
  for (const textureThreshold of config.smoothTextureStdThresholds) {
    const threshold = fround(textureThreshold);
    const maskMat = thresholdMask(cv, width, height, textureStd, threshold, true);
    const maskBytes = maskMat.data;
    for (let index = 0; index < width * height; index += 1) if (!(paperScore[index] > darkFloor)) maskBytes[index] = 0;
    emitMask(onMask, `smooth_std${formatGeneral(textureThreshold)}`, maskMat);
  }
  for (const textureThreshold of config.texturedStdThresholds) {
    const threshold = fround(textureThreshold);
    const maskMat = thresholdMask(cv, width, height, textureStd, threshold, false);
    emitMask(onMask, `textured_std${formatGeneral(textureThreshold)}`, maskMat);
  }
  const otsuThreshold = withMats((track) => {
    const paperScoreBytes = track(new cv.Mat(height, width, cv.CV_8U));
    const bytes = paperScoreBytes.data;
    for (let index = 0; index < width * height; index += 1) {
      const value = paperScore[index];
      bytes[index] = value <= 0 ? 0 : value >= 255 ? 255 : Math.trunc(value);
    }
    return cv.threshold(paperScoreBytes, track(new cv.Mat()), 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);
  });
  for (const offset of config.paperScoreOtsuOffsets) {
    const threshold = fround(otsuThreshold + offset); // a Python float compared to float32 is cast to float32
    const maskMat = thresholdMask(cv, width, height, paperScore, threshold, false);
    emitMask(onMask, `paper_otsu${formatGeneralSigned(offset)}`, maskMat);
  }
}

/** Gradient-threshold edge cells and Canny cells / hole-filled Canny masks. */
function emitEdgeMasks(cv, channels, config, onMask) {
  const { width, height } = channels;
  const gradientMagnitude = channels.gradientMagnitude.values;
  withMats((track) => {
    const dilationSize = 2 * config.edgeDilationPixels + 1;
    const dilationKernel = track(cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(dilationSize, dilationSize)));
    for (const gradientThreshold of config.edgeGradientThresholds) {
      const threshold = fround(gradientThreshold);
      const edgeMask = track(thresholdMask(cv, width, height, gradientMagnitude, threshold, false));
      const dilatedEdges = track(new cv.Mat());
      cv.dilate(edgeMask, dilatedEdges, dilationKernel);
      const cellMask = new cv.Mat();
      cv.bitwise_not(dilatedEdges, cellMask);
      emitMask(onMask, `edge_cells${formatGeneral(gradientThreshold)}`, cellMask);
    }
    const cannyDilationSize = 2 * config.cannyDilationPixels + 1;
    const cannyKernel = track(cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(cannyDilationSize, cannyDilationSize)));
    for (const [lowThreshold, highThreshold] of config.cannyThresholdPairs) {
      const cannyEdges = track(buildCombinedCannyEdges(cv, channels, lowThreshold, highThreshold, config));
      const dilatedEdges = track(new cv.Mat());
      cv.dilate(cannyEdges, dilatedEdges, cannyKernel);
      const cellMask = new cv.Mat();
      cv.bitwise_not(dilatedEdges, cellMask);
      emitMask(onMask, `canny_cells${formatGeneral(lowThreshold)}`, cellMask);
      if (config.useHoleFilledEdges) {
        emitMask(onMask, `canny_filled${formatGeneral(lowThreshold)}`, fillEnclosedHoles(cv, dilatedEdges));
      }
    }
  });
}

/** Call `onMask(name, maskMat)` for every candidate mask, in the Python order. */
export function forEachCandidateMask(cv, channels, config, onMask) {
  emitAppearanceMasks(cv, channels, config, onMask);
  emitEdgeMasks(cv, channels, config, onMask);
}

