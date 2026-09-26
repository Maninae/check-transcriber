/**
 * Does the inside of a candidate quad look like a check? (Python `interior_appearance.py`)
 *
 * Edge evidence alone accepts any bordered four-sided cell (a tile boxed by grout, a strip of
 * wood between grain lines). Every check has printed text on smooth, bright, near-neutral
 * paper. Measured over the quad shrunk by 10% (excluding the border), on every 2nd pixel:
 * - print fraction: share of pixels whose print residue exceeds a threshold;
 * - medians of texture std, paper score and chroma.
 * The interior raster uses `cv.fillConvexPoly` for pixel-exact agreement with the Python.
 */

import { createInt32PointMat, withMats } from "../numeric/mat_helpers.js";
import { medianOfFloat32InPlace, medianPassesThreshold, roundHalfEven } from "../numeric/numpy_compatibility.js";

const INTERIOR_SHRINK_FACTOR = 0.9;
const INTERIOR_SAMPLING_STRIDE = 2; // medians over every 2nd pixel in each direction (4x cheaper)
const fround = Math.fround;

/** Rasterize a convex polygon (integer points) into a new zeroed uint8 array of the crop size. */
function rasterizeConvexPolygon(cv, integerPoints, cropWidth, cropHeight) {
  return withMats((track) => {
    const maskMat = track(cv.Mat.zeros(cropHeight, cropWidth, cv.CV_8U));
    const pointMat = track(createInt32PointMat(cv, integerPoints));
    cv.fillConvexPoly(maskMat, pointMat, new cv.Scalar(1));
    return Uint8Array.from(maskMat.data);
  });
}

/**
 * Statistics over the shrunken quad interior; null when it has no in-frame pixels.
 *
 * Returns `{ printFraction, textureMedian, paperScoreMedian, chromaMedian }`. With
 * `stopAtFirstFailure` it instead returns null as soon as one statistic fails the check gate
 * and `{ passed: true }` when all four pass (no medians computed unless needed).
 */
export function measureInteriorAppearance(cv, corners, channels, config, stopAtFirstFailure = false) {
  const { width: imageWidth, height: imageHeight } = channels;
  const centroidX = (corners[0][0] + corners[1][0] + corners[2][0] + corners[3][0]) / 4;
  const centroidY = (corners[0][1] + corners[1][1] + corners[2][1] + corners[3][1]) / 4;
  const shrunkCorners = corners.map(([x, y]) => [
    centroidX + (x - centroidX) * INTERIOR_SHRINK_FACTOR, centroidY + (y - centroidY) * INTERIOR_SHRINK_FACTOR,
  ]);
  const shrunkXs = shrunkCorners.map(([x]) => x);
  const shrunkYs = shrunkCorners.map(([, y]) => y);
  const left = Math.trunc(Math.max(0, Math.floor(Math.min(...shrunkXs))));
  const top = Math.trunc(Math.max(0, Math.floor(Math.min(...shrunkYs))));
  const right = Math.trunc(Math.min(imageWidth, Math.ceil(Math.max(...shrunkXs)) + 1));
  const bottom = Math.trunc(Math.min(imageHeight, Math.ceil(Math.max(...shrunkYs)) + 1));
  if (right <= left || bottom <= top) return null;
  const cropWidth = right - left;
  const cropHeight = bottom - top;
  const integerPoints = shrunkCorners.map(([x, y]) => [roundHalfEven(x - left), roundHalfEven(y - top)]);
  const interiorMask = rasterizeConvexPolygon(cv, integerPoints, cropWidth, cropHeight);

  let insideCount = 0;
  for (let row = 0; row < cropHeight; row += INTERIOR_SAMPLING_STRIDE) {
    for (let column = 0; column < cropWidth; column += INTERIOR_SAMPLING_STRIDE) insideCount += interiorMask[row * cropWidth + column];
  }
  if (insideCount === 0) return null;
  const printResidue = channels.printResidue.values;
  const textureStd = channels.textureStd.values;
  const paperScore = channels.paperScore.values;
  const chroma = channels.chroma.values;
  const textureValues = new Float32Array(insideCount);
  const paperScoreValues = new Float32Array(insideCount);
  const chromaValues = new Float32Array(insideCount);
  const printThreshold = fround(config.printResidueThreshold);
  let printCount = 0;
  let cursor = 0;
  for (let row = 0; row < cropHeight; row += INTERIOR_SAMPLING_STRIDE) {
    for (let column = 0; column < cropWidth; column += INTERIOR_SAMPLING_STRIDE) {
      if (!interiorMask[row * cropWidth + column]) continue;
      const pixelIndex = (top + row) * imageWidth + left + column;
      if (printResidue[pixelIndex] > printThreshold) printCount += 1;
      textureValues[cursor] = textureStd[pixelIndex];
      paperScoreValues[cursor] = paperScore[pixelIndex];
      chromaValues[cursor] = chroma[pixelIndex];
      cursor += 1;
    }
  }
  const printFraction = printCount / insideCount;
  if (stopAtFirstFailure) {
    if (!(printFraction >= config.minimumInteriorPrintFraction)) return null;
    if (!medianPassesThreshold(textureValues, config.maximumInteriorTexture)) return null;
    if (!medianPassesThreshold(paperScoreValues, config.minimumInteriorPaperScore, true)) return null;
    if (!medianPassesThreshold(chromaValues, config.maximumInteriorChroma)) return null;
    return { passed: true };
  }
  return {
    printFraction,
    textureMedian: medianOfFloat32InPlace(textureValues),
    paperScoreMedian: medianOfFloat32InPlace(paperScoreValues),
    chromaMedian: medianOfFloat32InPlace(chromaValues),
  };
}

/**
 * `interiorLooksLikeCheck(measureInteriorAppearance(...))`, stopping at the first failing
 * statistic (same verdict; most rejected candidates then skip the median selections).
 */
export function interiorPassesCheckGate(cv, corners, channels, config) {
  return measureInteriorAppearance(cv, corners, channels, config, true) !== null;
}

/** All four interior statistics inside the check range. */
export function interiorLooksLikeCheck(appearance, config) {
  if (appearance === null) return false;
  return (
    appearance.printFraction >= config.minimumInteriorPrintFraction
    && appearance.textureMedian <= config.maximumInteriorTexture
    && appearance.paperScoreMedian >= config.minimumInteriorPaperScore
    && appearance.chromaMedian <= config.maximumInteriorChroma
  );
}
