/**
 * Build the per-pixel signal maps the candidate generators threshold (Python
 * `working_image_channels.py`).
 *
 * The working copy is the photo resized so its long side is `workingLongSidePixels`
 * (INTER_AREA, never upscaled). All maps are float32 of the working size, stored as
 * `{ values: Float32Array, width, height }` JS copies (no Mats are retained, so the caller
 * has nothing to free):
 * - `lightness`: Lab L after a grayscale closing that erases thin dark text strokes.
 * - `chroma`: distance from neutral in (blurred) Lab a/b.
 * - `paperScore`: lightness - w * chroma, high on bright neutral paper.
 * - `textureStd`: local std of `lightness` (paper is smooth, fabric is not).
 * - `labA`, `labB`: blurred Lab color channels centered on 0.
 * - `printResidue`: closed minus raw lightness, large only on thin dark marks (print).
 * - `gradientX/Y/Magnitude`: Sobel of the lightly blurred lightness, divided by 4.
 * Element-wise math reproduces numpy's float32 rounding with `Math.fround`.
 */

import { oddKernelSize } from "../classical_detector_config.js";
import { withMats } from "../numeric/mat_helpers.js";
import { resizeAreaUint8 } from "./area_resize.js";
import { magnitudeFloat32 } from "./neon_magnitude.js";
import { gaussianBlurFloat32, gaussianKernelTaps, sobelFloat32 } from "./separable_filters_float32.js";

const GRADIENT_PRE_BLUR_SIGMA = 1.0;
const COLOR_BLUR_SIGMA = 1.5;
const LAB_CHROMA_CENTER = 128.0;
const SOBEL_NORMALIZATION = 4.0;
const fround = Math.fround;

/** Wrap a Float32Array as a signal map. */
function signalMap(values, width, height) {
  return { values, width, height };
}

/**
 * Downscale (never upscale) so the long side is at most the working size (INTER_AREA).
 *
 * Returns `{ workingBgr, scaleToFullResolution, ownsWorkingBgr }`; when no resize is needed
 * the input Mat itself is returned and must not be deleted by the caller of this helper. The
 * generic-ratio path is the arm64-exact JS port (`area_resize.js`); integer ratios use cv.resize.
 */
export function resizeToWorkingResolution(cv, imageBgr, workingLongSidePixels) {
  const fullLongSide = Math.max(imageBgr.rows, imageBgr.cols);
  if (fullLongSide <= workingLongSidePixels) {
    return { workingBgr: imageBgr, scaleToFullResolution: 1.0, ownsWorkingBgr: false };
  }
  const downscale = workingLongSidePixels / fullLongSide;
  const workingBgr = new cv.Mat();
  const exactResize = imageBgr.isContinuous()
    ? resizeAreaUint8(imageBgr.data, imageBgr.cols, imageBgr.rows, imageBgr.channels(), downscale)
    : null;
  if (exactResize === null) {
    cv.resize(imageBgr, workingBgr, new cv.Size(0, 0), downscale, downscale, cv.INTER_AREA);
  } else {
    workingBgr.create(exactResize.height, exactResize.width, imageBgr.type());
    workingBgr.data.set(exactResize.bytes);
  }
  return {
    workingBgr,
    scaleToFullResolution: fullLongSide / Math.max(workingBgr.rows, workingBgr.cols),
    ownsWorkingBgr: true,
  };
}

/** Box-window local standard deviation (boxFilter + sqrBoxFilter, float32 like numpy). */
export function localStandardDeviation(cv, values, width, height, windowSize) {
  return withMats((track) => {
    const sourceMat = track(new cv.Mat(height, width, cv.CV_32F));
    sourceMat.data32F.set(values);
    const window = new cv.Size(windowSize, windowSize);
    const meanMat = track(new cv.Mat());
    const meanOfSquaresMat = track(new cv.Mat());
    cv.boxFilter(sourceMat, meanMat, cv.CV_32F, window, new cv.Point(-1, -1), true, cv.BORDER_DEFAULT);
    cv.sqrBoxFilter(sourceMat, meanOfSquaresMat, cv.CV_32F, window, new cv.Point(-1, -1), true, cv.BORDER_DEFAULT);
    const localMeans = meanMat.data32F;
    const localMeansOfSquares = meanOfSquaresMat.data32F;
    const standardDeviations = new Float32Array(values.length);
    for (let index = 0; index < values.length; index += 1) {
      const localMean = localMeans[index];
      const variance = fround(localMeansOfSquares[index] - fround(localMean * localMean));
      standardDeviations[index] = fround(Math.sqrt(variance > 0 ? variance : 0));
    }
    return standardDeviations;
  });
}

/** Lab L (raw and text-closed) plus float32 a/b minus 128, from the working BGR. */
function computeLabPlanes(cv, workingBgr, textKernelSize, track) {
  const width = workingBgr.cols;
  const height = workingBgr.rows;
  const labMat = track(new cv.Mat());
  cv.cvtColor(workingBgr, labMat, cv.COLOR_BGR2Lab);
  const rawLightnessMat = track(new cv.Mat(height, width, cv.CV_8U));
  const pixelCount = width * height;
  const centeredA = new Float32Array(pixelCount);
  const centeredB = new Float32Array(pixelCount);
  const labBytes = labMat.data;
  const rawLightnessBytes = rawLightnessMat.data;
  for (let index = 0; index < pixelCount; index += 1) {
    rawLightnessBytes[index] = labBytes[3 * index];
    centeredA[index] = labBytes[3 * index + 1] - LAB_CHROMA_CENTER;
    centeredB[index] = labBytes[3 * index + 2] - LAB_CHROMA_CENTER;
  }
  const textKernel = track(cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(textKernelSize, textKernelSize)));
  const closedLightnessMat = track(new cv.Mat());
  cv.morphologyEx(rawLightnessMat, closedLightnessMat, cv.MORPH_CLOSE, textKernel);
  return {
    rawLightness: Uint8Array.from(rawLightnessMat.data),
    closedLightness: Uint8Array.from(closedLightnessMat.data),
    centeredA,
    centeredB,
  };
}

/** Sobel x/y of the blurred lightness (divided by 4) and their magnitude, arm64-exact. */
function computeGradients(cv, lightness, width, height) {
  const blurredLightness = gaussianBlurFloat32(lightness, width, height, gaussianKernelTaps(cv, GRADIENT_PRE_BLUR_SIGMA));
  const sobelX = sobelFloat32(blurredLightness, width, height, "x");
  const sobelY = sobelFloat32(blurredLightness, width, height, "y");
  const pixelCount = width * height;
  const gradientX = new Float32Array(pixelCount);
  const gradientY = new Float32Array(pixelCount);
  for (let index = 0; index < pixelCount; index += 1) {
    gradientX[index] = fround(sobelX[index] / SOBEL_NORMALIZATION);
    gradientY[index] = fround(sobelY[index] / SOBEL_NORMALIZATION);
  }
  const gradientMagnitude = magnitudeFloat32(gradientX, gradientY);
  return { gradientX, gradientY, gradientMagnitude };
}

/**
 * Every signal map the detector uses, from a full-resolution BGR Mat (not modified or freed).
 *
 * Returns an object with `width`, `height`, `longSidePixels`, `areaPixels`,
 * `scaleToFullResolution` and the signal maps listed in the module docstring.
 */
export function buildWorkingImageChannels(cv, imageBgr, config) {
  const { workingBgr, scaleToFullResolution, ownsWorkingBgr } = resizeToWorkingResolution(
    cv, imageBgr, config.workingLongSidePixels,
  );
  try {
    return withMats((track) => {
      const width = workingBgr.cols;
      const height = workingBgr.rows;
      const longSide = Math.max(width, height);
      const pixelCount = width * height;
      const textKernelSize = oddKernelSize(config.textSuppressionKernelFraction, longSide);
      const { rawLightness, closedLightness, centeredA, centeredB } = computeLabPlanes(cv, workingBgr, textKernelSize, track);

      const lightness = new Float32Array(pixelCount);
      const printResidue = new Float32Array(pixelCount);
      for (let index = 0; index < pixelCount; index += 1) {
        lightness[index] = closedLightness[index];
        const residue = closedLightness[index] - rawLightness[index];
        printResidue[index] = residue > 0 ? residue : 0; // cv2.subtract saturates at 0
      }
      const colorTaps = gaussianKernelTaps(cv, COLOR_BLUR_SIGMA);
      const labA = gaussianBlurFloat32(centeredA, width, height, colorTaps);
      const labB = gaussianBlurFloat32(centeredB, width, height, colorTaps);
      const chroma = new Float32Array(pixelCount);
      const paperScore = new Float32Array(pixelCount);
      const chromaWeight = fround(config.chromaWeightInPaperScore);
      for (let index = 0; index < pixelCount; index += 1) {
        const valueA = labA[index];
        const valueB = labB[index];
        const chromaValue = fround(Math.sqrt(fround(fround(valueA * valueA) + fround(valueB * valueB))));
        chroma[index] = chromaValue;
        paperScore[index] = fround(lightness[index] - fround(chromaWeight * chromaValue));
      }
      const textureWindow = oddKernelSize(config.textureWindowFraction, longSide);
      const textureStd = localStandardDeviation(cv, lightness, width, height, textureWindow);
      const { gradientX, gradientY, gradientMagnitude } = computeGradients(cv, lightness, width, height);

      return {
        width,
        height,
        longSidePixels: longSide,
        areaPixels: pixelCount,
        scaleToFullResolution,
        lightness: signalMap(lightness, width, height),
        chroma: signalMap(chroma, width, height),
        labA: signalMap(labA, width, height),
        labB: signalMap(labB, width, height),
        paperScore: signalMap(paperScore, width, height),
        printResidue: signalMap(printResidue, width, height),
        textureStd: signalMap(textureStd, width, height),
        gradientX: signalMap(gradientX, width, height),
        gradientY: signalMap(gradientY, width, height),
        gradientMagnitude: signalMap(gradientMagnitude, width, height),
      };
    });
  } finally {
    if (ownsWorkingBgr) workingBgr.delete();
  }
}
