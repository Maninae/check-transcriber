/**
 * Stages 3-4 of the pipeline (spec section 5), run in the worker after the operator
 * confirms the count: for one confirmed quad, decide which way up the check is, then
 * warp it to its upright landscape crop.
 *
 * Orientation replaces the spec's OCR-based up/down vote with our own small classifier
 * (experiments/detection/REPORT.md: 99.3% on eval). The crop is returned upright; the
 * MICR band is blurred on the main thread at display time (review/crop_rendering.js).
 */

import {
  applyUpsideDownDecision,
  buildOrientationCropTensorData,
  rollCornersToLongSideFirst,
} from "./orientation/check_orientation.js";
import { classifyUpsideDownLogit, logitToProbability } from "./orientation/upside_down_classifier.js";
import { rectifyCheckToLandscapeCrop } from "./check_rectification.js";

/**
 * Resolves to `{ orientedCorners, upsideDownProbability, width, height, rgbaPixels }`.
 * `photo` is the worker's per-photo state: `{ imageBgr, workingGray, workingScale }`.
 * The classifier call is an onnxruntime (native) Promise; all OpenCV work is synchronous.
 */
export function orientAndRectifyCheck(cv, ort, classifierSession, photo, corners) {
  const landscapeCorners = rollCornersToLongSideFirst(corners);
  const cropTensorData = buildOrientationCropTensorData(cv, photo.workingGray, photo.workingScale, landscapeCorners);
  return classifyUpsideDownLogit(ort, classifierSession, cropTensorData).then((logit) => {
    const orientedCorners = applyUpsideDownDecision(landscapeCorners, logit > 0);
    const crop = rectifyCheckToLandscapeCrop(cv, photo.imageBgr, orientedCorners);
    return { orientedCorners, upsideDownProbability: logitToProbability(logit), ...crop };
  });
}
