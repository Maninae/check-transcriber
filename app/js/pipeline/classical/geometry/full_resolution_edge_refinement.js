/**
 * Final sub-pixel corners: snap each side to the paper edge on the original image (Python
 * `full_resolution_edge_refinement.py`).
 *
 * The working-resolution quad is accurate to a working pixel or two (2-5 full-res pixels).
 * We crop the quad's neighbourhood from the full-res image, convert just that crop to gray
 * (identical per pixel to converting the whole photo), erase thin dark print with a grayscale
 * closing, blur, and run `edge_line_snapping` with a search radius proportional to image size.
 * The crop is a fresh Mat (not a ROI view), so closing and blur see crop-edge borders exactly
 * as the Python does on its numpy slice.
 */

import { withMats } from "../numeric/mat_helpers.js";
import { roundHalfEven } from "../numeric/numpy_compatibility.js";
import { snapQuadrilateralSidesToEdges } from "./edge_line_snapping.js";

const CROP_MARGIN_PIXELS = 40;
const TEXT_SUPPRESSION_KERNEL_PIXELS = 5;
const CROP_BLUR_SIGMA = 1.0;
const MINIMUM_CROP_PIXELS = 8;
const MINIMUM_SEARCH_RADIUS_PIXELS = 3;

/**
 * Refined four corners in full-res pixels; each side falls back to its input line.
 *
 * Args:
 *   imageBgr: full-resolution CV_8UC3 BGR Mat (read only).
 *   corners: four `[x, y]` full-resolution corners.
 */
export function refineCornersAtFullResolution(cv, imageBgr, corners, config) {
  const imageWidth = imageBgr.cols;
  const imageHeight = imageBgr.rows;
  const searchRadius = Math.max(
    MINIMUM_SEARCH_RADIUS_PIXELS, Math.trunc(roundHalfEven(config.refinementSearchFraction * Math.max(imageWidth, imageHeight))),
  );
  const cornerXs = corners.map(([x]) => x);
  const cornerYs = corners.map(([, y]) => y);
  const left = Math.trunc(Math.max(0, Math.floor(Math.min(...cornerXs)) - CROP_MARGIN_PIXELS));
  const top = Math.trunc(Math.max(0, Math.floor(Math.min(...cornerYs)) - CROP_MARGIN_PIXELS));
  const right = Math.trunc(Math.min(imageWidth, Math.ceil(Math.max(...cornerXs)) + CROP_MARGIN_PIXELS));
  const bottom = Math.trunc(Math.min(imageHeight, Math.ceil(Math.max(...cornerYs)) + CROP_MARGIN_PIXELS));
  if (right - left < MINIMUM_CROP_PIXELS || bottom - top < MINIMUM_CROP_PIXELS) return corners;
  const cropIntensity = withMats((track) => {
    const cropView = track(imageBgr.roi(new cv.Rect(left, top, right - left, bottom - top)));
    const cropGray = track(new cv.Mat());
    cv.cvtColor(cropView, cropGray, cv.COLOR_BGR2GRAY);
    const kernel = track(cv.getStructuringElement(
      cv.MORPH_ELLIPSE, new cv.Size(TEXT_SUPPRESSION_KERNEL_PIXELS, TEXT_SUPPRESSION_KERNEL_PIXELS),
    ));
    const closedGray = track(new cv.Mat());
    cv.morphologyEx(cropGray, closedGray, cv.MORPH_CLOSE, kernel);
    const blurredGray = track(new cv.Mat());
    cv.GaussianBlur(closedGray, blurredGray, new cv.Size(0, 0), CROP_BLUR_SIGMA, 0, cv.BORDER_DEFAULT);
    return { values: Float32Array.from(blurredGray.data), width: right - left, height: bottom - top };
  });
  const cropCorners = corners.map(([x, y]) => [x - left, y - top]);
  const snappedCropCorners = snapQuadrilateralSidesToEdges(
    cv, cropIntensity, cropCorners, searchRadius, config.refinementSamplesPerSide,
    config.refinementMinimumStepStrength, [imageWidth, imageHeight], [left, top],
  );
  return snappedCropCorners.map(([x, y]) => [x + left, y + top]);
}
