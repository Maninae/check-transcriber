/**
 * Decides whether each detected check is upright or upside down, a port of
 * experiments/detection/orientation/ (rectify_check_crop.py + assign_check_orientation.py).
 *
 * Per photo, once: a grayscale copy scaled so the long side is 1280 px (the scale the
 * classifier's training crops were cut at, so text size matches). Per check:
 * 1. roll the clockwise corners so side 0->1 is a long side (landscape),
 * 2. warp that quad to a 224 x 96 grayscale crop, scale to [0, 1],
 * 3. the classifier (upside_down_classifier.js) returns a logit, > 0 = upside down,
 * 4. if upside down, roll the corners by two so they start at the check's own top-left.
 *
 * Every step matches the Python exactly (same resize and warp flags, same corner scaling
 * with no pixel-centre shift), because the classifier was trained on the Python's crops.
 */

const ORIENTATION_WORKING_LONG_SIDE_PIXELS = 1280;
export const ORIENTATION_CROP_WIDTH = 224;
export const ORIENTATION_CROP_HEIGHT = 96;
const PIXEL_VALUE_SCALE = 255;

function sideLength(pointA, pointB) {
  return Math.hypot(pointA[0] - pointB[0], pointA[1] - pointB[1]);
}

/** Roll the clockwise corner order so side 0->1 is a long side (the Python's exact rule). */
export function rollCornersToLongSideFirst(corners) {
  const firstPairLength = sideLength(corners[1], corners[0]) + sideLength(corners[2], corners[3]);
  const secondPairLength = sideLength(corners[2], corners[1]) + sideLength(corners[3], corners[0]);
  const startIndex = firstPairLength >= secondPairLength ? 0 : 1;
  return corners.map((unused, index) => [...corners[(index + startIndex) % 4]]);
}

/** Corners starting at the check's own top-left, given the classifier's decision. */
export function applyUpsideDownDecision(landscapeCorners, isUpsideDown) {
  const startIndex = isUpsideDown ? 2 : 0;
  return landscapeCorners.map((unused, index) => [...landscapeCorners[(index + startIndex) % 4]]);
}

/**
 * The per-photo grayscale working copy: `{ workingGray: cv.Mat, workingScale }`.
 * The caller deletes `workingGray` when the photo is released.
 */
export function buildOrientationWorkingGray(cv, imageBgr) {
  const workingScale = Math.min(1, ORIENTATION_WORKING_LONG_SIDE_PIXELS / Math.max(imageBgr.rows, imageBgr.cols));
  const resizedBgr = new cv.Mat();
  const workingGray = new cv.Mat();
  try {
    cv.resize(imageBgr, resizedBgr, new cv.Size(0, 0), workingScale, workingScale, cv.INTER_AREA);
    cv.cvtColor(resizedBgr, workingGray, cv.COLOR_BGR2GRAY);
  } finally {
    resizedBgr.delete();
  }
  return { workingGray, workingScale };
}

/** The classifier input for one landscape quad: Float32Array of 96 x 224 values in [0, 1]. */
export function buildOrientationCropTensorData(cv, workingGray, workingScale, landscapeCorners) {
  const scaledCorners = landscapeCorners.flatMap(([x, y]) => [x * workingScale, y * workingScale]);
  const destinationCorners = [
    0, 0,
    ORIENTATION_CROP_WIDTH - 1, 0,
    ORIENTATION_CROP_WIDTH - 1, ORIENTATION_CROP_HEIGHT - 1,
    0, ORIENTATION_CROP_HEIGHT - 1,
  ];
  const sourceMat = cv.matFromArray(4, 1, cv.CV_32FC2, scaledCorners);
  const destinationMat = cv.matFromArray(4, 1, cv.CV_32FC2, destinationCorners);
  const homography = cv.getPerspectiveTransform(sourceMat, destinationMat);
  const crop = new cv.Mat();
  try {
    cv.warpPerspective(
      workingGray, crop, homography, new cv.Size(ORIENTATION_CROP_WIDTH, ORIENTATION_CROP_HEIGHT),
      cv.INTER_AREA, cv.BORDER_REPLICATE, new cv.Scalar(),
    );
    const tensorData = new Float32Array(ORIENTATION_CROP_WIDTH * ORIENTATION_CROP_HEIGHT);
    for (let index = 0; index < tensorData.length; index += 1) {
      tensorData[index] = crop.data[index] / PIXEL_VALUE_SCALE;
    }
    return tensorData;
  } finally {
    sourceMat.delete();
    destinationMat.delete();
    homography.delete();
    crop.delete();
  }
}
