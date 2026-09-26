/**
 * Perspective-warps one oriented check quad from the full-resolution photo to a flat
 * landscape crop (spec section 5, stage 3), `RECTIFIED_CROP_WIDTH` px wide.
 *
 * The height follows the quad's own measured aspect (mean of the two short sides over
 * the mean of the two long sides), so personal (6 x 2.75 in) and business (8.5 x 3.5 in)
 * checks both come out undistorted. Corners map pixel-centre to pixel-centre, the same
 * convention as the orientation crop. Cubic interpolation keeps small print crisp when a
 * check is upsampled from a low-resolution photo.
 */

export const RECTIFIED_CROP_WIDTH = 1600;
const MINIMUM_CROP_HEIGHT = 200;

function sideLength(pointA, pointB) {
  return Math.hypot(pointA[0] - pointB[0], pointA[1] - pointB[1]);
}

/**
 * `orientedCorners` start at the check's own top-left (TL, TR, BR, BL).
 * Returns `{ width, height, rgbaPixels: Uint8ClampedArray }` (a fresh buffer the caller may transfer).
 */
export function rectifyCheckToLandscapeCrop(cv, imageBgr, orientedCorners) {
  const [topLeft, topRight, bottomRight, bottomLeft] = orientedCorners;
  const meanWidth = (sideLength(topLeft, topRight) + sideLength(bottomLeft, bottomRight)) / 2;
  const meanHeight = (sideLength(topLeft, bottomLeft) + sideLength(topRight, bottomRight)) / 2;
  const width = RECTIFIED_CROP_WIDTH;
  const height = Math.max(MINIMUM_CROP_HEIGHT, Math.round((width * meanHeight) / meanWidth));

  const sourceMat = cv.matFromArray(4, 1, cv.CV_32FC2, orientedCorners.flat());
  const destinationMat = cv.matFromArray(4, 1, cv.CV_32FC2, [0, 0, width - 1, 0, width - 1, height - 1, 0, height - 1]);
  const homography = cv.getPerspectiveTransform(sourceMat, destinationMat);
  const warpedBgr = new cv.Mat();
  const warpedRgba = new cv.Mat();
  try {
    cv.warpPerspective(imageBgr, warpedBgr, homography, new cv.Size(width, height), cv.INTER_CUBIC, cv.BORDER_REPLICATE, new cv.Scalar());
    cv.cvtColor(warpedBgr, warpedRgba, cv.COLOR_BGR2RGBA);
    return { width, height, rgbaPixels: new Uint8ClampedArray(warpedRgba.data) };
  } finally {
    sourceMat.delete();
    destinationMat.delete();
    homography.delete();
    warpedBgr.delete();
    warpedRgba.delete();
  }
}
