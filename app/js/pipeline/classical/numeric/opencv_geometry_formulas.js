/**
 * OpenCV contour formulas ported to plain JS, so hot per-candidate gates allocate no Mats.
 *
 * Each function reproduces the OpenCV 5 C++ arithmetic on the same float32-cast points the
 * Python passes (`np.asarray(corners, dtype=np.float32)`), so results match cv2 exactly:
 * - `contourAreaOfPoints`: `cv::contourArea` (shoelace accumulated in double, float32 points).
 * - `isContourConvexFloat32`: `cv::isContourConvex` for Point2f (float32 cross-product signs;
 *   a collinear triple counts as not convex).
 * Points are flat arrays `[x0, y0, x1, y1, ...]` or arrays of `[x, y]` pairs (see each doc).
 */

const fround = Math.fround;

/**
 * Unsigned polygon area of `[[x, y], ...]` points after a float32 cast (cv2.contourArea).
 *
 * Integer contours are exact in float32 below 2^24, so the same routine covers int32 contours.
 */
export function contourAreaOfPoints(points) {
  const pointCount = points.length;
  if (pointCount === 0) return 0;
  let previousX = fround(points[pointCount - 1][0]);
  let previousY = fround(points[pointCount - 1][1]);
  let doubledArea = 0;
  for (let index = 0; index < pointCount; index += 1) {
    const currentX = fround(points[index][0]);
    const currentY = fround(points[index][1]);
    doubledArea += previousX * currentY - previousY * currentX;
    previousX = currentX;
    previousY = currentY;
  }
  return Math.abs(doubledArea * 0.5);
}

/** `cv2.contourArea` of a flat Int32Array contour `[x0, y0, x1, y1, ...]`. */
export function contourAreaOfFlatContour(flatPoints) {
  const pointCount = flatPoints.length >> 1;
  if (pointCount === 0) return 0;
  let previousX = flatPoints[2 * pointCount - 2];
  let previousY = flatPoints[2 * pointCount - 1];
  let doubledArea = 0;
  for (let index = 0; index < pointCount; index += 1) {
    const currentX = flatPoints[2 * index];
    const currentY = flatPoints[2 * index + 1];
    doubledArea += previousX * currentY - previousY * currentX;
    previousX = currentX;
    previousY = currentY;
  }
  return Math.abs(doubledArea * 0.5);
}

/** `cv2.isContourConvex` of `[[x, y], ...]` points cast to float32. */
export function isContourConvexFloat32(points) {
  const pointCount = points.length;
  let previousX = fround(points[(pointCount - 2 + pointCount) % pointCount][0]);
  let previousY = fround(points[(pointCount - 2 + pointCount) % pointCount][1]);
  let currentX = fround(points[pointCount - 1][0]);
  let currentY = fround(points[pointCount - 1][1]);
  let previousDeltaX = fround(currentX - previousX);
  let previousDeltaY = fround(currentY - previousY);
  let orientation = 0;
  for (let index = 0; index < pointCount; index += 1) {
    previousX = currentX;
    previousY = currentY;
    currentX = fround(points[index][0]);
    currentY = fround(points[index][1]);
    const deltaX = fround(currentX - previousX);
    const deltaY = fround(currentY - previousY);
    const deltaXTimesPreviousDeltaY = fround(deltaX * previousDeltaY);
    const deltaYTimesPreviousDeltaX = fround(deltaY * previousDeltaX);
    if (deltaYTimesPreviousDeltaX > deltaXTimesPreviousDeltaY) orientation |= 1;
    else if (deltaYTimesPreviousDeltaX < deltaXTimesPreviousDeltaY) orientation |= 2;
    else orientation |= 3;
    if (orientation === 3) return false;
    previousDeltaX = deltaX;
    previousDeltaY = deltaY;
  }
  return true;
}
