/**
 * Pure geometry helpers for check quadrilaterals (Python `quadrilateral_geometry.py`).
 *
 * Quads are arrays of four `[x, y]` pairs (float64). Everything here is free of image state.
 * Area and convexity use the ported OpenCV formulas on float32-cast corners (as the Python
 * casts before calling cv2), so no Mat is allocated on these hot paths; only the polygon
 * intersection calls into OpenCV.js.
 */

import { withMats, createFloat32PointMat } from "../numeric/mat_helpers.js";
import { argminFirst, numpyArgsort } from "../numeric/numpy_compatibility.js";
import { contourAreaOfPoints, isContourConvexFloat32 } from "../numeric/opencv_geometry_formulas.js";

const NEAR_PARALLEL_DETERMINANT = 1e-9;
const MINIMUM_ASPECT_DENOMINATOR = 1e-6;
const ANGLE_COSINE_EPSILON = 1e-9;
const RADIANS_TO_DEGREES = 180 / Math.PI;
const fround = Math.fround;

/** Euclidean length of (deltaX, deltaY) the way `np.linalg.norm` computes it. */
export function vectorLength(deltaX, deltaY) {
  return Math.sqrt(deltaX * deltaX + deltaY * deltaY);
}

/**
 * The four corners clockwise in image coordinates (y down), smallest x + y first.
 *
 * Sorting by angle around the centroid gives a cyclic order (increasing angle is clockwise
 * when y points down); the start corner is the one with the smallest x + y.
 */
export function orderCornersClockwise(corners) {
  const centroidX = (corners[0][0] + corners[1][0] + corners[2][0] + corners[3][0]) / 4;
  const centroidY = (corners[0][1] + corners[1][1] + corners[2][1] + corners[3][1]) / 4;
  const angles = corners.map(([x, y]) => Math.atan2(y - centroidY, x - centroidX));
  const clockwiseCorners = Array.from(numpyArgsort(angles), (index) => [corners[index][0], corners[index][1]]);
  const startIndex = argminFirst(clockwiseCorners.map(([x, y]) => x + y));
  return clockwiseCorners.slice(startIndex).concat(clockwiseCorners.slice(0, startIndex));
}

/** Lengths of the four sides, side i running from corner i to corner i+1. */
export function quadrilateralSideLengths(corners) {
  return corners.map((corner, index) => {
    const nextCorner = corners[(index + 1) % 4];
    return vectorLength(nextCorner[0] - corner[0], nextCorner[1] - corner[1]);
  });
}

/** Long over short dimension, averaging each pair of opposite sides (>= 1). */
export function quadrilateralAspectRatio(corners) {
  const sideLengths = quadrilateralSideLengths(corners);
  const firstPair = (sideLengths[0] + sideLengths[2]) / 2.0;
  const secondPair = (sideLengths[1] + sideLengths[3]) / 2.0;
  return Math.max(firstPair, secondPair) / Math.max(Math.min(firstPair, secondPair), MINIMUM_ASPECT_DENOMINATOR);
}

/** Unsigned polygon area (cv2.contourArea of the float32-cast corners). */
export function quadrilateralArea(corners) {
  return contourAreaOfPoints(corners);
}

/** Interior angle at each corner, in degrees. */
export function quadrilateralInteriorAnglesDegrees(corners) {
  return corners.map((corner, index) => {
    const previousCorner = corners[(index + 3) % 4];
    const nextCorner = corners[(index + 1) % 4];
    const previousX = previousCorner[0] - corner[0];
    const previousY = previousCorner[1] - corner[1];
    const nextX = nextCorner[0] - corner[0];
    const nextY = nextCorner[1] - corner[1];
    const cosine = (previousX * nextX + previousY * nextY)
      / (vectorLength(previousX, previousY) * vectorLength(nextX, nextY) + ANGLE_COSINE_EPSILON);
    return Math.acos(Math.min(1.0, Math.max(-1.0, cosine))) * RADIANS_TO_DEGREES;
  });
}

/** True when the four corners form a convex, non-self-intersecting polygon. */
export function isConvexQuadrilateral(corners) {
  return isContourConvexFloat32(corners);
}

/** Area shared by two convex polygons (cv.intersectConvexConvex, nested handled). */
export function convexPolygonIntersectionArea(cv, firstCorners, secondCorners) {
  return withMats((track) => {
    const firstMat = track(createFloat32PointMat(cv, firstCorners));
    const secondMat = track(createFloat32PointMat(cv, secondCorners));
    const intersectionMat = track(new cv.Mat());
    const intersectionArea = cv.intersectConvexConvex(firstMat, secondMat, intersectionMat, true);
    return Math.max(intersectionArea, 0.0);
  });
}

/** Intersection over union of two convex quadrilaterals. */
export function convexQuadrilateralIou(cv, firstCorners, secondCorners) {
  const intersectionArea = convexPolygonIntersectionArea(cv, firstCorners, secondCorners);
  const unionArea = quadrilateralArea(firstCorners) + quadrilateralArea(secondCorners) - intersectionArea;
  return unionArea > 0 ? intersectionArea / unionArea : 0.0;
}

/**
 * Intersection of two lines `[vx, vy, x0, y0]`; null when near-parallel.
 *
 * `useFloat32Arithmetic` reproduces numpy float32 scalar math, for lines that come straight
 * from `cv2.fitLine` (float32) as in the Python region side refit.
 */
export function intersectLines(firstLine, secondLine, useFloat32Arithmetic = false) {
  const [firstDirectionX, firstDirectionY, firstPointX, firstPointY] = firstLine;
  const [secondDirectionX, secondDirectionY, secondPointX, secondPointY] = secondLine;
  if (useFloat32Arithmetic) {
    const determinant = fround(fround(firstDirectionX * fround(-secondDirectionY)) - fround(firstDirectionY * fround(-secondDirectionX)));
    if (Math.abs(determinant) < NEAR_PARALLEL_DETERMINANT) return null;
    const offsetX = fround(secondPointX - firstPointX);
    const offsetY = fround(secondPointY - firstPointY);
    const firstParameter = fround(
      fround(fround(offsetX * fround(-secondDirectionY)) - fround(offsetY * fround(-secondDirectionX))) / determinant,
    );
    return [fround(firstPointX + fround(firstParameter * firstDirectionX)), fround(firstPointY + fround(firstParameter * firstDirectionY))];
  }
  const determinant = firstDirectionX * -secondDirectionY - firstDirectionY * -secondDirectionX;
  if (Math.abs(determinant) < NEAR_PARALLEL_DETERMINANT) return null;
  const offsetX = secondPointX - firstPointX;
  const offsetY = secondPointY - firstPointY;
  const firstParameter = (offsetX * -secondDirectionY - offsetY * -secondDirectionX) / determinant;
  return [firstPointX + firstParameter * firstDirectionX, firstPointY + firstParameter * firstDirectionY];
}
