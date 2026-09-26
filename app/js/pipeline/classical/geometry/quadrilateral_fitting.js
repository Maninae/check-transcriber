/**
 * Fit a quadrilateral to a candidate region's contour (Python `quadrilateral_fitting.py`).
 *
 * 1. Coarse quad: `approxPolyN(hull, 4)`, the 4-gon enclosing the convex hull with the least
 *    added area. It always returns 4 corners and tolerates rounded corners and notches.
 * 2. Side refit: each contour point goes to its nearest coarse side; points near the middle
 *    of a side get a Huber `fitLine`, and adjacent lines are intersected (float32, as in the
 *    Python, since fitLine returns float32). This recovers corners placed outside a lifted
 *    or clipped corner.
 * Rectangularity (filled region area / quad area) lets selection reject merged blobs.
 * A fitted quad is `{ corners, rectangularity, sourceName }`.
 */

import { createFloat32PointMatFromFlat, fitLineHuber, withMats } from "../numeric/mat_helpers.js";
import { contourAreaOfFlatContour } from "../numeric/opencv_geometry_formulas.js";
import {
  intersectLines, orderCornersClockwise, quadrilateralArea, quadrilateralAspectRatio, vectorLength,
} from "./quadrilateral_geometry.js";

const SIDE_POINT_DISTANCE_FRACTION = 0.03; // contour points farther than this x side length are ignored
const SIDE_POINT_DISTANCE_SLACK_PIXELS = 2.0;
const SIDE_CORNER_EXCLUSION_FRACTION = 0.12; // skip each side's ends, where corners round off
const MINIMUM_POINTS_PER_SIDE = 8;
const MAXIMUM_CORNER_SHIFT_FRACTION = 0.15; // refit corner may move at most this x the shorter side
const COARSE_GATE_SLACK = 0.9; // coarse-quad pre-gates are this much looser than the final gates
const MAXIMUM_RECTANGULARITY = 1.5;
const MINIMUM_SQUARED_LENGTH = 1e-9;

/** Distance from each contour point to one segment (flat int contour, numpy formula). */
function distancesToSegment(flatPoints, pointCount, segmentStart, segmentEnd) {
  const segmentX = segmentEnd[0] - segmentStart[0];
  const segmentY = segmentEnd[1] - segmentStart[1];
  const segmentLengthSquared = Math.max(segmentX * segmentX + segmentY * segmentY, MINIMUM_SQUARED_LENGTH);
  const distances = new Float64Array(pointCount);
  for (let index = 0; index < pointCount; index += 1) {
    const offsetX = flatPoints[2 * index] - segmentStart[0];
    const offsetY = flatPoints[2 * index + 1] - segmentStart[1];
    let projection = (offsetX * segmentX + offsetY * segmentY) / segmentLengthSquared;
    projection = projection < 0 ? 0 : projection > 1 ? 1 : projection;
    const closestX = segmentStart[0] + projection * segmentX;
    const closestY = segmentStart[1] + projection * segmentY;
    distances[index] = vectorLength(flatPoints[2 * index] - closestX, flatPoints[2 * index + 1] - closestY);
  }
  return distances;
}

/** Refine a coarse quad by fitting a robust line to each side's contour points. */
function refitSidesWithLines(cv, flatPoints, coarseCorners) {
  const pointCount = flatPoints.length >> 1;
  const sideDistances = [0, 1, 2, 3].map((side) => distancesToSegment(flatPoints, pointCount, coarseCorners[side], coarseCorners[(side + 1) % 4]));
  const nearestSide = new Uint8Array(pointCount);
  for (let index = 0; index < pointCount; index += 1) {
    let bestSide = 0;
    for (let side = 1; side < 4; side += 1) if (sideDistances[side][index] < sideDistances[bestSide][index]) bestSide = side;
    nearestSide[index] = bestSide;
  }
  const fittedLines = [];
  const sidePoints = new Float32Array(2 * pointCount);
  for (let side = 0; side < 4; side += 1) {
    const sideStart = coarseCorners[side];
    const sideEnd = coarseCorners[(side + 1) % 4];
    const sideX = sideEnd[0] - sideStart[0];
    const sideY = sideEnd[1] - sideStart[1];
    const sideLength = vectorLength(sideX, sideY);
    const alongDenominator = Math.max(sideLength * sideLength, MINIMUM_SQUARED_LENGTH);
    const distanceLimit = SIDE_POINT_DISTANCE_FRACTION * sideLength + SIDE_POINT_DISTANCE_SLACK_PIXELS;
    let keptCount = 0;
    for (let index = 0; index < pointCount; index += 1) {
      const pointX = flatPoints[2 * index];
      const pointY = flatPoints[2 * index + 1];
      const alongFraction = ((pointX - sideStart[0]) * sideX + (pointY - sideStart[1]) * sideY) / alongDenominator;
      if (
        nearestSide[index] === side && sideDistances[side][index] < distanceLimit
        && alongFraction > SIDE_CORNER_EXCLUSION_FRACTION && alongFraction < 1.0 - SIDE_CORNER_EXCLUSION_FRACTION
      ) {
        sidePoints[2 * keptCount] = pointX;
        sidePoints[2 * keptCount + 1] = pointY;
        keptCount += 1;
      }
    }
    if (keptCount < MINIMUM_POINTS_PER_SIDE) return coarseCorners;
    fittedLines.push(fitLineHuber(cv, sidePoints, keptCount));
  }
  const refitCorners = [];
  for (let corner = 0; corner < 4; corner += 1) {
    const intersection = intersectLines(fittedLines[(corner + 3) % 4], fittedLines[corner], true);
    if (intersection === null) return coarseCorners;
    refitCorners.push(intersection);
  }
  const shorterSide = Math.min(...[0, 1, 2, 3].map((corner) => {
    const nextCorner = coarseCorners[(corner + 1) % 4];
    return vectorLength(nextCorner[0] - coarseCorners[corner][0], nextCorner[1] - coarseCorners[corner][1]);
  }));
  const largestShift = Math.max(...refitCorners.map((corner, index) => vectorLength(corner[0] - coarseCorners[index][0], corner[1] - coarseCorners[index][1])));
  if (largestShift > MAXIMUM_CORNER_SHIFT_FRACTION * shorterSide) return coarseCorners;
  return refitCorners;
}

/** Coarse quad: convex hull, then `approxPolyN(hull, 4)`; null when degenerate. */
function coarseQuadrilateral(cv, flatContour) {
  return withMats((track) => {
    const contourMat = track(createFloat32PointMatFromFlat(cv, Float32Array.from(flatContour)));
    const hullMat = track(new cv.Mat());
    cv.convexHull(contourMat, hullMat, false, true);
    if (hullMat.rows < 4) return null;
    const approximationMat = track(new cv.Mat());
    cv.approxPolyN(hullMat, approximationMat, 4, -1, true);
    if (approximationMat.rows * approximationMat.cols !== 4) return null;
    const cornerValues = approximationMat.type() === cv.CV_32SC2 ? approximationMat.data32S : approximationMat.data32F;
    return [0, 1, 2, 3].map((corner) => [cornerValues[2 * corner], cornerValues[2 * corner + 1]]);
  });
}

/**
 * Coarse approxPolyN quad plus a per-side line refit; null for degenerate regions.
 *
 * `minimumRectangularity` and `aspectRange` reject on the coarse quad before the costly
 * refit; pass loose values, the caller re-gates the refit quad.
 */
export function fitQuadrilateralToContour(cv, flatContour, sourceName, minimumRectangularity = 0.0, aspectRange = [0.0, Infinity]) {
  const rawCoarseCorners = coarseQuadrilateral(cv, flatContour);
  if (rawCoarseCorners === null) return null;
  const coarseCorners = orderCornersClockwise(rawCoarseCorners);
  const filledRegionArea = contourAreaOfFlatContour(flatContour);
  const coarseArea = quadrilateralArea(coarseCorners);
  if (coarseArea <= 0 || filledRegionArea / coarseArea < minimumRectangularity * COARSE_GATE_SLACK) return null;
  const coarseAspect = quadrilateralAspectRatio(coarseCorners);
  if (!(aspectRange[0] * COARSE_GATE_SLACK <= coarseAspect && coarseAspect <= aspectRange[1] / COARSE_GATE_SLACK)) return null;
  const refinedCorners = orderCornersClockwise(refitSidesWithLines(cv, flatContour, coarseCorners));
  const quadArea = quadrilateralArea(refinedCorners);
  if (quadArea <= 0) return null;
  return { corners: refinedCorners, rectangularity: Math.min(filledRegionArea / quadArea, MAXIMUM_RECTANGULARITY), sourceName };
}

