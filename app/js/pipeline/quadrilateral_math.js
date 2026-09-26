/**
 * Small pure-JS geometry on check quadrilaterals, shared by the pipeline stages and the
 * count-step UI. No OpenCV: these run on the main thread too.
 *
 * A quadrilateral ("quad") is always `[[x, y], [x, y], [x, y], [x, y]]` in full-resolution
 * photo pixels, clockwise in image coordinates (y points down).
 */

/** Shoelace area; positive for clockwise order in image coordinates (y down). */
export function computeSignedArea(corners) {
  let doubledArea = 0;
  for (let index = 0; index < corners.length; index += 1) {
    const [x1, y1] = corners[index];
    const [x2, y2] = corners[(index + 1) % corners.length];
    doubledArea += x1 * y2 - x2 * y1;
  }
  return doubledArea / 2;
}

/** Euclidean distance between two points. */
export function distanceBetweenPoints(pointA, pointB) {
  return Math.hypot(pointA[0] - pointB[0], pointA[1] - pointB[1]);
}

/** Returns the corners reordered clockwise (image coordinates), starting from the same corner. */
export function orderCornersClockwise(corners) {
  return computeSignedArea(corners) >= 0 ? corners.map((corner) => [...corner]) : [corners[0], corners[3], corners[2], corners[1]].map((corner) => [...corner]);
}

/** Centre of mass of the four corners. */
export function computeQuadCentre(corners) {
  const sum = corners.reduce((total, [x, y]) => [total[0] + x, total[1] + y], [0, 0]);
  return [sum[0] / corners.length, sum[1] / corners.length];
}

/** Axis-aligned bounds `{ minX, minY, maxX, maxY }`. */
export function computeQuadBounds(corners) {
  const xs = corners.map(([x]) => x);
  const ys = corners.map(([, y]) => y);
  return { minX: Math.min(...xs), minY: Math.min(...ys), maxX: Math.max(...xs), maxY: Math.max(...ys) };
}

/** Mean lengths of the two opposite-side pairs: `{ longSide, shortSide }`. */
export function measureQuadSideLengths(corners) {
  const firstPair = (distanceBetweenPoints(corners[0], corners[1]) + distanceBetweenPoints(corners[2], corners[3])) / 2;
  const secondPair = (distanceBetweenPoints(corners[1], corners[2]) + distanceBetweenPoints(corners[3], corners[0])) / 2;
  return { longSide: Math.max(firstPair, secondPair), shortSide: Math.min(firstPair, secondPair) };
}

/** Rolls the clockwise corner list so it starts at `startIndex`. */
export function rollCorners(corners, startIndex) {
  return corners.map((unused, index) => [...corners[(index + startIndex) % corners.length]]);
}

/** Sutherland-Hodgman clip of convex polygon `subject` by convex clockwise polygon `clip`. */
function clipConvexPolygon(subject, clip) {
  let output = subject;
  for (let edgeIndex = 0; edgeIndex < clip.length && output.length > 0; edgeIndex += 1) {
    const edgeStart = clip[edgeIndex];
    const edgeEnd = clip[(edgeIndex + 1) % clip.length];
    // Clockwise in y-down coordinates: the inside is where the cross product is >= 0.
    const isInside = ([x, y]) =>
      (edgeEnd[0] - edgeStart[0]) * (y - edgeStart[1]) - (edgeEnd[1] - edgeStart[1]) * (x - edgeStart[0]) >= 0;
    const intersect = (pointA, pointB) => {
      const edgeDx = edgeEnd[0] - edgeStart[0];
      const edgeDy = edgeEnd[1] - edgeStart[1];
      const segmentDx = pointB[0] - pointA[0];
      const segmentDy = pointB[1] - pointA[1];
      const denominator = edgeDx * segmentDy - edgeDy * segmentDx;
      const t = (edgeDy * (pointA[0] - edgeStart[0]) - edgeDx * (pointA[1] - edgeStart[1])) / denominator;
      return [pointA[0] + t * segmentDx, pointA[1] + t * segmentDy];
    };
    const input = output;
    output = [];
    for (let index = 0; index < input.length; index += 1) {
      const current = input[index];
      const previous = input[(index + input.length - 1) % input.length];
      if (isInside(current)) {
        if (!isInside(previous)) output.push(intersect(previous, current));
        output.push(current);
      } else if (isInside(previous)) {
        output.push(intersect(previous, current));
      }
    }
  }
  return output;
}

/** Intersection area of two convex quads (either winding). */
export function computeIntersectionArea(cornersA, cornersB) {
  const clockwiseA = orderCornersClockwise(cornersA);
  const clockwiseB = orderCornersClockwise(cornersB);
  const overlap = clipConvexPolygon(clockwiseA, clockwiseB);
  return overlap.length < 3 ? 0 : Math.abs(computeSignedArea(overlap));
}

/** Intersection over union of two convex quads. */
export function computeQuadIoU(cornersA, cornersB) {
  const intersectionArea = computeIntersectionArea(cornersA, cornersB);
  const unionArea = Math.abs(computeSignedArea(cornersA)) + Math.abs(computeSignedArea(cornersB)) - intersectionArea;
  return unionArea > 0 ? intersectionArea / unionArea : 0;
}

/** Whether the quad is convex (all turns in the same direction). */
export function isConvexQuad(corners) {
  let sign = 0;
  for (let index = 0; index < 4; index += 1) {
    const [ax, ay] = corners[index];
    const [bx, by] = corners[(index + 1) % 4];
    const [cx, cy] = corners[(index + 2) % 4];
    const cross = (bx - ax) * (cy - by) - (by - ay) * (cx - bx);
    if (cross === 0) continue;
    if (sign === 0) sign = Math.sign(cross);
    else if (Math.sign(cross) !== sign) return false;
  }
  return sign !== 0;
}
