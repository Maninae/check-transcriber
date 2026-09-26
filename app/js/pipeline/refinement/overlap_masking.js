/**
 * Keep a check's sides from locking onto the edges of OTHER detected checks.
 *
 * Port of experiments/detection/refinement/overlap_masking.py. Where check B lies over
 * check A, B's edge crosses A's search band and looks exactly like a paper edge. Pixels
 * cannot say which check is on top, so every score cell inside another detection's quad
 * carries no evidence (-Infinity); A's side is fitted on its clear stretches and a hidden
 * corner is extrapolated from them.
 *
 * - Only quads whose bounding box comes near this check are tested.
 * - Quads are [[x, y] x 4] in either winding.
 */

import { vectorNorm } from "./numpy_compatible_math.js";

const MINIMUM_DIRECTION_LENGTH = 1e-9;
const MINIMUM_LINE_DETERMINANT = 1e-9;

/** Centroid [x, y] of a quad. */
export function quadCentroid(corners) {
  return [
    (corners[0][0] + corners[1][0] + corners[2][0] + corners[3][0]) / 4,
    (corners[0][1] + corners[1][1] + corners[2][1] + corners[3][1]) / 4,
  ];
}

/** Move every side of a convex quad outward by `dilationPixels`; parallel neighbours keep the corner. */
export function dilateConvexQuad(corners, dilationPixels) {
  const centroid = quadCentroid(corners);
  const movedLines = [];
  for (let side = 0; side < 4; side += 1) {
    const start = corners[side];
    const end = corners[(side + 1) % 4];
    const length = Math.max(vectorNorm(end[0] - start[0], end[1] - start[1]), MINIMUM_DIRECTION_LENGTH);
    const direction = [(end[0] - start[0]) / length, (end[1] - start[1]) / length];
    let outwardNormal = [direction[1], -direction[0]];
    if (outwardNormal[0] * (start[0] - centroid[0]) + outwardNormal[1] * (start[1] - centroid[1]) < 0) outwardNormal = [-outwardNormal[0], -outwardNormal[1]];
    movedLines.push({ point: [start[0] + dilationPixels * outwardNormal[0], start[1] + dilationPixels * outwardNormal[1]], direction });
  }
  return corners.map((corner, cornerIndex) => {
    const first = movedLines[(cornerIndex + 3) % 4];
    const second = movedLines[cornerIndex];
    // Solve first.point + t * first.direction = second.point + u * second.direction for t.
    const determinant = first.direction[0] * -second.direction[1] - -second.direction[0] * first.direction[1];
    if (Math.abs(determinant) < MINIMUM_LINE_DETERMINANT) return [corner[0], corner[1]];
    const rightX = second.point[0] - first.point[0];
    const rightY = second.point[1] - first.point[1];
    const parameter = (rightX * -second.direction[1] - -second.direction[0] * rightY) / determinant;
    return [first.point[0] + parameter * first.direction[0], first.point[1] + parameter * first.direction[1]];
  });
}

/** True when (x, y) is strictly on the inner side of all four edges (either winding). */
export function pointInsideConvexQuad(x, y, corners) {
  let positive = 0;
  let negative = 0;
  for (let edge = 0; edge < 4; edge += 1) {
    const start = corners[edge];
    const end = corners[(edge + 1) % 4];
    const cross = (end[0] - start[0]) * (y - start[1]) - (end[1] - start[1]) * (x - start[0]);
    if (cross > 0) positive += 1;
    else if (cross < 0) negative += 1;
  }
  return positive === 4 || negative === 4;
}

/** Other quads whose bounding box comes within `reachPixels` of this quad's bounding box. */
export function selectNearbyQuads(ownCorners, otherQuads, reachPixels) {
  const xs = ownCorners.map((corner) => corner[0]);
  const ys = ownCorners.map((corner) => corner[1]);
  const [minimumX, maximumX] = [Math.min(...xs) - reachPixels, Math.max(...xs) + reachPixels];
  const [minimumY, maximumY] = [Math.min(...ys) - reachPixels, Math.max(...ys) + reachPixels];
  return otherQuads.filter((quad) => {
    const quadXs = quad.map((corner) => corner[0]);
    const quadYs = quad.map((corner) => corner[1]);
    return Math.max(...quadXs) >= minimumX && Math.max(...quadYs) >= minimumY && Math.min(...quadXs) <= maximumX && Math.min(...quadYs) <= maximumY;
  });
}

/** Set score cells lying inside any other (dilated) quad to -Infinity in place; returns how many. */
export function maskCellsInsideOtherQuads(profiles, dilatedOtherQuads) {
  if (dilatedOtherQuads.length === 0) return 0;
  const { numberOfSamples, numberOfOffsets, scores } = profiles;
  let maskedCount = 0;
  for (let sample = 0; sample < numberOfSamples; sample += 1) {
    for (let offset = 0; offset < numberOfOffsets; offset += 1) {
      const [x, y] = profiles.pointAt(profiles.positionsPixels[sample], profiles.normalOffsets[offset]);
      if (dilatedOtherQuads.some((quad) => pointInsideConvexQuad(x, y, quad))) {
        scores[sample * numberOfOffsets + offset] = -Infinity;
        maskedCount += 1;
      }
    }
  }
  return maskedCount;
}
