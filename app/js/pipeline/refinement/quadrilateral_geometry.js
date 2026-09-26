/**
 * Quad measurements and the guard rails that reject implausible refinements.
 *
 * Corners are [[x, y] x 4] in image coordinates (y down). Port of the geometry helpers and
 * `apply_guard_rails` in experiments/detection/refinement/quadrilateral_refinement.py.
 */

import { vectorNorm } from "./numpy_compatible_math.js";

const MINIMUM_SIDE_LENGTH_DIVISOR = 1e-9;

/** Length of each side i (corner i -> corner i + 1). */
export function computeSideLengths(corners) {
  return corners.map((corner, index) => {
    const next = corners[(index + 1) % 4];
    return vectorNorm(next[0] - corner[0], next[1] - corner[1]);
  });
}

/** Mean length of the shorter pair of opposite sides. */
export function computeShortSideLength(corners) {
  const sideLengths = computeSideLengths(corners);
  return Math.min(sideLengths[0] + sideLengths[2], sideLengths[1] + sideLengths[3]) / 2;
}

/** Shoelace signed area (positive for clockwise order in image coordinates). */
export function computeSignedArea(corners) {
  let forward = 0;
  let backward = 0;
  for (let index = 0; index < 4; index += 1) {
    const next = corners[(index + 1) % 4];
    forward += corners[index][0] * next[1];
    backward += next[0] * corners[index][1];
  }
  return 0.5 * (forward - backward);
}

/** True when every turn has the same sign as `expectedSign` (convex, same winding). */
export function isConvexWithOrientation(corners, expectedSign) {
  const edges = corners.map((corner, index) => [corners[(index + 1) % 4][0] - corner[0], corners[(index + 1) % 4][1] - corner[1]]);
  return edges.every((edge, index) => {
    const nextEdge = edges[(index + 1) % 4];
    return (edge[0] * nextEdge[1] - edge[1] * nextEdge[0]) * expectedSign > 0;
  });
}

/**
 * Revert corners (or the whole quad) whose refinement looks implausible, vs the INPUT quad.
 *
 * - A corner that moved further than `maximumCornerMoveBandMultiple * firstBand`, or is not
 *   finite, reverts to its input position.
 * - A non-convex or flipped quad, or one whose side length changed by more than
 *   `maximumSideLengthChangeFraction`, reverts entirely.
 * Records `cornersRevertedForDistance` and `quadReverted` in `diagnostics`.
 */
export function applyGuardRails(inputCorners, refinedCorners, firstBand, config, diagnostics) {
  const tooFar = refinedCorners.map((corner, index) => {
    const move = vectorNorm(corner[0] - inputCorners[index][0], corner[1] - inputCorners[index][1]);
    return move > config.maximumCornerMoveBandMultiple * firstBand || !(Number.isFinite(corner[0]) && Number.isFinite(corner[1]));
  });
  const guarded = refinedCorners.map((corner, index) => (tooFar[index] ? [...inputCorners[index]] : [...corner]));
  diagnostics.cornersRevertedForDistance = tooFar;
  const inputSideLengths = computeSideLengths(inputCorners);
  const guardedSideLengths = computeSideLengths(guarded);
  const lengthsPlausible = guardedSideLengths.every(
    (length, index) => Math.abs(length / Math.max(inputSideLengths[index], MINIMUM_SIDE_LENGTH_DIVISOR) - 1) <= config.maximumSideLengthChangeFraction,
  );
  const plausible = isConvexWithOrientation(guarded, Math.sign(computeSignedArea(inputCorners))) && lengthsPlausible;
  diagnostics.quadReverted = !plausible;
  return plausible ? guarded : inputCorners.map((corner) => [...corner]);
}
