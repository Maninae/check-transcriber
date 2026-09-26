/**
 * The count step's "confident" cue (spec 4.2): a detection is confident when all four of
 * its sides had strong border evidence, its aspect ratio is in the check range, and it
 * overlaps no other detection. When every detection is confident the header says so
 * and the operator can Continue on a glance; the step is still never skipped.
 *
 * These are UI cues, not gates: nothing is dropped here.
 */

import { computeIntersectionArea, computeSignedArea, measureQuadSideLengths } from "./quadrilateral_math.js";

// Fraction of boundary samples on a side with edge evidence (the classical detector's
// `sideSupports`); verification already requires 0.45 on the second-weakest side.
const MINIMUM_SIDE_SUPPORT_FOR_CONFIDENCE = 0.7;
// Personal checks are 6 x 2.75 in (2.18), business 8.5 x 3.5 in (2.43); allow perspective.
const CONFIDENT_ASPECT_RANGE = [1.9, 2.8];
// Overlap as a fraction of the smaller quad's area above which neither quad is confident.
const MAXIMUM_OVERLAP_FRACTION = 0.01;

/** Whether one detection (with `corners` and optional `sideSupports`) is confident among `allCornerSets`. */
export function isDetectionConfident(detection, allCornerSets) {
  const { longSide, shortSide } = measureQuadSideLengths(detection.corners);
  const aspectRatio = longSide / Math.max(shortSide, 1);
  if (aspectRatio < CONFIDENT_ASPECT_RANGE[0] || aspectRatio > CONFIDENT_ASPECT_RANGE[1]) return false;
  if (!detection.sideSupports || Math.min(...detection.sideSupports) < MINIMUM_SIDE_SUPPORT_FOR_CONFIDENCE) return false;
  const ownArea = Math.abs(computeSignedArea(detection.corners));
  return allCornerSets.every((otherCorners) => {
    if (otherCorners === detection.corners) return true;
    const smallerArea = Math.min(ownArea, Math.abs(computeSignedArea(otherCorners)));
    return computeIntersectionArea(detection.corners, otherCorners) <= MAXIMUM_OVERLAP_FRACTION * smallerArea;
  });
}
