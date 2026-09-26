/**
 * Detect a straight border running across a candidate's interior, a "seam" (Python
 * `interior_seam_detection.py`).
 *
 * A quad assembled from line segments can span two checks plus the background between them:
 * its sides borrow real borders, so edge support is high. Its tell is a straight, strong step
 * crossing the interior parallel to a side. A real check has none (printed rules are erased
 * by the text-suppression closing; a hand-shadow boundary is soft and curved). For both side
 * directions we sample lines across the quad (interpolated between opposite sides, so
 * perspective is respected) and record the fraction of samples whose gradient along the
 * line's normal exceeds the threshold. Seam strength is the maximum over all lines.
 */

import { sampleBilinearReplicate } from "../numeric/bilinear_sampling.js";
import { linspace } from "../numeric/numpy_compatibility.js";
import { vectorLength } from "../geometry/quadrilateral_geometry.js";

const SEAM_LINE_FRACTIONS = linspace(0.12, 0.88, 17); // positions across the quad, away from its own sides
const SAMPLES_PER_SEAM_LINE = 48;
const SEAM_LINE_END_MARGIN = 0.04; // skip where the line meets the quad's own sides
const SEAM_GRADIENT_SEARCH_OFFSETS = [-1.5, 0.0, 1.5];
const MINIMUM_LINE_LENGTH_PIXELS = 1.0;
const ALONG_FRACTIONS = linspace(SEAM_LINE_END_MARGIN, 1.0 - SEAM_LINE_END_MARGIN, SAMPLES_PER_SEAM_LINE);

/** |gradient . normal| at a point, maximized over a few pixels of normal offset. */
function gradientAlongNormal(channels, pointX, pointY, normalX, normalY) {
  const { gradientX, gradientY, width, height } = channels;
  let bestResponse = 0;
  for (const offset of SEAM_GRADIENT_SEARCH_OFFSETS) {
    const shiftedX = pointX + offset * normalX;
    const shiftedY = pointY + offset * normalY;
    const valueX = sampleBilinearReplicate(gradientX.values, width, height, shiftedX, shiftedY);
    const valueY = sampleBilinearReplicate(gradientY.values, width, height, shiftedX, shiftedY);
    bestResponse = Math.max(bestResponse, Math.abs(valueX * normalX + valueY * normalY));
  }
  return bestResponse;
}

/** Largest fraction of on-edge samples along any interior line parallel to a side. */
export function measureInteriorSeamStrength(corners, channels, gradientThreshold) {
  const { width: imageWidth, height: imageHeight } = channels;
  const minimumInFrameSamples = Math.floor(SAMPLES_PER_SEAM_LINE / 2);
  let strongestSeam = 0.0;
  for (const firstSide of [0, 1]) {
    // lines run from side `firstSide` toward its opposite side, parallel to the other pair
    const sideStart = corners[firstSide];
    const sideEnd = corners[(firstSide + 1) % 4];
    const oppositeStart = corners[(firstSide + 3) % 4];
    const oppositeEnd = corners[(firstSide + 2) % 4];
    for (const acrossFraction of SEAM_LINE_FRACTIONS) {
      const lineStartX = sideStart[0] + acrossFraction * (oppositeStart[0] - sideStart[0]);
      const lineStartY = sideStart[1] + acrossFraction * (oppositeStart[1] - sideStart[1]);
      const lineEndX = sideEnd[0] + acrossFraction * (oppositeEnd[0] - sideEnd[0]);
      const lineEndY = sideEnd[1] + acrossFraction * (oppositeEnd[1] - sideEnd[1]);
      const lineX = lineEndX - lineStartX;
      const lineY = lineEndY - lineStartY;
      const lineLength = vectorLength(lineX, lineY);
      if (lineLength < MINIMUM_LINE_LENGTH_PIXELS) continue;
      const inFramePoints = [];
      for (const alongFraction of ALONG_FRACTIONS) {
        const pointX = lineStartX + alongFraction * lineX;
        const pointY = lineStartY + alongFraction * lineY;
        if (pointX >= 0 && pointX < imageWidth && pointY >= 0 && pointY < imageHeight) inFramePoints.push([pointX, pointY]);
      }
      if (inFramePoints.length < minimumInFrameSamples) continue;
      const normalX = -lineY / lineLength;
      const normalY = lineX / lineLength;
      let aboveCount = 0;
      for (const [pointX, pointY] of inFramePoints) {
        if (gradientAlongNormal(channels, pointX, pointY, normalX, normalY) > gradientThreshold) aboveCount += 1;
      }
      strongestSeam = Math.max(strongestSeam, aboveCount / inFramePoints.length);
    }
  }
  return strongestSeam;
}
