/**
 * The check's paper colour: per-channel median over a grid spanning the quad's central region.
 *
 * Port of `estimate_paper_colour` in experiments/detection/refinement/quadrilateral_refinement.py.
 * Ink is a minority of the paper, so the median lands on the paper. Grid points inside another
 * detection's quad are skipped, unless that leaves fewer than MINIMUM_PAPER_COLOUR_SAMPLES.
 */

import { sampleImageBilinear } from "./image_sampling.js";
import { linspace, median } from "./numpy_compatible_math.js";
import { pointInsideConvexQuad } from "./overlap_masking.js";

const PAPER_COLOUR_GRID_SIZE = 24;
const MINIMUM_PAPER_COLOUR_SAMPLES = 40;

/** Bilinear point of the quad at (along the top side, along the left side), both in [0, 1]. */
function bilinearQuadPoint(corners, alongTop, alongLeft) {
  const point = [0, 0];
  for (let axis = 0; axis < 2; axis += 1) {
    const top = corners[0][axis] * (1 - alongTop) + corners[1][axis] * alongTop;
    const bottom = corners[3][axis] * (1 - alongTop) + corners[2][axis] * alongTop;
    point[axis] = top * (1 - alongLeft) + bottom * alongLeft;
  }
  return point;
}

/** Returns the paper colour as an array of per-channel medians (float64, image channel order). */
export function estimatePaperColour(cv, image, corners, insetFraction, otherQuads) {
  const gridValues = linspace(insetFraction, 1 - insetFraction, PAPER_COLOUR_GRID_SIZE);
  const numberOfPoints = PAPER_COLOUR_GRID_SIZE * PAPER_COLOUR_GRID_SIZE;
  const mapX = new Float32Array(numberOfPoints);
  const mapY = new Float32Array(numberOfPoints);
  const covered = new Uint8Array(numberOfPoints);
  for (let row = 0; row < PAPER_COLOUR_GRID_SIZE; row += 1) {
    for (let column = 0; column < PAPER_COLOUR_GRID_SIZE; column += 1) {
      const [x, y] = bilinearQuadPoint(corners, gridValues[column], gridValues[row]);
      const index = row * PAPER_COLOUR_GRID_SIZE + column;
      mapX[index] = x;
      mapY[index] = y;
      covered[index] = otherQuads.some((quad) => pointInsideConvexQuad(x, y, quad)) ? 1 : 0;
    }
  }
  const numberOfChannels = image.channels();
  const sampled = sampleImageBilinear(cv, image, mapX, mapY, PAPER_COLOUR_GRID_SIZE, PAPER_COLOUR_GRID_SIZE);
  const uncoveredCount = numberOfPoints - covered.reduce((sum, value) => sum + value, 0);
  const useOnlyUncovered = uncoveredCount >= MINIMUM_PAPER_COLOUR_SAMPLES;
  const paperColour = [];
  for (let channel = 0; channel < numberOfChannels; channel += 1) {
    const values = [];
    for (let index = 0; index < numberOfPoints; index += 1) {
      if (!useOnlyUncovered || !covered[index]) values.push(sampled[index * numberOfChannels + channel]);
    }
    paperColour.push(median(values));
  }
  return paperColour;
}
