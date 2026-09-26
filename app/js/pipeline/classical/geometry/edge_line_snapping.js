/**
 * Snap each side of a quadrilateral to the strongest nearby intensity step (Python
 * `edge_line_snapping.py`).
 *
 * Used at working resolution right after fitting (mask quads sit a few pixels inside the true
 * border) and at full resolution for the final sub-pixel corners. For each side, intensity
 * profiles are sampled perpendicular to it; the strongest step near the current position wins,
 * with a Gaussian prior on the offset so it does not jump to a check's printed inner border.
 * A parabola through the peak gives the sub-pixel offset, a Huber `fitLine` the side's line,
 * and adjacent lines intersect into corners. Sides on the image border are left alone.
 * Profiles use the FMA-exact bilinear sampler (the Python samples with `cv2.remap`).
 */

import { sampleBilinearReplicate } from "../numeric/bilinear_sampling.js";
import { fitLineHuberExact } from "../numeric/huber_line_fit.js";
import { linspace } from "../numeric/numpy_compatibility.js";
import { intersectLines, vectorLength } from "./quadrilateral_geometry.js";

const SIDE_END_EXCLUSION_FRACTION = 0.1;
const MINIMUM_VALID_FRACTION = 0.35; // of a side's samples; otherwise the side keeps its old line
const MAXIMUM_CORNER_SHIFT_RADIUS_FACTOR = 1.5; // corner may move at most this x the search radius
const BORDER_MARGIN_PIXELS = 3.0;
const MINIMUM_SIDE_LENGTH_PIXELS = 4;
const FLAT_CURVATURE = 1e-6;
const MAXIMUM_SUBPIXEL_SHIFT = 0.5;
const MINIMUM_DIRECTION_LENGTH = 1e-9;
const fround = Math.fround;

/** Both endpoints near the same image border: the check leaves the frame here. */
export function sideIsOnImageBorder(sideStart, sideEnd, imageWidth, imageHeight) {
  const borders = [[0, 0.0], [0, imageWidth - 1.0], [1, 0.0], [1, imageHeight - 1.0]];
  for (const [coordinateIndex, borderValue] of borders) {
    if (
      Math.abs(sideStart[coordinateIndex] - borderValue) < BORDER_MARGIN_PIXELS
      && Math.abs(sideEnd[coordinateIndex] - borderValue) < BORDER_MARGIN_PIXELS
    ) return true;
  }
  return false;
}

/**
 * Line `[vx, vy, x0, y0]` through the strongest nearby step along one side; null if too weak.
 *
 * `intensityMap` is `{ values: Float32Array, width, height }`.
 */
export function fitSideLineFromProfiles(cv, intensityMap, sideStart, sideEnd, searchRadiusPixels, samplesPerSide, minimumStepStrength) {
  const { values, width, height } = intensityMap;
  const sideX = sideEnd[0] - sideStart[0];
  const sideY = sideEnd[1] - sideStart[1];
  const sideLength = vectorLength(sideX, sideY);
  if (sideLength < MINIMUM_SIDE_LENGTH_PIXELS) return null;
  const normalX = -sideY / sideLength;
  const normalY = sideX / sideLength;
  const fractions = linspace(SIDE_END_EXCLUSION_FRACTION, 1 - SIDE_END_EXCLUSION_FRACTION, samplesPerSide);
  const offsetCount = 2 * searchRadiusPixels + 3; // offsets -r-1 .. r+1
  const stepCount = offsetCount - 2; // centered differences at offsets -r .. r
  const priorSigma = searchRadiusPixels / 2.0;
  const prior = new Float64Array(stepCount);
  for (let step = 0; step < stepCount; step += 1) {
    const offset = step - searchRadiusPixels;
    prior[step] = Math.exp(-0.5 * ((offset / priorSigma) * (offset / priorSigma)));
  }
  const profile = new Float32Array(offsetCount);
  const stepStrength = new Float32Array(stepCount);
  const edgePoints = new Float32Array(2 * samplesPerSide);
  let validCount = 0;
  for (let sample = 0; sample < samplesPerSide; sample += 1) {
    const baseX = sideStart[0] + fractions[sample] * sideX;
    const baseY = sideStart[1] + fractions[sample] * sideY;
    for (let offsetIndex = 0; offsetIndex < offsetCount; offsetIndex += 1) {
      const offset = offsetIndex - searchRadiusPixels - 1;
      profile[offsetIndex] = sampleBilinearReplicate(values, width, height, baseX + offset * normalX, baseY + offset * normalY);
    }
    let peakIndex = 0;
    let peakScore = -Infinity;
    for (let step = 0; step < stepCount; step += 1) {
      stepStrength[step] = fround(Math.abs(fround(profile[step + 2] - profile[step])) / 2.0);
      const weightedStrength = stepStrength[step] * prior[step];
      if (weightedStrength > peakScore) {
        peakScore = weightedStrength;
        peakIndex = step;
      }
    }
    const peakStrength = stepStrength[peakIndex];
    const leftValue = stepStrength[Math.max(peakIndex - 1, 0)];
    const rightValue = stepStrength[Math.min(peakIndex + 1, stepCount - 1)];
    const curvature = fround(fround(leftValue - fround(2 * peakStrength)) + rightValue);
    let subpixelShift = 0.0;
    if (Math.abs(curvature) > fround(FLAT_CURVATURE)) subpixelShift = fround(fround(0.5 * fround(leftValue - rightValue)) / curvature);
    subpixelShift = Math.min(MAXIMUM_SUBPIXEL_SHIFT, Math.max(-MAXIMUM_SUBPIXEL_SHIFT, subpixelShift));
    if (!(peakStrength > fround(minimumStepStrength))) continue;
    const edgeOffset = (peakIndex - searchRadiusPixels) + subpixelShift;
    edgePoints[2 * validCount] = baseX + edgeOffset * normalX;
    edgePoints[2 * validCount + 1] = baseY + edgeOffset * normalY;
    validCount += 1;
  }
  if (validCount / samplesPerSide < MINIMUM_VALID_FRACTION) return null;
  return fitLineHuberExact(edgePoints, validCount);
}

/**
 * Snapped four corners in `intensityMap` coordinates; falls back side by side to the input.
 *
 * `fullImageSize` `[width, height]` and `cornerOffset` `[x, y]` let a caller pass a crop:
 * the image-border test then runs in full-image coordinates (corners + offset).
 */
export function snapQuadrilateralSidesToEdges(
  cv, intensityMap, corners, searchRadiusPixels, samplesPerSide, minimumStepStrength,
  fullImageSize = null, cornerOffset = [0, 0],
) {
  const [imageWidth, imageHeight] = fullImageSize || [intensityMap.width, intensityMap.height];
  const sideLines = [];
  for (let side = 0; side < 4; side += 1) {
    const sideStart = corners[side];
    const sideEnd = corners[(side + 1) % 4];
    const directionLength = Math.max(vectorLength(sideEnd[0] - sideStart[0], sideEnd[1] - sideStart[1]), MINIMUM_DIRECTION_LENGTH);
    const originalLine = [(sideEnd[0] - sideStart[0]) / directionLength, (sideEnd[1] - sideStart[1]) / directionLength, sideStart[0], sideStart[1]];
    const startInImage = [sideStart[0] + cornerOffset[0], sideStart[1] + cornerOffset[1]];
    const endInImage = [sideEnd[0] + cornerOffset[0], sideEnd[1] + cornerOffset[1]];
    if (sideIsOnImageBorder(startInImage, endInImage, imageWidth, imageHeight)) {
      sideLines.push(originalLine);
      continue;
    }
    const fittedLine = fitSideLineFromProfiles(cv, intensityMap, sideStart, sideEnd, searchRadiusPixels, samplesPerSide, minimumStepStrength);
    sideLines.push(fittedLine === null ? originalLine : fittedLine);
  }
  const snappedCorners = [];
  for (let corner = 0; corner < 4; corner += 1) {
    let snappedCorner = intersectLines(sideLines[(corner + 3) % 4], sideLines[corner]);
    if (
      snappedCorner === null
      || vectorLength(snappedCorner[0] - corners[corner][0], snappedCorner[1] - corners[corner][1]) > MAXIMUM_CORNER_SHIFT_RADIUS_FACTOR * searchRadiusPixels
    ) snappedCorner = [corners[corner][0], corners[corner][1]];
    snappedCorners.push(snappedCorner);
  }
  return snappedCorners;
}
