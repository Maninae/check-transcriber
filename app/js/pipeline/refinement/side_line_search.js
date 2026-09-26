/**
 * Find a side's paper edge from its score grid: line-level search, then sub-pixel points.
 *
 * Port of experiments/detection/refinement/side_line_search.py. Per-sample peaks are
 * unreliable (a faint paper edge loses to printed text at most samples), but the paper edge
 * runs the WHOLE side. So clipped scores are integrated along candidate lines (a small Radon
 * transform over angles near the side) and the OUTERMOST line reaching a fraction of the
 * best wins: printed borders and rules always lie inside the paper edge.
 *
 * - Clip to [-c, +c], never positive-only: background texture then integrates to ~0 instead
 *   of forming a fake outer line the outermost rule would pick.
 * - Line offsets use np.round (half to even) of slope * position.
 */

import { argmax, argminAbsolute, degreesToRadians, floatArange, median, roundHalfToEven } from "./numpy_compatible_math.js";

const ANGLE_RANGE_EPSILON = 1e-9;
const MAXIMUM_SUB_PIXEL_SHIFT = 0.5;

/** Integrated clipped score per (slope, offset): Float64Array (slopes x J). */
function integrateAlongLines(clippedScores, numberOfSamples, numberOfOffsets, slopes, relativePositions) {
  const lineScores = new Float64Array(slopes.length * numberOfOffsets);
  for (let slopeIndex = 0; slopeIndex < slopes.length; slopeIndex += 1) {
    const row = slopeIndex * numberOfOffsets;
    for (let sample = 0; sample < numberOfSamples; sample += 1) {
      const shift = roundHalfToEven(slopes[slopeIndex] * relativePositions[sample]);
      const firstOffset = Math.max(0, -shift);
      const lastOffset = Math.min(numberOfOffsets, numberOfOffsets - shift);
      const sampleRow = sample * numberOfOffsets + shift;
      for (let offset = firstOffset; offset < lastOffset; offset += 1) lineScores[row + offset] += clippedScores[sampleRow + offset];
    }
  }
  return lineScores;
}

/**
 * Returns [offset at the side's midpoint, slope d offset / d position], or null.
 *
 * Confidence gate: when `keepInputLineRatio` > 0 and the INPUT side (angle 0, offset 0)
 * integrates to at least that fraction of the chosen line, [0, 0] is returned.
 */
export function searchOutermostStrongLine(profiles, maximumAngleDegrees, angleStepDegrees, outermostLineRatio, minimumMeanScore, absoluteScoreClip, keepInputLineRatio = 0.0) {
  const { scores, numberOfSamples, numberOfOffsets } = profiles;
  const rowMaxima = [];
  for (let sample = 0; sample < numberOfSamples; sample += 1) {
    let rowHasValid = false;
    let rowMaximum = 0;
    for (let offset = 0; offset < numberOfOffsets; offset += 1) {
      const score = scores[sample * numberOfOffsets + offset];
      if (!Number.isFinite(score)) continue;
      rowHasValid = true;
      rowMaximum = Math.max(rowMaximum, score);
    }
    if (rowHasValid) rowMaxima.push(rowMaximum);
  }
  if (rowMaxima.length < 2) return null;
  const clipValue = Math.min(absoluteScoreClip, median(rowMaxima));
  if (clipValue <= 0) return null;
  const clippedScores = new Float64Array(scores.length);
  for (let cell = 0; cell < scores.length; cell += 1) {
    const score = scores[cell];
    clippedScores[cell] = Number.isFinite(score) ? Math.min(Math.max(score, -clipValue), clipValue) : 0.0;
  }

  const relativePositions = profiles.positionsPixels.map((position) => position - profiles.sideLength / 2);
  const angles = floatArange(-maximumAngleDegrees, maximumAngleDegrees + ANGLE_RANGE_EPSILON, angleStepDegrees);
  const slopes = angles.map((angle) => Math.tan(degreesToRadians(angle)));
  const lineScores = integrateAlongLines(clippedScores, numberOfSamples, numberOfOffsets, slopes, relativePositions);

  const bestSlopeIndexPerOffset = new Int32Array(numberOfOffsets);
  const bestScorePerOffset = new Float64Array(numberOfOffsets);
  for (let offset = 0; offset < numberOfOffsets; offset += 1) {
    let bestSlopeIndex = 0;
    for (let slopeIndex = 1; slopeIndex < slopes.length; slopeIndex += 1) {
      if (lineScores[slopeIndex * numberOfOffsets + offset] > lineScores[bestSlopeIndex * numberOfOffsets + offset]) bestSlopeIndex = slopeIndex;
    }
    bestSlopeIndexPerOffset[offset] = bestSlopeIndex;
    bestScorePerOffset[offset] = lineScores[bestSlopeIndex * numberOfOffsets + offset];
  }
  const globalBest = bestScorePerOffset[argmax(bestScorePerOffset)];
  if (globalBest < minimumMeanScore * rowMaxima.length) return null;
  let chosenOffsetIndex = -1;
  for (let offset = 0; offset < numberOfOffsets; offset += 1) {
    const score = bestScorePerOffset[offset];
    const left = offset > 0 ? bestScorePerOffset[offset - 1] : -Infinity;
    const right = offset < numberOfOffsets - 1 ? bestScorePerOffset[offset + 1] : -Infinity;
    if (score >= left && score >= right && score >= outermostLineRatio * globalBest) chosenOffsetIndex = offset;
  }
  const chosenScore = bestScorePerOffset[chosenOffsetIndex];
  if (keepInputLineRatio > 0) {
    const inputOffsetIndex = argminAbsolute(profiles.normalOffsets);
    const inputSlopeIndex = argminAbsolute(slopes);
    const inputScore = lineScores[inputSlopeIndex * numberOfOffsets + inputOffsetIndex];
    if (inputScore > 0 && inputScore >= keepInputLineRatio * chosenScore) return [0.0, 0.0];
  }
  return [profiles.normalOffsets[chosenOffsetIndex], slopes[bestSlopeIndexPerOffset[chosenOffsetIndex]]];
}

/**
 * Per-sample sub-pixel peak within +-tolerance of `centreOffsets` (length S).
 *
 * Returns { positions, offsets, strengths } (Float64Arrays) for the samples whose windowed
 * peak is an interior maximum with finite neighbours and a score >= `minimumScore`.
 */
export function extractEdgePointsNearCentre(profiles, centreOffsets, tolerancePixels, minimumScore) {
  const { scores, numberOfSamples, numberOfOffsets, normalOffsets } = profiles;
  const positions = [];
  const offsets = [];
  const strengths = [];
  const lastIndex = numberOfOffsets - 1;
  const windowedScoreAt = (sample, offset) => (Math.abs(normalOffsets[offset] - centreOffsets[sample]) <= tolerancePixels ? scores[sample * numberOfOffsets + offset] : -Infinity);
  for (let sample = 0; sample < numberOfSamples; sample += 1) {
    let bestIndex = 0;
    let bestScore = windowedScoreAt(sample, 0);
    for (let offset = 1; offset < numberOfOffsets; offset += 1) {
      const score = windowedScoreAt(sample, offset);
      if (score > bestScore) {
        bestScore = score;
        bestIndex = offset;
      }
    }
    const left = windowedScoreAt(sample, Math.max(bestIndex - 1, 0));
    const right = windowedScoreAt(sample, Math.min(bestIndex + 1, lastIndex));
    const interiorPeak = bestIndex > 0 && bestIndex < lastIndex && Number.isFinite(left) && Number.isFinite(right);
    if (!(interiorPeak && Number.isFinite(bestScore) && bestScore >= minimumScore)) continue;
    const denominator = left - 2 * bestScore + right;
    let subPixelShift = denominator < 0 ? (0.5 * (left - right)) / denominator : 0.0;
    if (Number.isNaN(subPixelShift)) subPixelShift = 0.0;
    subPixelShift = Math.min(Math.max(subPixelShift, -MAXIMUM_SUB_PIXEL_SHIFT), MAXIMUM_SUB_PIXEL_SHIFT);
    positions.push(profiles.positionsPixels[sample]);
    offsets.push(normalOffsets[bestIndex] + subPixelShift);
    strengths.push(bestScore);
  }
  return { positions: Float64Array.from(positions), offsets: Float64Array.from(offsets), strengths: Float64Array.from(strengths) };
}
