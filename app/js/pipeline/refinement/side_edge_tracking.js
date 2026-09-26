/**
 * Track a side's edge sample by sample with dynamic programming (Viterbi), for curved sides.
 *
 * Port of experiments/detection/refinement/side_edge_tracking.py. A curl or lifted corner
 * bends the paper edge only near one end, which one polynomial per side cannot follow. The
 * path may move at most `maximumStepOffsets` offsets between neighbouring samples; it
 * maximizes the total clipped score minus a per-step cost and a pull toward offset 0 (the
 * previous pass's side, so it does not wander onto a parallel printed rule). Path offsets
 * are refined to sub-pixel with a parabola. The coordinator fits corner-local lines to it.
 */

import { argmax } from "./numpy_compatible_math.js";

const MAXIMUM_SUB_PIXEL_SHIFT = 0.5;

/**
 * Best smooth path through the score grid.
 *
 * Returns { positions, offsets, strengths } (Float64Arrays: px along the side, sub-pixel
 * offsets px, clipped score on the path) for samples whose path cell is a finite score.
 */
export function trackEdgePath(profiles, scoreClip, maximumStepOffsets, stepCost, centreCostPerPixel) {
  const { scores, numberOfSamples, numberOfOffsets, normalOffsets } = profiles;
  const clipped = new Float64Array(scores.length);
  const pathScores = new Float64Array(scores.length);
  for (let cell = 0; cell < scores.length; cell += 1) {
    const score = scores[cell];
    clipped[cell] = Number.isFinite(score) ? Math.min(Math.max(score, -scoreClip), scoreClip) : 0.0;
    pathScores[cell] = clipped[cell] - centreCostPerPixel * Math.abs(normalOffsets[cell % numberOfOffsets]);
  }
  let accumulated = pathScores.slice(0, numberOfOffsets);
  const backPointers = new Int32Array(numberOfSamples * numberOfOffsets);
  for (let sample = 1; sample < numberOfSamples; sample += 1) {
    const nextAccumulated = new Float64Array(numberOfOffsets);
    for (let offset = 0; offset < numberOfOffsets; offset += 1) {
      // Candidates in step order -k..k; the first maximum wins (numpy argmax).
      let bestCandidate = -Infinity;
      let bestStep = -maximumStepOffsets;
      let firstCandidate = true;
      for (let step = -maximumStepOffsets; step <= maximumStepOffsets; step += 1) {
        const previous = offset + step;
        const previousScore = previous >= 0 && previous < numberOfOffsets ? accumulated[previous] : -Infinity;
        const candidate = previousScore - stepCost * Math.abs(step);
        if (firstCandidate || candidate > bestCandidate) {
          bestCandidate = candidate;
          bestStep = step;
          firstCandidate = false;
        }
      }
      nextAccumulated[offset] = pathScores[sample * numberOfOffsets + offset] + bestCandidate;
      backPointers[sample * numberOfOffsets + offset] = offset + bestStep;
    }
    accumulated = nextAccumulated;
  }
  const path = new Int32Array(numberOfSamples);
  path[numberOfSamples - 1] = argmax(accumulated);
  for (let sample = numberOfSamples - 1; sample > 0; sample -= 1) {
    path[sample - 1] = backPointers[sample * numberOfOffsets + path[sample]];
  }

  const positions = [];
  const offsets = [];
  const strengths = [];
  const lastIndex = numberOfOffsets - 1;
  for (let sample = 0; sample < numberOfSamples; sample += 1) {
    const pathOffset = path[sample];
    const row = sample * numberOfOffsets;
    const centreScore = scores[row + pathOffset];
    if (!Number.isFinite(centreScore)) continue;
    const left = scores[row + Math.max(pathOffset - 1, 0)];
    const right = scores[row + Math.min(pathOffset + 1, lastIndex)];
    const denominator = left - 2 * centreScore + right;
    const usableParabola = Number.isFinite(left) && Number.isFinite(right) && denominator < 0 && pathOffset > 0 && pathOffset < lastIndex;
    let shift = usableParabola ? (0.5 * (left - right)) / denominator : 0.0;
    if (Number.isNaN(shift)) shift = 0.0;
    shift = Math.min(Math.max(shift, -MAXIMUM_SUB_PIXEL_SHIFT), MAXIMUM_SUB_PIXEL_SHIFT);
    positions.push(profiles.positionsPixels[sample]);
    offsets.push(normalOffsets[pathOffset] + shift);
    strengths.push(clipped[row + pathOffset]);
  }
  return { positions: Float64Array.from(positions), offsets: Float64Array.from(offsets), strengths: Float64Array.from(strengths) };
}
