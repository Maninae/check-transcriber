/**
 * Refine ONE side of the current quad: score it, find its line, grow a robust curve.
 *
 * Port of the per-side helpers in experiments/detection/refinement/quadrilateral_refinement.py:
 * `scoreSide` (score grid, other checks masked), `refineOneSide` (pass-1 line search, then
 * curve growing from sub-pixel peaks near the current curve), `refineSideNarrowFirst`
 * (a +-narrow band first, widened only without support), and the tracking-pass helpers
 * `trackSidePoints` and `fitCornerLocalLine`.
 *
 * `passSettings` = { inwardBand, outwardBand, searchLine, degree, maximumSamples } (one pass).
 */

import { scoreSideEdgeProfiles } from "./edge_profile_sampling.js";
import { clip, hypot, median, radiansToDegrees } from "./numpy_compatible_math.js";
import { maskCellsInsideOtherQuads, quadCentroid } from "./overlap_masking.js";
import { fitRobustSideCurve } from "./robust_side_curve_fitting.js";
import { SideCurve } from "./side_curve.js";
import { trackEdgePath } from "./side_edge_tracking.js";
import { extractEdgePointsNearCentre, searchOutermostStrongLine } from "./side_line_search.js";

const MINIMUM_SIDE_LENGTH_FOR_ANGLE = 1.0;
const MINIMUM_SIDE_LENGTH_DIVISOR = 1e-9;
const CORNER_LOCAL_MINIMUM_SUPPORT_FRACTION = 0.5;

/** Score grid for one side of `corners`, cells over other checks masked: { profiles, numberOfSamples }. */
export function scoreSide(cv, image, corners, sideIndex, passSettings, paperColour, config, dilatedOtherQuads) {
  const sideStart = corners[sideIndex];
  const sideEnd = corners[(sideIndex + 1) % 4];
  const sideLength = hypot(sideEnd[0] - sideStart[0], sideEnd[1] - sideStart[1]);
  const numberOfSamples = Math.trunc(clip(sideLength / config.sampleSpacingPixels, config.minimumSamplesPerSide, passSettings.maximumSamples));
  const cornerMargin = Math.max(config.cornerMarginFraction * sideLength, config.minimumCornerMarginPixels);
  const profiles = scoreSideEdgeProfiles(
    cv, image, sideStart, sideEnd, quadCentroid(corners),
    passSettings.inwardBand, passSettings.outwardBand, numberOfSamples, cornerMargin, paperColour, config,
  );
  maskCellsInsideOtherQuads(profiles, dilatedOtherQuads);
  return { profiles, numberOfSamples };
}

/** A zero-offset (unmoved, unsupported) curve on the profiles' own frame. */
function unmovedCurve(profiles) {
  return new SideCurve({
    sideStart: profiles.sideStart, unitTangent: profiles.unitTangent, unitNormal: profiles.unitNormal,
    sideLength: profiles.sideLength, coefficients: new Float64Array(2),
  });
}

/** Line search (when the pass searches) then curve growing; the unmoved curve on failure. */
export function refineOneSide(profiles, numberOfSamples, passSettings, config, randomGenerator) {
  const { sideLength } = profiles;
  const currentCurve = unmovedCurve(profiles);
  let seedCurve = currentCurve;
  if (passSettings.searchLine) {
    const maximumAngle = clip(
      radiansToDegrees(Math.atan((2 * passSettings.inwardBand) / Math.max(sideLength, MINIMUM_SIDE_LENGTH_FOR_ANGLE))),
      config.minimumLineAngleDegrees, config.maximumLineAngleDegrees,
    );
    const line = searchOutermostStrongLine(
      profiles, maximumAngle, config.lineAngleStepDegrees, config.outermostLineRatio,
      config.minimumEdgeScore, config.lineScoreClip, config.keepInputLineRatio,
    );
    if (line === null) return currentCurve;
    const [offsetAtMiddle, slope] = line;
    seedCurve = currentCurve.withFit({ coefficients: Float64Array.of(offsetAtMiddle, (slope * sideLength) / 2) });
  }
  const requiredInliers = Math.max(config.minimumInlierCount, Math.ceil(config.minimumInlierFraction * numberOfSamples));
  let fittedCurve = null;
  for (let iteration = 0; iteration < config.curveGrowIterations; iteration += 1) {
    const guideCurve = fittedCurve ?? seedCurve;
    const centreOffsets = guideCurve.offsetsAt(profiles.positionsPixels);
    const { positions, offsets, strengths } = extractEdgePointsNearCentre(profiles, centreOffsets, config.pointTolerancePixels, config.minimumEdgeScore);
    const candidate = fitRobustSideCurve(
      seedCurve, positions, offsets, strengths, passSettings.degree, config.ransacInlierDistancePixels, config.ransacIterations, randomGenerator,
    );
    if (candidate === null || candidate.inlierCount < requiredInliers) break;
    fittedCurve = candidate;
  }
  return fittedCurve ?? currentCurve;
}

/**
 * Search a small band around the input side first; widen only if it finds no well-supported edge.
 *
 * Deep inside the paper the narrow band cannot fake support: each sample's background colour
 * is then paper-coloured, so those samples carry no evidence. Returns
 * { scored: { profiles, numberOfSamples }, curve, acceptedInNarrowBand }.
 */
export function refineSideNarrowFirst(cv, image, corners, sideIndex, passSettings, paperColour, config, dilatedOtherQuads, randomGenerator, narrowBand) {
  if (narrowBand !== null) {
    const narrowSettings = { ...passSettings, inwardBand: narrowBand, outwardBand: narrowBand };
    const narrowScored = scoreSide(cv, image, corners, sideIndex, narrowSettings, paperColour, config, dilatedOtherQuads);
    const curve = refineOneSide(narrowScored.profiles, narrowScored.numberOfSamples, narrowSettings, config, randomGenerator);
    if (curve.inlierCount >= config.narrowFirstMinimumSupport * narrowScored.numberOfSamples) {
      return { scored: narrowScored, curve, acceptedInNarrowBand: true };
    }
  }
  const scored = scoreSide(cv, image, corners, sideIndex, passSettings, paperColour, config, dilatedOtherQuads);
  return { scored, curve: refineOneSide(scored.profiles, scored.numberOfSamples, passSettings, config, randomGenerator), acceptedInNarrowBand: false };
}

/** Viterbi-tracked edge points of one side, weak ones dropped: { positions, offsets, strengths }. */
export function trackSidePoints(profiles, config) {
  const tracked = trackEdgePath(profiles, config.lineScoreClip, config.trackMaximumStepOffsets, config.trackStepCost, config.trackCentreCostPerPixel);
  const strong = Array.from(tracked.strengths, (strength) => strength >= config.minimumEdgeScore);
  const keep = (values) => Float64Array.from(values.filter((_, index) => strong[index]));
  return { positions: keep(tracked.positions), offsets: keep(tracked.offsets), strengths: keep(tracked.strengths) };
}

/**
 * Straight fit to the tracked points near one corner, IF they bend away from the side curve.
 *
 * Returns null (keep the side curve) with too few points near the corner, when their median
 * deviation from `sideCurve` is under `cornerBendThresholdPixels`, or without local support.
 */
export function fitCornerLocalLine(sideCurve, points, nearStart, fraction, config, randomGenerator) {
  const { positions, offsets, strengths } = points;
  if (positions.length === 0) return null;
  const lengthDivisor = Math.max(sideCurve.sideLength, MINIMUM_SIDE_LENGTH_DIVISOR);
  const keptIndices = [];
  positions.forEach((position, index) => {
    const relative = position / lengthDivisor;
    if (nearStart ? relative <= fraction : relative >= 1 - fraction) keptIndices.push(index);
  });
  if (keptIndices.length < config.minimumInlierCount) return null;
  const deviations = keptIndices.map((index) => offsets[index] - sideCurve.offsetAt(positions[index]));
  if (Math.abs(median(deviations)) < config.cornerBendThresholdPixels) return null;
  const pick = (values) => Float64Array.from(keptIndices, (index) => values[index]);
  const local = fitRobustSideCurve(
    sideCurve, pick(positions), pick(offsets), pick(strengths), 1, config.ransacInlierDistancePixels, config.ransacIterations, randomGenerator,
  );
  const requiredSupport = Math.max(config.minimumInlierCount, Math.trunc(CORNER_LOCAL_MINIMUM_SUPPORT_FRACTION * keptIndices.length));
  if (local === null || local.inlierCount < requiredSupport) return null;
  return local;
}

/** Evidence summary of one fitted side (what the acceptance gate looks at). */
export function describeSideCurve(curve, numberOfSamples, shortSide) {
  const startOffset = curve.offsetAt(0.0);
  const endOffset = curve.offsetAt(curve.sideLength);
  const { sideStart, unitTangent, unitNormal, sideLength } = curve;
  return {
    supportFraction: curve.inlierCount / Math.max(numberOfSamples, 1),
    medianStrength: curve.medianInlierStrength,
    residualRms: curve.inlierResidualRms,
    moveFractionOfShortSide: Math.max(Math.abs(startOffset), Math.abs(endOffset)) / Math.max(shortSide, MINIMUM_SIDE_LENGTH_DIVISOR),
    startPoint: [sideStart[0] + unitNormal[0] * startOffset, sideStart[1] + unitNormal[1] * startOffset],
    endPoint: [
      sideStart[0] + unitTangent[0] * sideLength + unitNormal[0] * endOffset,
      sideStart[1] + unitTangent[1] * sideLength + unitNormal[1] * endOffset,
    ],
  };
}
