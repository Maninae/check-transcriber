/**
 * Every threshold of the classical (no model) check detector, mirroring the Python
 * `ClassicalDetectorConfig` defaults one for one (snake_case -> camelCase, tuples -> arrays).
 *
 * Sizes that depend on image scale are fractions of the working image's long side (or
 * area), so one config serves 2 MP and 12 MP photos; `oddKernelSize` turns a fraction into
 * an odd pixel kernel at runtime. Values marked "tuned" came from the Python sweeps on val;
 * change them only together with the Python config.
 */

import { roundHalfEven } from "./numeric/numpy_compatibility.js";

export const CLASSICAL_DETECTOR_CONFIG_DEFAULTS = {
  workingLongSidePixels: 1600,

  textSuppressionKernelFraction: 0.0045,
  textureWindowFraction: 0.009,
  chromaWeightInPaperScore: 1.0,

  smoothTextureStdThresholds: [3.0, 5.0],
  edgeGradientThresholds: [12.0, 24.0],
  edgeDilationPixels: 1,
  cannyThresholdPairs: [[20.0, 50.0], [40.0, 100.0]],
  cannyDilationPixels: 2,
  useChromaEdges: true,
  chromaEdgeGain: 6.0, // tuned (sweep_v2)
  useHoleFilledEdges: true,
  texturedStdThresholds: [3.0],
  mergeAdjacentCells: true,
  minimumCellPieceFraction: 0.0015,
  minimumSharedBoundaryFraction: 0.03,
  paperScoreOtsuOffsets: [0.0],
  maskOpeningFraction: 0.006,

  useLineHypotheses: true,
  lineCannyLow: 15.0,
  lineCannyHigh: 40.0,
  lineTextureSuppressionStd: 6.0,
  lineMinimumLengthFraction: 0.04,
  lineMaximumGapFraction: 0.006,
  lineMergeGapFraction: 0.03,
  lineMaximumSegments: 120,
  lineMaximumPairs: 1500,
  lineMaximumHypotheses: 400,
  lineParallelToleranceDegrees: 7.0,
  linePerpendicularToleranceDegrees: 10.0,
  lineMinimumSeparationFraction: 0.03,
  lineHypothesisRectangularity: 0.9,

  minimumAreaFraction: 0.004,
  maximumAreaFraction: 0.8,
  minimumAspectRatio: 1.5,
  maximumAspectRatio: 3.4,
  borderTruncatedAspectRange: [1.0, 8.0],
  minimumInteriorAngleDegrees: 55.0,
  minimumRegionRectangularity: 0.8,
  approxPolyEpsilonFraction: 0.02, // present in the Python config, unused by its pipeline too

  workingSnapSearchFraction: 0.006,
  workingSnapMinimumStep: 2.0,

  boundarySamplesPerSide: 40,
  edgeSupportGradientThreshold: 10.0,
  edgeSupportColorThreshold: 8.0,
  edgeSupportTextureThreshold: 4.0,
  edgeSupportSeamResidueThreshold: 30.0,
  minimumEdgeSupport: 0.45,
  minimumVerificationScore: 0.75, // tuned (sweep_v2)

  printResidueThreshold: 20.0,
  minimumInteriorPrintFraction: 0.05,
  maximumInteriorTexture: 4.5,
  minimumInteriorPaperScore: 130.0,
  maximumInteriorChroma: 35.0,

  seamGradientThreshold: 16.0,
  maximumInteriorSeamStrength: 0.45, // tuned (sweep_v2)
  minimumLineHypothesisScore: 0.9, // tuned (sweep_v1)

  weakestSideStrengthRankWeight: 0.0,

  duplicateIouThreshold: 0.5,
  maximumContainedFraction: 0.75,
  relativeAreaFloor: 0.3,
  maximumCoveredFraction: 0.75, // tuned (sweep_v1)

  refineCornersAtFullResolution: true,
  refinementSearchFraction: 0.006,
  refinementSamplesPerSide: 60,
  refinementMinimumStepStrength: 3.0,
};

/** Recursively freeze a plain config object (arrays included). */
function deepFreeze(value) {
  if (value && typeof value === "object") {
    for (const child of Object.values(value)) deepFreeze(child);
    Object.freeze(value);
  }
  return value;
}

deepFreeze(CLASSICAL_DETECTOR_CONFIG_DEFAULTS);

/** Convert a fraction of the long side into an odd kernel size in pixels (Python `odd_kernel_size`). */
export function oddKernelSize(fractionOfLongSide, longSidePixels, minimumSize = 3) {
  const kernelSize = Math.max(minimumSize, Math.trunc(roundHalfEven(fractionOfLongSide * longSidePixels)));
  return kernelSize % 2 === 1 ? kernelSize : kernelSize + 1;
}
