/**
 * Tuning knobs for edge-based corner refinement (pixels or fractions), tuned on val.
 *
 * A camelCase mirror of `CornerRefinementConfig` in
 * experiments/detection/refinement/refinement_config.py; every value is the Python default
 * and the comments there explain each one. Per-pass tuples are arrays indexed by pass.
 *
 * - Bands scale with the check's short side, clamped to [minimumBandPixels, maximumBandPixels].
 * - Pass 1 searches mostly inward with a Radon line search (narrow band first); pass 2 only
 *   grows the pass-1 curve and, with finalPassMode "track", follows bends near corners.
 */

export const DEFAULT_CORNER_REFINEMENT_CONFIG = Object.freeze({
  inwardBandFractionPerPass: Object.freeze([0.14, 0.03]),
  outwardBandFractionPerPass: Object.freeze([0.04, 0.03]),
  minimumBandPixels: 6.0,
  maximumBandPixels: 140.0,
  narrowFirstBandFraction: 0.03,
  narrowFirstMinimumSupport: 0.6,
  minimumLineAngleDegrees: 3.0,
  maximumLineAngleDegrees: 15.0,
  lineAngleStepDegrees: 0.5,
  outermostLineRatio: 0.65,
  lineScoreClip: 0.6,
  keepInputLineRatio: 0.9,
  curveDegreePerPass: Object.freeze([1, 2]),
  curveGrowIterations: 3,
  finalPassMode: "track",
  trackMaximumStepOffsets: 1,
  trackStepCost: 0.05,
  trackCentreCostPerPixel: 0.01,
  cornerLocalFraction: 0.2,
  cornerBendThresholdPixels: 1.5,
  pointTolerancePixels: 3.0,

  sampleSpacingPixels: 6.0,
  minimumSamplesPerSide: 16,
  maximumSamplesPerSidePerPass: Object.freeze([64, 128]),
  cornerMarginFraction: 0.03,
  minimumCornerMarginPixels: 4.0,

  tangentialOffsetsPixels: Object.freeze([-2.0, -1.0, 0.0, 1.0, 2.0]),
  innerWindowPixels: 4,
  outerWindowPixels: 4,
  edgeScoreMode: "two_class",
  backgroundWindowPixels: 6,
  minimumPaperBackgroundContrast: 6.0,
  tentBeyondPaper: true,
  contactLineHalfWidthPixels: 3,
  contactLineContrastUnit: 20.0,
  contactLineBelowContrast: 25.0,
  paperSampleInsetFraction: 0.15,

  ransacIterations: 48,
  ransacInlierDistancePixels: 1.5,
  minimumInlierFraction: 0.25,
  minimumInlierCount: 6,
  minimumEdgeScore: 0.2,

  maximumCornerMoveBandMultiple: 1.5,
  maximumSideLengthChangeFraction: 0.35,

  maskOtherChecks: true,
  otherCheckDilationPixels: 0.0,

  randomSeed: 0,
});
