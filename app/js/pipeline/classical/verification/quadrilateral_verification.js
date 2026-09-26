/**
 * Geometry gates and a border-evidence score for a candidate quad (Python
 * `quadrilateral_verification.py`).
 *
 * One question per side: is there a real border here? Samples along the middle of each side
 * count as "on a border" when ANY signal fires, each the natural cue on some background:
 * - gradient along the side normal, maximized over +-2 px (rugs, wood, drop shadows);
 * - Lab distance between a point just inside and just outside (pastel on same-lightness fabric);
 * - texture: local std just outside minus just inside (white check on white crochet);
 * - thin seam: print residue right on the side (touching checks meet along a thin line).
 * Out-of-frame samples count as supported (the check runs out of frame).
 *     score = 0.6 * mean side support + 0.4 * second-weakest side support
 * which forgives one weak side but rejects a strip of background between two checks.
 * Dtype mixing follows the Python: float32 signals promoted to float64 where numpy promotes.
 */

import { oddKernelSize } from "../classical_detector_config.js";
import { sampleBilinearReplicate } from "../numeric/bilinear_sampling.js";
import { linspace, pairwiseSum } from "../numeric/numpy_compatibility.js";
import { sideIsOnImageBorder } from "../geometry/edge_line_snapping.js";
import {
  isConvexQuadrilateral, quadrilateralArea, quadrilateralAspectRatio, quadrilateralInteriorAnglesDegrees, vectorLength,
} from "../geometry/quadrilateral_geometry.js";

const SIDE_END_EXCLUSION_FRACTION = 0.08;
const GRADIENT_SEARCH_OFFSETS_PIXELS = [-2.0, -1.0, 0.0, 1.0, 2.0];
const COLOR_SAMPLE_OFFSET_PIXELS = 4.0;
const MEAN_SUPPORT_WEIGHT = 0.6;
const SECOND_WEAKEST_SUPPORT_WEIGHT = 0.4;
const MAXIMUM_STRENGTH_RATIO = 3.0;
const MINIMUM_DIRECTION_LENGTH = 1e-9;
const fround = Math.fround;

/** True when any side lies along the image border (the check runs out of frame). */
export function quadrilateralTouchesImageBorder(corners, imageWidth, imageHeight) {
  return [0, 1, 2, 3].some((side) => sideIsOnImageBorder(corners[side], corners[(side + 1) % 4], imageWidth, imageHeight));
}

/** Convex, check-sized, check-shaped (aspect in range, no needle corners). */
export function passesGeometryGates(corners, imageWidth, imageHeight, config) {
  if (!isConvexQuadrilateral(corners)) return false;
  const areaFraction = quadrilateralArea(corners) / (imageWidth * imageHeight);
  if (!(config.minimumAreaFraction <= areaFraction && areaFraction <= config.maximumAreaFraction)) return false;
  const aspectRatio = quadrilateralAspectRatio(corners);
  const [minimumAspect, maximumAspect] = quadrilateralTouchesImageBorder(corners, imageWidth, imageHeight)
    ? config.borderTruncatedAspectRange
    : [config.minimumAspectRatio, config.maximumAspectRatio];
  if (!(minimumAspect <= aspectRatio && aspectRatio <= maximumAspect)) return false;
  return Math.min(...quadrilateralInteriorAnglesDegrees(corners)) >= config.minimumInteriorAngleDegrees;
}

/** Sample one float32 signal map at a point (FMA-exact cv2.remap emulation). */
function sampleAt(signalMap, pointX, pointY) {
  return sampleBilinearReplicate(signalMap.values, signalMap.width, signalMap.height, pointX, pointY);
}

/**
 * `[support, strength]` of one side.
 *
 * support: fraction of samples where at least one border signal fires.
 * strength: mean of the strongest signal's ratio to its threshold, clipped at 3; out-of-frame
 * samples count as exactly 1 so an image-border side never outranks a real border.
 */
function measureSideBorderSupport(sideStart, sideEnd, inwardNormal, channels, config) {
  const { width, height } = channels;
  const [normalX, normalY] = inwardNormal;
  const sampleCount = config.boundarySamplesPerSide;
  const fractions = linspace(SIDE_END_EXCLUSION_FRACTION, 1.0 - SIDE_END_EXCLUSION_FRACTION, sampleCount);
  const textureOffset = oddKernelSize(config.textureWindowFraction, channels.longSidePixels);
  const colorThreshold = fround(config.edgeSupportColorThreshold);
  const textureThreshold = fround(config.edgeSupportTextureThreshold);
  const signalRatios = new Float64Array(sampleCount);
  let supportedCount = 0;
  for (let sample = 0; sample < sampleCount; sample += 1) {
    const pointX = sideStart[0] + fractions[sample] * (sideEnd[0] - sideStart[0]);
    const pointY = sideStart[1] + fractions[sample] * (sideEnd[1] - sideStart[1]);
    const outOfFrame = pointX < 0 || pointX > width - 1 || pointY < 0 || pointY > height - 1;
    let normalGradient = 0;
    let seamResidue = 0;
    for (const offset of GRADIENT_SEARCH_OFFSETS_PIXELS) {
      const shiftedX = pointX + offset * normalX;
      const shiftedY = pointY + offset * normalY;
      seamResidue = Math.max(seamResidue, sampleAt(channels.printResidue, shiftedX, shiftedY));
      const gradientX = sampleAt(channels.gradientX, shiftedX, shiftedY);
      const gradientY = sampleAt(channels.gradientY, shiftedX, shiftedY);
      normalGradient = Math.max(normalGradient, Math.abs(gradientX * normalX + gradientY * normalY));
    }
    const innerX = pointX + COLOR_SAMPLE_OFFSET_PIXELS * normalX;
    const innerY = pointY + COLOR_SAMPLE_OFFSET_PIXELS * normalY;
    const outerX = pointX - COLOR_SAMPLE_OFFSET_PIXELS * normalX;
    const outerY = pointY - COLOR_SAMPLE_OFFSET_PIXELS * normalY;
    // Python: sqrt(sum(float32 difference ** 2 over L, a, b)), all float32
    let squaredColorDistance = 0;
    for (const channelMap of [channels.lightness, channels.labA, channels.labB]) {
      const difference = fround(sampleAt(channelMap, innerX, innerY) - sampleAt(channelMap, outerX, outerY));
      squaredColorDistance = fround(squaredColorDistance + fround(difference * difference));
    }
    const colorDistance = fround(Math.sqrt(squaredColorDistance));
    const textureContrast = fround(
      sampleAt(channels.textureStd, pointX - textureOffset * normalX, pointY - textureOffset * normalY)
      - sampleAt(channels.textureStd, pointX + textureOffset * normalX, pointY + textureOffset * normalY),
    );
    let signalRatio = Math.max(
      normalGradient / config.edgeSupportGradientThreshold,
      fround(colorDistance / colorThreshold),
      fround(textureContrast / textureThreshold),
      seamResidue / config.edgeSupportSeamResidueThreshold,
    );
    signalRatio = outOfFrame ? 1.0 : Math.min(signalRatio, MAXIMUM_STRENGTH_RATIO);
    signalRatios[sample] = signalRatio;
    if (signalRatio > 1.0 || outOfFrame) supportedCount += 1;
  }
  return [supportedCount / sampleCount, pairwiseSum(signalRatios) / sampleCount];
}

/**
 * Border support of all four sides and the combined score.
 *
 * Returns `{ sideSupports: number[4], sideStrengths: number[4], score }`.
 */
export function measureQuadrilateralEvidence(corners, channels, config) {
  const centroidX = (corners[0][0] + corners[1][0] + corners[2][0] + corners[3][0]) / 4;
  const centroidY = (corners[0][1] + corners[1][1] + corners[2][1] + corners[3][1]) / 4;
  const sideSupports = [];
  const sideStrengths = [];
  for (let side = 0; side < 4; side += 1) {
    const sideStart = corners[side];
    const sideEnd = corners[(side + 1) % 4];
    const sideX = sideEnd[0] - sideStart[0];
    const sideY = sideEnd[1] - sideStart[1];
    const sideLength = Math.max(vectorLength(sideX, sideY), MINIMUM_DIRECTION_LENGTH);
    const normalX = -sideY / sideLength;
    const normalY = sideX / sideLength;
    const midpointX = (sideStart[0] + sideEnd[0]) / 2.0;
    const midpointY = (sideStart[1] + sideEnd[1]) / 2.0;
    const pointsInward = (centroidX - midpointX) * normalX + (centroidY - midpointY) * normalY > 0;
    const inwardNormal = pointsInward ? [normalX, normalY] : [-normalX, -normalY];
    const [support, strength] = measureSideBorderSupport(sideStart, sideEnd, inwardNormal, channels, config);
    sideSupports.push(support);
    sideStrengths.push(strength);
  }
  const secondWeakestSupport = [...sideSupports].sort((first, second) => first - second)[1];
  const meanSupport = pairwiseSum(sideSupports) / 4;
  return {
    sideSupports,
    sideStrengths,
    score: MEAN_SUPPORT_WEIGHT * meanSupport + SECOND_WEAKEST_SUPPORT_WEIGHT * secondWeakestSupport,
  };
}
