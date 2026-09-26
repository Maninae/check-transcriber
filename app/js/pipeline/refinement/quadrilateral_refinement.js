/**
 * Refine an approximate check quadrilateral to sub-pixel corners using the paper's edges.
 *
 * Faithful port of experiments/detection/refinement/quadrilateral_refinement.py. Per check,
 * sampling the full-resolution uint8 image directly with bilinear remap:
 *
 * 1. Paper colour = median of a grid inside the input quad.
 * 2. Each pass, for each side: score a (position x normal offset) grid, pick the outermost
 *    strong line (pass 1, narrow band first), grow a robust curve from sub-pixel peaks near
 *    it. Pass 1 is straight and mostly inward; pass 2 is narrow and quadratic, and (tracking
 *    mode) takes a corner from local straight fits where the Viterbi edge path bends.
 * 3. A side without enough support (off the frame, hidden, no contrast) keeps its line.
 * 4. Corners = intersections of adjacent curves, so the input corner ORDER is preserved.
 * 5. Guard rails vs the INPUT quad revert implausible corners or the whole quad.
 *
 * Runs unchanged in Node and in a Web Worker: `cv` is always an argument, nothing is async,
 * no DOM. Corners are [[x, y] x 4]; the caller owns `imageBgr` (CV_8UC3, or CV_8UC1).
 */

import { computeShortSideLength, applyGuardRails } from "./quadrilateral_geometry.js";
import { clip } from "./numpy_compatible_math.js";
import { dilateConvexQuad, selectNearbyQuads } from "./overlap_masking.js";
import { estimatePaperColour } from "./paper_colour_estimation.js";
import { DEFAULT_CORNER_REFINEMENT_CONFIG } from "./refinement_config.js";
import { intersectSideCurves } from "./robust_side_curve_fitting.js";
import { createRandomGeneratorForConfig } from "./seeded_random_generator.js";
import { describeSideCurve, fitCornerLocalLine, refineSideNarrowFirst, trackSidePoints } from "./side_refinement.js";

export { DEFAULT_CORNER_REFINEMENT_CONFIG };

/** A band fraction of the short side, clamped to the configured pixel range. */
function bandPixels(fraction, shortSide, config) {
  return clip(fraction * shortSide, config.minimumBandPixels, config.maximumBandPixels);
}

/** One settings object per pass: { inwardBand, outwardBand, searchLine, degree, maximumSamples }. */
function buildPassSettings(shortSide, config) {
  const numberOfPasses = Math.min(
    config.inwardBandFractionPerPass.length, config.outwardBandFractionPerPass.length,
    config.curveDegreePerPass.length, config.maximumSamplesPerSidePerPass.length,
  );
  return Array.from({ length: numberOfPasses }, (_, passIndex) => ({
    inwardBand: bandPixels(config.inwardBandFractionPerPass[passIndex], shortSide, config),
    outwardBand: bandPixels(config.outwardBandFractionPerPass[passIndex], shortSide, config),
    searchLine: passIndex === 0,
    degree: config.curveDegreePerPass[passIndex],
    maximumSamples: config.maximumSamplesPerSidePerPass[passIndex],
  }));
}

/** Validate and copy a [[x, y] x 4] quad into plain float arrays; fail loud on anything else. */
function copyQuadCorners(corners, label) {
  if (!Array.isArray(corners) && !ArrayBuffer.isView(corners)) throw new Error(`${label}: expected 4 [x, y] corners`);
  if (corners.length !== 4) throw new Error(`${label}: expected 4 corners, got ${corners.length}`);
  return Array.from(corners, (corner) => {
    const x = Number(corner[0]);
    const y = Number(corner[1]);
    if (!Number.isFinite(x) || !Number.isFinite(y)) throw new Error(`${label}: non-finite corner ${corner}`);
    return [x, y];
  });
}

/** For each corner i, the local-fit (or side) curves meeting there after the tracking pass. */
function selectCornerCurvesWithLocalFits(scoredSides, sideCurves, config, randomGenerator) {
  const trackedPoints = scoredSides.map(({ profiles }) => trackSidePoints(profiles, config));
  const cornerCurvePairs = [];
  const localCornerFits = [];
  for (let cornerIndex = 0; cornerIndex < 4; cornerIndex += 1) {
    const incomingSide = (cornerIndex + 3) % 4;
    const outgoingSide = cornerIndex;
    const incomingLocal = fitCornerLocalLine(sideCurves[incomingSide], trackedPoints[incomingSide], false, config.cornerLocalFraction, config, randomGenerator);
    const outgoingLocal = fitCornerLocalLine(sideCurves[outgoingSide], trackedPoints[outgoingSide], true, config.cornerLocalFraction, config, randomGenerator);
    localCornerFits.push(incomingLocal !== null || outgoingLocal !== null);
    cornerCurvePairs.push([incomingLocal ?? sideCurves[incomingSide], outgoingLocal ?? sideCurves[outgoingSide]]);
  }
  return { cornerCurvePairs, localCornerFits };
}

/**
 * Move each corner of an approximate quad onto the paper's physical corner.
 *
 * Args:
 *   cv: the OpenCV.js module.
 *   imageBgr: cv.Mat CV_8UC3 (or CV_8UC1), full resolution; the caller owns and deletes it.
 *   corners: [[x, y] x 4] in either winding; the ORDER is preserved.
 *   config: DEFAULT_CORNER_REFINEMENT_CONFIG or a variant with the same keys.
 *   otherQuads: the other detections' corner sets; score cells inside them carry no evidence.
 * Returns:
 *   { corners: refined [[x, y] x 4], diagnostics: paper colour, per-pass side statistics,
 *     narrow-band acceptance, local corner fits, and which guard rails fired }.
 */
export function refineCheckQuadrilateral(cv, imageBgr, corners, config = DEFAULT_CORNER_REFINEMENT_CONFIG, otherQuads = []) {
  const inputCorners = copyQuadCorners(corners, "refineCheckQuadrilateral corners");
  const randomGenerator = createRandomGeneratorForConfig(config);
  const shortSide = computeShortSideLength(inputCorners);
  const passSettingsList = buildPassSettings(shortSide, config);
  const largestBand = Math.max(...passSettingsList.map((settings) => Math.max(settings.inwardBand, settings.outwardBand)));
  const otherQuadCorners = otherQuads.map((quad, index) => copyQuadCorners(quad, `otherQuads[${index}]`));
  const nearbyQuads = selectNearbyQuads(inputCorners, otherQuadCorners, 2 * largestBand);
  const dilatedOtherQuads = config.maskOtherChecks ? nearbyQuads.map((quad) => dilateConvexQuad(quad, config.otherCheckDilationPixels)) : [];
  const paperColour = estimatePaperColour(cv, imageBgr, inputCorners, config.paperSampleInsetFraction, config.maskOtherChecks ? nearbyQuads : []);
  const diagnostics = { paperColour, passes: [] };
  const narrowBand = config.narrowFirstBandFraction > 0 ? bandPixels(config.narrowFirstBandFraction, shortSide, config) : null;

  let currentCorners = inputCorners.map((corner) => [...corner]);
  passSettingsList.forEach((passSettings, passIndex) => {
    const isLastPass = passIndex === passSettingsList.length - 1;
    const trackingPass = config.finalPassMode === "track" && isLastPass && passIndex > 0;
    const scoredSides = [];
    const sideCurves = [];
    const narrowAccepted = [];
    for (let sideIndex = 0; sideIndex < 4; sideIndex += 1) {
      const { scored, curve, acceptedInNarrowBand } = refineSideNarrowFirst(
        cv, imageBgr, currentCorners, sideIndex, passSettings, paperColour, config, dilatedOtherQuads, randomGenerator,
        passIndex === 0 ? narrowBand : null,
      );
      scoredSides.push(scored);
      sideCurves.push(curve);
      narrowAccepted.push(acceptedInNarrowBand);
    }
    let cornerCurvePairs = [0, 1, 2, 3].map((cornerIndex) => [sideCurves[(cornerIndex + 3) % 4], sideCurves[cornerIndex]]);
    let localCornerFits = [false, false, false, false];
    if (trackingPass) ({ cornerCurvePairs, localCornerFits } = selectCornerCurvesWithLocalFits(scoredSides, sideCurves, config, randomGenerator));
    const newCorners = currentCorners.map((corner, cornerIndex) => {
      const intersection = intersectSideCurves(...cornerCurvePairs[cornerIndex]);
      return intersection ?? [...corner];
    });
    diagnostics.passes.push({
      ...passSettings,
      localCornerFits,
      sidesAcceptedInNarrowBand: narrowAccepted,
      sidesWithoutSupport: sideCurves.map((curve) => curve.inlierCount === 0),
      sideStatistics: sideCurves.map((curve, sideIndex) => describeSideCurve(curve, scoredSides[sideIndex].numberOfSamples, shortSide)),
    });
    currentCorners = newCorners;
  });
  const refinedCorners = applyGuardRails(inputCorners, currentCorners, passSettingsList[0].inwardBand, config, diagnostics);
  return { corners: refinedCorners, diagnostics };
}

/**
 * Refine every detection's corners, each knowing the others' INPUT quads (overlap masking).
 *
 * Returns the refined corner sets in the same order (corner order within each preserved).
 */
export function refineDetectedChecks(cv, imageBgr, cornerSets, config = DEFAULT_CORNER_REFINEMENT_CONFIG) {
  return cornerSets.map((corners, checkIndex) => {
    const otherQuads = cornerSets.filter((_, otherIndex) => otherIndex !== checkIndex);
    return refineCheckQuadrilateral(cv, imageBgr, corners, config, otherQuads).corners;
  });
}
