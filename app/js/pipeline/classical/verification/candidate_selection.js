/**
 * Pick the final checks from the pooled, verified candidates (Python `candidate_selection.py`).
 *
 * Mask families usually find the same check, so candidates arrive in near-duplicate clusters.
 * Greedy suppression in rank order (score + 0.25 * rectangularity + w * weakest side strength):
 * - duplicate: IoU with a kept quad above `duplicateIouThreshold`;
 * - containment: most of the candidate lies inside a kept quad (text box, inner border);
 * - coverage: most of the candidate is already covered by the union of kept quads.
 * Then a scene size prior: kept quads smaller than `relativeAreaFloor` x the median kept area
 * are fragments. A verified candidate is `{ corners, score, rectangularity, sideSupports,
 * sideStrengths, sourceName, weakestSideStrengthRankWeight }`.
 */

import { createInt32PointMat, withMats } from "../numeric/mat_helpers.js";
import { medianOfFloat64, roundHalfEven } from "../numeric/numpy_compatibility.js";
import {
  convexPolygonIntersectionArea, convexQuadrilateralIou, quadrilateralArea,
} from "../geometry/quadrilateral_geometry.js";

const RECTANGULARITY_RANK_WEIGHT = 0.25;

/** Sort key: evidence score, then rectangularity and the weakest side's border strength. */
export function candidateRankValue(candidate) {
  const weakestSideStrength = Math.min(...candidate.sideStrengths);
  return (
    candidate.score
    + RECTANGULARITY_RANK_WEIGHT * Math.min(candidate.rectangularity, 1.0)
    + candidate.weakestSideStrengthRankWeight * weakestSideStrength
  );
}

/** Whether a kept candidate already explains this one (duplicate or containment). */
function isSuppressedByKept(cv, candidate, keptCandidates, config) {
  const candidateArea = quadrilateralArea(candidate.corners);
  for (const keptCandidate of keptCandidates) {
    if (convexQuadrilateralIou(cv, candidate.corners, keptCandidate.corners) > config.duplicateIouThreshold) return true;
    const sharedArea = convexPolygonIntersectionArea(cv, candidate.corners, keptCandidate.corners);
    const smallerArea = Math.min(candidateArea, quadrilateralArea(keptCandidate.corners));
    if (smallerArea > 0 && sharedArea / smallerArea > config.maximumContainedFraction) return true;
  }
  return false;
}

/** Integer polygon points `np.rint(corners).astype(int32)`. */
function roundedPolygon(corners) {
  return corners.map(([x, y]) => [roundHalfEven(x), roundHalfEven(y)]);
}

/**
 * Share of a quad's rasterized pixels already set in the kept-union mask.
 *
 * Rasterizes onto `scratchMask` (all zero on entry and on exit) instead of allocating a full
 * image per candidate; counts only inside the polygon's bounding box, which holds every pixel.
 */
function fractionCoveredByMask(cv, corners, keptMask, scratchMask) {
  const polygon = roundedPolygon(corners);
  return withMats((track) => {
    const pointMat = track(createInt32PointMat(cv, polygon));
    cv.fillConvexPoly(scratchMask, pointMat, new cv.Scalar(1));
    const { cols: width, rows: height } = scratchMask;
    const left = Math.max(0, Math.min(...polygon.map(([x]) => x)) - 1);
    const right = Math.min(width - 1, Math.max(...polygon.map(([x]) => x)) + 1);
    const top = Math.max(0, Math.min(...polygon.map(([, y]) => y)) - 1);
    const bottom = Math.min(height - 1, Math.max(...polygon.map(([, y]) => y)) + 1);
    const scratchBytes = scratchMask.data;
    const keptBytes = keptMask.data;
    let candidatePixelCount = 0;
    let coveredPixelCount = 0;
    for (let row = top; row <= bottom; row += 1) {
      for (let column = left; column <= right; column += 1) {
        const index = row * width + column;
        if (!scratchBytes[index]) continue;
        candidatePixelCount += 1;
        if (keptBytes[index]) coveredPixelCount += 1;
      }
    }
    cv.fillConvexPoly(scratchMask, pointMat, new cv.Scalar(0));
    if (candidatePixelCount === 0) return 1.0;
    return coveredPixelCount / candidatePixelCount;
  });
}

/** Remove quads much smaller than the scene's median kept quad (needs 2+ candidates). */
export function dropFragmentsByRelativeArea(candidates, relativeAreaFloor) {
  if (candidates.length < 2) return candidates;
  const medianArea = medianOfFloat64(candidates.map((candidate) => quadrilateralArea(candidate.corners)));
  return candidates.filter((candidate) => quadrilateralArea(candidate.corners) >= relativeAreaFloor * medianArea);
}

/** Greedy suppression in rank order (stable, like Python's `sorted`); working image size. */
export function selectNonOverlappingCandidates(cv, candidates, config, imageWidth, imageHeight) {
  const rankedCandidates = candidates
    .map((candidate) => ({ candidate, rankValue: candidateRankValue(candidate) }))
    .sort((first, second) => second.rankValue - first.rankValue)
    .map(({ candidate }) => candidate);
  const keptCandidates = [];
  withMats((track) => {
    const keptUnionMask = track(cv.Mat.zeros(imageHeight, imageWidth, cv.CV_8U));
    const scratchMask = track(cv.Mat.zeros(imageHeight, imageWidth, cv.CV_8U));
    for (const candidate of rankedCandidates) {
      if (isSuppressedByKept(cv, candidate, keptCandidates, config)) continue;
      if (keptCandidates.length > 0 && fractionCoveredByMask(cv, candidate.corners, keptUnionMask, scratchMask) > config.maximumCoveredFraction) continue;
      keptCandidates.push(candidate);
      const pointMat = track(createInt32PointMat(cv, roundedPolygon(candidate.corners)));
      cv.fillConvexPoly(keptUnionMask, pointMat, new cv.Scalar(1));
    }
  });
  return dropFragmentsByRelativeArea(keptCandidates, config.relativeAreaFloor);
}
