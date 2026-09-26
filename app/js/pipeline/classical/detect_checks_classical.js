/**
 * Top-level classical check detector: full-res BGR photo in, check quadrilaterals out. A faithful
 * port of the Python `experiments/detection/classical/detect_checks_classical.py`.
 *
 * Pipeline (each stage in its own module):
 * 1. `preprocessing/working_image_channels`: resize to ~1600 px, text-suppressed lightness,
 *    chroma, paper score, texture, gradients.
 * 2. `candidates/candidate_regions`: smooth-paper, textured, paper-score and edge-bounded-cell
 *    masks and their check-sized regions (plus adjacent-cell unions).
 * 3. `geometry/quadrilateral_fitting`: approxPolyN quad + per-side refit per region;
 *    `candidates/line_segment_extraction` + `line_quadrilateral_hypotheses`: rectangles from
 *    parallel Hough segment pairs. Every quad is snapped (`geometry/edge_line_snapping`).
 * 4. `verification/*`: shape gates, border evidence, interior seam, interior appearance.
 * 5. `verification/candidate_selection`: greedy duplicate / containment / coverage suppression.
 * 6. `geometry/full_resolution_edge_refinement`: sub-pixel side snapping on the original image.
 *
 * Contract: every function receives `cv`; plain synchronous code, no DOM or worker globals,
 * so the same files run in Node (parity) and in a browser Web Worker. Every Mat created is
 * deleted; the caller's `imageBgr` is only read.
 */

import { CLASSICAL_DETECTOR_CONFIG_DEFAULTS } from "./classical_detector_config.js";
import { extractAllCandidateRegions } from "./candidates/candidate_regions.js";
import { buildLineQuadrilateralHypotheses } from "./candidates/line_quadrilateral_hypotheses.js";
import { extractLineSegments } from "./candidates/line_segment_extraction.js";
import { snapQuadrilateralSidesToEdges } from "./geometry/edge_line_snapping.js";
import { refineCornersAtFullResolution } from "./geometry/full_resolution_edge_refinement.js";
import { fitQuadrilateralToContour } from "./geometry/quadrilateral_fitting.js";
import { orderCornersClockwise } from "./geometry/quadrilateral_geometry.js";
import { roundHalfEven } from "./numeric/numpy_compatibility.js";
import { buildWorkingImageChannels } from "./preprocessing/working_image_channels.js";
import { selectNonOverlappingCandidates } from "./verification/candidate_selection.js";
import { interiorPassesCheckGate } from "./verification/interior_appearance.js";
import { measureInteriorSeamStrength } from "./verification/interior_seam_detection.js";
import { measureQuadrilateralEvidence, passesGeometryGates } from "./verification/quadrilateral_verification.js";

export const DEFAULT_CLASSICAL_DETECTOR_CONFIG = CLASSICAL_DETECTOR_CONFIG_DEFAULTS;

const LINE_SOURCE_NAME = "lines";
const MINIMUM_SNAP_SEARCH_RADIUS_PIXELS = 2;
const PIXEL_CENTER_OFFSET = 0.5;
const MAXIMUM_REPORTED_SCORE = 1.0;

/** Every fitted quad from regions and from line hypotheses, in the Python order. */
export function collectFittedQuadrilaterals(cv, channels, config) {
  const aspectRange = [
    Math.min(config.minimumAspectRatio, config.borderTruncatedAspectRange[0]),
    Math.max(config.maximumAspectRatio, config.borderTruncatedAspectRange[1]),
  ];
  const fittedQuads = [];
  for (const region of extractAllCandidateRegions(cv, channels, config)) {
    const fittedQuad = fitQuadrilateralToContour(cv, region.contour, region.sourceName, config.minimumRegionRectangularity, aspectRange);
    if (fittedQuad !== null && fittedQuad.rectangularity >= config.minimumRegionRectangularity) fittedQuads.push(fittedQuad);
  }
  if (config.useLineHypotheses) {
    const lineSegments = extractLineSegments(cv, channels, config);
    for (const corners of buildLineQuadrilateralHypotheses(lineSegments, channels.longSidePixels, config)) {
      fittedQuads.push({ corners, rectangularity: config.lineHypothesisRectangularity, sourceName: LINE_SOURCE_NAME });
    }
  }
  return fittedQuads;
}

/** Snap, then run the gates cheapest first; a verified candidate or null when any gate fails. */
export function verifyFittedQuadrilateral(cv, fittedQuad, channels, config) {
  const { width, height } = channels;
  if (!passesGeometryGates(fittedQuad.corners, width, height, config)) return null; // snapping rarely rescues a bad shape
  const snapSearchRadius = Math.max(
    MINIMUM_SNAP_SEARCH_RADIUS_PIXELS, Math.trunc(roundHalfEven(config.workingSnapSearchFraction * channels.longSidePixels)),
  );
  const snappedCorners = orderCornersClockwise(snapQuadrilateralSidesToEdges(
    cv, channels.lightness, fittedQuad.corners, snapSearchRadius, config.boundarySamplesPerSide, config.workingSnapMinimumStep,
  ));
  if (!passesGeometryGates(snappedCorners, width, height, config)) return null;
  const evidence = measureQuadrilateralEvidence(snappedCorners, channels, config);
  if (evidence.score < config.minimumVerificationScore) return null;
  if ([...evidence.sideSupports].sort((first, second) => first - second)[1] < config.minimumEdgeSupport) return null;
  if (fittedQuad.sourceName === LINE_SOURCE_NAME && evidence.score < config.minimumLineHypothesisScore) return null;
  if (measureInteriorSeamStrength(snappedCorners, channels, config.seamGradientThreshold) > config.maximumInteriorSeamStrength) return null;
  if (!interiorPassesCheckGate(cv, snappedCorners, channels, config)) return null;
  return {
    corners: snappedCorners,
    score: evidence.score,
    rectangularity: fittedQuad.rectangularity,
    sideSupports: evidence.sideSupports,
    sideStrengths: evidence.sideStrengths,
    sourceName: fittedQuad.sourceName,
    weakestSideStrengthRankWeight: config.weakestSideStrengthRankWeight,
  };
}

/** Every fitted quad that passes the shape gates and the evidence thresholds. */
export function collectVerifiedCandidates(cv, channels, config) {
  const verifiedCandidates = [];
  for (const fittedQuad of collectFittedQuadrilaterals(cv, channels, config)) {
    const verifiedCandidate = verifyFittedQuadrilateral(cv, fittedQuad, channels, config);
    if (verifiedCandidate !== null) verifiedCandidates.push(verifiedCandidate);
  }
  return verifiedCandidates;
}

/**
 * Find every check in a full-resolution BGR photo.
 *
 * Args:
 *   cv: the OpenCV.js module.
 *   imageBgr: cv.Mat CV_8UC3, full resolution, BGR. Owned by the caller; not modified or deleted.
 *   config: `DEFAULT_CLASSICAL_DETECTOR_CONFIG` or a copy with overrides.
 * Returns:
 *   `[{ corners: [[x, y] x4], score, sourceName, sideSupports }]`; corners are full-resolution,
 *   clockwise in image coordinates, pixel-center convention, smallest x + y corner first.
 */
export function detectChecksClassical(cv, imageBgr, config = DEFAULT_CLASSICAL_DETECTOR_CONFIG) {
  if (imageBgr.type() !== cv.CV_8UC3) throw new Error(`detectChecksClassical expects CV_8UC3 BGR, got Mat type ${imageBgr.type()}`);
  const channels = buildWorkingImageChannels(cv, imageBgr, config);
  const selectedCandidates = selectNonOverlappingCandidates(
    cv, collectVerifiedCandidates(cv, channels, config), config, channels.width, channels.height,
  );
  const scale = channels.scaleToFullResolution;
  return selectedCandidates.map((candidate) => {
    // pixel-center convention: working pixel centers map to (x + 0.5) * scale - 0.5
    let fullResolutionCorners = candidate.corners.map(([x, y]) => [
      (x + PIXEL_CENTER_OFFSET) * scale - PIXEL_CENTER_OFFSET, (y + PIXEL_CENTER_OFFSET) * scale - PIXEL_CENTER_OFFSET,
    ]);
    if (config.refineCornersAtFullResolution) {
      fullResolutionCorners = refineCornersAtFullResolution(cv, imageBgr, fullResolutionCorners, config);
    }
    return {
      corners: orderCornersClockwise(fullResolutionCorners),
      score: Math.min(candidate.score, MAXIMUM_REPORTED_SCORE),
      sourceName: candidate.sourceName,
      sideSupports: [...candidate.sideSupports],
    };
  });
}
