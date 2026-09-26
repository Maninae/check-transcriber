/**
 * Stage 2 of the pipeline (spec section 5), run in the worker right after a photo
 * arrives: find every check and hand the count step numbered quads.
 *
 * 1. classical/  - the no-model detector (full-resolution quads, arbitrary start corner)
 * 2. refinement/ - sub-pixel corners from the paper's own edges, each check refined
 *                  knowing the others (overlap masking)
 * 3. reading order + the confident cue for the count step header
 *
 * This is the same "classical + refine" pipeline scored in experiments/detection/REPORT.md;
 * orientation runs later, after the operator confirms the count (check_pipeline_stage.js).
 */

import { detectChecksClassical } from "./classical/detect_checks_classical.js";
import { refineDetectedChecks } from "./refinement/quadrilateral_refinement.js";
import { isDetectionConfident } from "./detection_confidence.js";
import { sortIntoReadingOrder } from "./reading_order.js";

/**
 * `imageBgr` is the full-resolution photo (CV_8UC3, caller-owned). `onProgress(message)`
 * reports each stage. Returns `[{ corners, confident }]` in reading order.
 */
export function detectChecksInPhoto(cv, imageBgr, onProgress) {
  onProgress("Finding checks");
  const classicalDetections = detectChecksClassical(cv, imageBgr);
  onProgress("Tightening the outlines");
  const refinedCornerSets = refineDetectedChecks(cv, imageBgr, classicalDetections.map(({ corners }) => corners));
  const refinedDetections = classicalDetections.map((detection, index) => ({
    corners: refinedCornerSets[index],
    sideSupports: detection.sideSupports,
  }));
  const allCornerSets = refinedDetections.map(({ corners }) => corners);
  const ordered = sortIntoReadingOrder(refinedDetections, ({ corners }) => corners);
  return ordered.map((detection) => ({
    corners: detection.corners,
    confident: isDetectionConfident(detection, allCornerSets),
  }));
}
