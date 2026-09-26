/**
 * Check-sized connected regions of every candidate mask (Python `candidate_regions.py`,
 * `extract_regions_from_mask` and `extract_all_candidate_regions`).
 *
 * Each mask is opened to cut thin bridges between touching blobs, labelled (connectivity 4),
 * and every component within the area gates is returned as its largest external contour in
 * working pixels. For edge-bounded cell masks, unions of adjacent cells are added first
 * (`adjacent_cell_merging.js`), matching the Python region order.
 */

import { oddKernelSize } from "../classical_detector_config.js";
import { kernelRowHalfWidths, openBinary } from "../numeric/binary_morphology.js";
import { findExternalContours, withMats } from "../numeric/mat_helpers.js";
import {
  STAT_AREA, STAT_COLUMNS, STAT_HEIGHT, STAT_LEFT, STAT_TOP, STAT_WIDTH,
  findAdjacentCellPairs, largestContourIndex, mergedPairRegions, offsetFlatContour,
} from "./adjacent_cell_merging.js";
import { forEachCandidateMask } from "./candidate_masks.js";

const REGION_SHRINK_UNDER_OPENING_FACTOR = 0.5; // regions shrink under opening

/** Region contour of one labelled component (its bounding-box crop, largest contour). */
function componentContour(cv, labels, width, componentIndex, left, top, componentWidth, componentHeight) {
  const componentMask = new cv.Mat(componentHeight, componentWidth, cv.CV_8U);
  try {
    const maskBytes = componentMask.data;
    for (let row = 0; row < componentHeight; row += 1) {
      const labelRowStart = (top + row) * width + left;
      const maskRowStart = row * componentWidth;
      for (let column = 0; column < componentWidth; column += 1) {
        maskBytes[maskRowStart + column] = labels[labelRowStart + column] === componentIndex ? 1 : 0;
      }
    }
    const flatContours = findExternalContours(cv, componentMask, cv.CHAIN_APPROX_NONE);
    return offsetFlatContour(flatContours[largestContourIndex(flatContours)], left, top);
  } finally {
    componentMask.delete();
  }
}

/**
 * Open the mask, then return every check-sized component's contour (plus cell-pair unions).
 *
 * Returns `[{ contour: Int32Array, regionArea, sourceName }, ...]`.
 */
export function extractRegionsFromMask(cv, binaryMaskMat, sourceName, channels, config) {
  const { width, height, longSidePixels, areaPixels } = channels;
  const openingSize = oddKernelSize(config.maskOpeningFraction, longSidePixels);
  const minimumArea = config.minimumAreaFraction * areaPixels * REGION_SHRINK_UNDER_OPENING_FACTOR;
  const maximumArea = config.maximumAreaFraction * areaPixels;
  const { componentCount, labels, stats } = withMats((track) => {
    const openingKernel = track(cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(openingSize, openingSize)));
    const { halfWidths, anchorRow } = kernelRowHalfWidths(openingKernel);
    // exact binary opening in JS (identical to cv.morphologyEx MORPH_OPEN, ~10x faster than WASM)
    const openedMask = track(new cv.Mat(height, width, cv.CV_8U));
    openedMask.data.set(openBinary(binaryMaskMat.data, width, height, halfWidths, anchorRow));
    const labelsMat = track(new cv.Mat());
    const statsMat = track(new cv.Mat());
    const centroidsMat = track(new cv.Mat());
    const count = cv.connectedComponentsWithStats(openedMask, labelsMat, statsMat, centroidsMat, 4, cv.CV_32S);
    return { componentCount: count, labels: Int32Array.from(labelsMat.data32S), stats: Int32Array.from(statsMat.data32S) };
  });
  const statOf = (label, column) => stats[label * STAT_COLUMNS + column];
  const regions = [];
  if (config.mergeAdjacentCells && sourceName.includes("cells") && componentCount > 1) {
    const eligibleLabels = [];
    for (let label = 1; label < componentCount; label += 1) {
      const pieceArea = statOf(label, STAT_AREA);
      if (pieceArea >= config.minimumCellPieceFraction * areaPixels && pieceArea <= maximumArea) eligibleLabels.push(label);
    }
    const separationWidth = openingSize + 2 * Math.max(config.edgeDilationPixels, config.cannyDilationPixels) + 2;
    const cellPairs = findAdjacentCellPairs(
      cv, labels, width, height, componentCount - 1, eligibleLabels, separationWidth,
      Math.trunc(config.minimumSharedBoundaryFraction * longSidePixels),
    );
    regions.push(...mergedPairRegions(cv, labels, width, stats, cellPairs, separationWidth, maximumArea, sourceName));
  }
  for (let componentIndex = 1; componentIndex < componentCount; componentIndex += 1) {
    const area = statOf(componentIndex, STAT_AREA);
    if (area < minimumArea || area > maximumArea) continue;
    const contour = componentContour(
      cv, labels, width, componentIndex,
      statOf(componentIndex, STAT_LEFT), statOf(componentIndex, STAT_TOP),
      statOf(componentIndex, STAT_WIDTH), statOf(componentIndex, STAT_HEIGHT),
    );
    regions.push({ contour, regionArea: area, sourceName });
  }
  return regions;
}

/** Regions from every mask family, pooled in the Python order (duplicates resolved later). */
export function extractAllCandidateRegions(cv, channels, config) {
  const regions = [];
  forEachCandidateMask(cv, channels, config, (sourceName, maskMat) => {
    regions.push(...extractRegionsFromMask(cv, maskMat, sourceName, channels, config));
  });
  return regions;
}
