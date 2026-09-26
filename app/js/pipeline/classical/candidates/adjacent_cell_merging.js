/**
 * Offer unions of neighbouring edge-bounded cells as extra candidate regions (Python
 * `adjacent_cell_merging.py`).
 *
 * A check cut in two by an edge crossing it (hand-shadow boundary, fold, printed color band)
 * yields two non-check-shaped cells whose union is check-shaped. Adjacency without per-pair
 * loops: a max filter and a min filter of the label image over a window slightly wider than
 * the edge band give, at each pixel, the largest and smallest neighbouring labels; where they
 * differ the pixel sits between those two cells, and counting (min, max) pairs measures the
 * shared boundary length. Each qualifying pair's union (separating band closed) is emitted.
 */

import { contourAreaOfFlatContour } from "../numeric/opencv_geometry_formulas.js";
import { closePackedToBytes, kernelRowHalfWidths, orBitmapInto } from "../numeric/binary_morphology.js";
import { findExternalContours, withMats } from "../numeric/mat_helpers.js";

const NO_LABEL_SENTINEL = 2 ** 30;
const STAT_LEFT = 0;
const STAT_TOP = 1;
const STAT_WIDTH = 2;
const STAT_HEIGHT = 3;
const STAT_AREA = 4;
const STAT_COLUMNS = 5;

/**
 * Pairs of eligible labels whose cells touch across an edge band of `windowSize`.
 *
 * Args:
 *   labels: Int32Array label image (connectivity-4 components), `width` x `height`.
 *   maximumLabel: `labels.max()`.
 *   eligibleLabels: labels allowed to pair (piece-size gated).
 *   windowSize: square max/min filter size.
 *   minimumSharedPixels: pixels of shared band a pair needs.
 * Returns:
 *   `[[smallerLabel, largerLabel], ...]` sorted by numpy's pair code (ascending).
 */
export function findAdjacentCellPairs(cv, labels, width, height, maximumLabel, eligibleLabels, windowSize, minimumSharedPixels) {
  const isEligible = new Uint8Array(maximumLabel + 1);
  for (const label of eligibleLabels) isEligible[label] = 1;
  const pixelCount = width * height;
  return withMats((track) => {
    const eligibleImage = track(new cv.Mat(height, width, cv.CV_32F));
    const smallestSource = track(new cv.Mat(height, width, cv.CV_32F));
    const eligibleValues = eligibleImage.data32F;
    const smallestSourceValues = smallestSource.data32F;
    for (let index = 0; index < pixelCount; index += 1) {
      const label = labels[index];
      const eligibleLabel = isEligible[label] ? label : 0;
      eligibleValues[index] = eligibleLabel;
      smallestSourceValues[index] = eligibleLabel > 0 ? eligibleLabel : NO_LABEL_SENTINEL;
    }
    const kernel = track(cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(windowSize, windowSize)));
    const largestNeighbour = track(new cv.Mat());
    const smallestNeighbour = track(new cv.Mat());
    cv.dilate(eligibleImage, largestNeighbour, kernel);
    cv.erode(smallestSource, smallestNeighbour, kernel);
    const largestValues = largestNeighbour.data32F;
    const smallestValues = smallestNeighbour.data32F;
    const labelModulus = maximumLabel + 1;
    const pairCodes = new Float64Array(pixelCount);
    let pairCodeCount = 0;
    for (let index = 0; index < pixelCount; index += 1) {
      const largest = Math.trunc(largestValues[index]);
      const smallest = Math.trunc(smallestValues[index]);
      if (largest > 0 && smallest < NO_LABEL_SENTINEL && smallest !== largest) {
        pairCodes[pairCodeCount] = smallest * labelModulus + largest;
        pairCodeCount += 1;
      }
    }
    // np.unique(..., return_counts=True): sort the codes, then count each run
    const sortedCodes = pairCodes.subarray(0, pairCodeCount).sort();
    const cellPairs = [];
    for (let runStart = 0; runStart < pairCodeCount;) {
      let runEnd = runStart + 1;
      while (runEnd < pairCodeCount && sortedCodes[runEnd] === sortedCodes[runStart]) runEnd += 1;
      if (runEnd - runStart >= minimumSharedPixels) {
        const pairCode = sortedCodes[runStart];
        cellPairs.push([Math.floor(pairCode / labelModulus), pairCode % labelModulus]);
      }
      runStart = runEnd;
    }
    return cellPairs;
  });
}

/** Index of the largest-area contour (first on ties, like Python's `max`). */
export function largestContourIndex(flatContours) {
  let bestIndex = 0;
  let bestArea = -Infinity;
  for (let index = 0; index < flatContours.length; index += 1) {
    const area = contourAreaOfFlatContour(flatContours[index]);
    if (area > bestArea) {
      bestArea = area;
      bestIndex = index;
    }
  }
  return bestIndex;
}

/** Shift a flat contour by (offsetX, offsetY) in place and return it. */
export function offsetFlatContour(flatContour, offsetX, offsetY) {
  for (let index = 0; index < flatContour.length; index += 2) {
    flatContour[index] += offsetX;
    flatContour[index + 1] += offsetY;
  }
  return flatContour;
}

/**
 * Bit-packed mask of each listed label over its own bounding box, from one label-image pass.
 *
 * Returns a Map label -> { bits: Uint32Array, wordsPerRow, rows }.
 */
function packLabelBitmaps(labels, width, stats, wantedLabels) {
  const statOf = (label, column) => stats[label * STAT_COLUMNS + column];
  const bitmaps = new Map();
  let maximumLabel = 0;
  for (const label of wantedLabels) {
    const wordsPerRow = Math.ceil(statOf(label, STAT_WIDTH) / 32);
    bitmaps.set(label, { bits: new Uint32Array(wordsPerRow * statOf(label, STAT_HEIGHT)), wordsPerRow, rows: statOf(label, STAT_HEIGHT) });
    maximumLabel = Math.max(maximumLabel, label);
  }
  const bitmapOfLabel = new Array(maximumLabel + 1).fill(null);
  for (const [label, bitmap] of bitmaps) bitmapOfLabel[label] = bitmap;
  for (const label of wantedLabels) {
    const { bits, wordsPerRow } = bitmaps.get(label);
    const left = statOf(label, STAT_LEFT);
    const top = statOf(label, STAT_TOP);
    const labelWidth = statOf(label, STAT_WIDTH);
    for (let row = 0; row < statOf(label, STAT_HEIGHT); row += 1) {
      const labelRowStart = (top + row) * width + left;
      const bitsRowStart = row * wordsPerRow;
      for (let column = 0; column < labelWidth; column += 1) {
        if (labels[labelRowStart + column] === label) bits[bitsRowStart + (column >>> 5)] |= 1 << (column & 31);
      }
    }
  }
  return bitmaps;
}

/**
 * One candidate region per pair: the two cells plus the edge band between them.
 *
 * The union is assembled from per-label bitmaps (each cell scanned once, however many pairs
 * it joins) into a zero-padded canvas (== copyMakeBorder), closed with the exact bit-packed
 * rect closing, and its largest external contour returned.
 * Returns `[{ contour: Int32Array, regionArea, sourceName: "<source>_pair" }, ...]`.
 */
export function mergedPairRegions(cv, labels, width, stats, cellPairs, gapClosingSize, maximumArea, sourceName) {
  const statOf = (label, column) => stats[label * STAT_COLUMNS + column];
  const keptPairs = cellPairs.filter(([firstLabel, secondLabel]) => statOf(firstLabel, STAT_AREA) + statOf(secondLabel, STAT_AREA) <= maximumArea);
  const bitmaps = packLabelBitmaps(labels, width, stats, new Set(keptPairs.flat()));
  const regions = [];
  withMats((track) => {
    const closingKernel = track(cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(gapClosingSize, gapClosingSize)));
    const { halfWidths, anchorRow } = kernelRowHalfWidths(closingKernel);
    for (const [firstLabel, secondLabel] of keptPairs) {
      const combinedArea = statOf(firstLabel, STAT_AREA) + statOf(secondLabel, STAT_AREA);
      const left = Math.min(statOf(firstLabel, STAT_LEFT), statOf(secondLabel, STAT_LEFT));
      const top = Math.min(statOf(firstLabel, STAT_TOP), statOf(secondLabel, STAT_TOP));
      const right = Math.max(
        statOf(firstLabel, STAT_LEFT) + statOf(firstLabel, STAT_WIDTH), statOf(secondLabel, STAT_LEFT) + statOf(secondLabel, STAT_WIDTH),
      );
      const bottom = Math.max(
        statOf(firstLabel, STAT_TOP) + statOf(firstLabel, STAT_HEIGHT), statOf(secondLabel, STAT_TOP) + statOf(secondLabel, STAT_HEIGHT),
      );
      const paddedWidth = right - left + 2 * gapClosingSize;
      const paddedHeight = bottom - top + 2 * gapClosingSize;
      const paddedWordsPerRow = Math.ceil(paddedWidth / 32);
      const unionBits = new Uint32Array(paddedWordsPerRow * paddedHeight + 1); // +1: carry slack for the last word
      for (const label of [firstLabel, secondLabel]) {
        const { bits, wordsPerRow, rows } = bitmaps.get(label);
        orBitmapInto(
          unionBits, paddedWordsPerRow, bits, wordsPerRow, rows,
          statOf(label, STAT_LEFT) - left + gapClosingSize, statOf(label, STAT_TOP) - top + gapClosingSize,
        );
      }
      const closedMask = new cv.Mat(paddedHeight, paddedWidth, cv.CV_8U);
      try {
        closedMask.data.set(closePackedToBytes(unionBits.subarray(0, paddedWordsPerRow * paddedHeight), paddedWidth, paddedHeight, halfWidths, anchorRow));
        const flatContours = findExternalContours(cv, closedMask, cv.CHAIN_APPROX_NONE);
        if (flatContours.length === 0) continue;
        const largestContour = flatContours[largestContourIndex(flatContours)];
        offsetFlatContour(largestContour, left - gapClosingSize, top - gapClosingSize);
        regions.push({ contour: largestContour, regionArea: combinedArea, sourceName: `${sourceName}_pair` });
      } finally {
        closedMask.delete();
      }
    }
  });
  return regions;
}

export { STAT_AREA, STAT_COLUMNS, STAT_HEIGHT, STAT_LEFT, STAT_TOP, STAT_WIDTH };
