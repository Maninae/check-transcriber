/**
 * Score every candidate edge offset along one side of an approximate quad.
 *
 * Port of experiments/detection/refinement/edge_profile_sampling.py. For each sample
 * position along the side, one `cv.remap` (bilinear, straight from the uint8 image, border
 * replicate) reads a colour profile along the outward normal; a few profiles offset along
 * the tangent are averaged to suppress texture. Each normal offset is then scored as a paper
 * boundary (see edge_feature_scoring.js): high where paper gives way to something else.
 *
 * - Offsets whose windows leave the image score -Infinity: a side running off the frame
 *   produces no evidence and the caller keeps its approximate line.
 * - Maps are computed in float64 and stored as float32, exactly like the numpy original,
 *   so OpenCV.js remap sees bit-identical maps and returns bit-identical pixels.
 */

import {
  computeBackgroundColours,
  computeBoxWindowScores,
  computePaperDistanceFeature,
  computeTwoClassFeature,
  writeContactLineScoresForSample,
} from "./edge_feature_scoring.js";
import { hypot, linspace } from "./numpy_compatible_math.js";
import { sampleImageBilinear } from "./image_sampling.js";

const MINIMUM_SIDE_LENGTH = 1e-9;
const MAXIMUM_CORNER_MARGIN_FRACTION_OF_SIDE = 0.3;

/** Edge scores for one side on a (sample position x normal offset) grid, plus its frame. */
export class SideScoreProfiles {
  /** Fields mirror the Python dataclass; `scores` is a row-major Float64Array (S x J). */
  constructor({ sideStart, unitTangent, unitNormal, sideLength, positionsPixels, normalOffsets, scores }) {
    this.sideStart = sideStart;
    this.unitTangent = unitTangent;
    this.unitNormal = unitNormal;
    this.sideLength = sideLength;
    this.positionsPixels = positionsPixels;
    this.normalOffsets = normalOffsets;
    this.scores = scores;
    this.numberOfSamples = positionsPixels.length;
    this.numberOfOffsets = normalOffsets.length;
  }

  /** Image point [x, y] of (position along the side, normal offset), numpy's evaluation order. */
  pointAt(positionPixels, offsetPixels) {
    return [
      this.sideStart[0] + positionPixels * this.unitTangent[0] + offsetPixels * this.unitNormal[0],
      this.sideStart[1] + positionPixels * this.unitTangent[1] + offsetPixels * this.unitNormal[1],
    ];
  }
}

/** { unitTangent start->end, unitNormal pointing away from the centroid, sideLength }. */
export function computeSideFrame(sideStart, sideEnd, quadCentroid) {
  const sideVectorX = sideEnd[0] - sideStart[0];
  const sideVectorY = sideEnd[1] - sideStart[1];
  const sideLength = hypot(sideVectorX, sideVectorY);
  const lengthDivisor = Math.max(sideLength, MINIMUM_SIDE_LENGTH);
  const unitTangent = [sideVectorX / lengthDivisor, sideVectorY / lengthDivisor];
  let unitNormal = [unitTangent[1], -unitTangent[0]];
  const middleToCentroidX = (sideStart[0] + sideEnd[0]) / 2 - quadCentroid[0];
  const middleToCentroidY = (sideStart[1] + sideEnd[1]) / 2 - quadCentroid[1];
  if (unitNormal[0] * middleToCentroidX + unitNormal[1] * middleToCentroidY < 0) unitNormal = [-unitNormal[0], -unitNormal[1]];
  return { unitTangent, unitNormal, sideLength };
}

/**
 * Sample tangent-averaged normal profiles: { profiles (S, Joffsets, C) float32 values, insideImage (S, Joffsets) }.
 *
 * `insideImage` is true only where every tangential copy lies inside [0, W-1] x [0, H-1].
 */
function sampleNormalProfiles(cv, image, sideStart, frame, positionsPixels, sampledOffsets, tangentialOffsets) {
  const numberOfSamples = positionsPixels.length;
  const numberOfTangential = tangentialOffsets.length;
  const numberOfOffsets = sampledOffsets.length;
  const numberOfRows = numberOfSamples * numberOfTangential;
  const mapX = new Float32Array(numberOfRows * numberOfOffsets);
  const mapY = new Float32Array(numberOfRows * numberOfOffsets);
  const rowInside = new Uint8Array(numberOfRows * numberOfOffsets);
  const imageWidth = image.cols;
  const imageHeight = image.rows;
  const { unitTangent, unitNormal } = frame;
  for (let row = 0; row < numberOfRows; row += 1) {
    const along = positionsPixels[Math.floor(row / numberOfTangential)] + tangentialOffsets[row % numberOfTangential];
    const baseX = sideStart[0] + along * unitTangent[0];
    const baseY = sideStart[1] + along * unitTangent[1];
    for (let offset = 0; offset < numberOfOffsets; offset += 1) {
      const pointX = baseX + sampledOffsets[offset] * unitNormal[0];
      const pointY = baseY + sampledOffsets[offset] * unitNormal[1];
      const cell = row * numberOfOffsets + offset;
      mapX[cell] = pointX;
      mapY[cell] = pointY;
      rowInside[cell] = pointX >= 0 && pointX <= imageWidth - 1 && pointY >= 0 && pointY <= imageHeight - 1 ? 1 : 0;
    }
  }
  const numberOfChannels = image.channels();
  const sampledPixels = sampleImageBilinear(cv, image, mapX, mapY, numberOfRows, numberOfOffsets);
  const profiles = new Float32Array(numberOfSamples * numberOfOffsets * numberOfChannels);
  const insideImage = new Uint8Array(numberOfSamples * numberOfOffsets);
  const valuesPerRow = numberOfOffsets * numberOfChannels;
  for (let sample = 0; sample < numberOfSamples; sample += 1) {
    for (let value = 0; value < valuesPerRow; value += 1) {
      let sum = 0; // uint8 sums are exact; the float32 mean is one rounding
      for (let tangential = 0; tangential < numberOfTangential; tangential += 1) {
        sum += sampledPixels[(sample * numberOfTangential + tangential) * valuesPerRow + value];
      }
      profiles[sample * valuesPerRow + value] = sum / numberOfTangential;
    }
    for (let offset = 0; offset < numberOfOffsets; offset += 1) {
      let allInside = 1;
      for (let tangential = 0; tangential < numberOfTangential; tangential += 1) {
        allInside &= rowInside[(sample * numberOfTangential + tangential) * numberOfOffsets + offset];
      }
      insideImage[sample * numberOfOffsets + offset] = allInside;
    }
  }
  return { profiles, insideImage, shape: { numberOfSamples, numberOfOffsets, numberOfChannels } };
}

/**
 * Paper-boundary score for offsets in [-inwardBand, +outwardBand] at each sample.
 *
 * Args mirror score_side_edge_profiles; `config` supplies the window sizes and score mode.
 * Returns a SideScoreProfiles whose scores are -Infinity where a window leaves the image.
 */
export function scoreSideEdgeProfiles(cv, image, sideStart, sideEnd, quadCentroid, inwardBandPixels, outwardBandPixels, numberOfSamples, cornerMarginPixels, paperColour, config) {
  const frame = computeSideFrame(sideStart, sideEnd, quadCentroid);
  const { sideLength } = frame;
  const margin = Math.min(cornerMarginPixels, MAXIMUM_CORNER_MARGIN_FRACTION_OF_SIDE * sideLength);
  const positionsPixels = linspace(margin, sideLength - margin, numberOfSamples);
  const inward = Math.ceil(inwardBandPixels);
  const outward = Math.ceil(outwardBandPixels);
  const innerWindow = config.innerWindowPixels;
  const outerWindow = config.outerWindowPixels;
  const sampledOffsets = new Float64Array(inward + innerWindow + outward + outerWindow + 1);
  for (let index = 0; index < sampledOffsets.length; index += 1) sampledOffsets[index] = -inward - innerWindow + index;
  const { profiles, insideImage, shape } = sampleNormalProfiles(cv, image, sideStart, frame, positionsPixels, sampledOffsets, config.tangentialOffsetsPixels);
  const numberOfOffsets = shape.numberOfOffsets;

  let edgeFeature;
  if (config.edgeScoreMode === "paper_distance") {
    edgeFeature = computePaperDistanceFeature(profiles, shape, paperColour);
  } else if (config.edgeScoreMode === "two_class") {
    edgeFeature = computeTwoClassFeature(profiles, shape, paperColour, config.backgroundWindowPixels, config.minimumPaperBackgroundContrast, config.tentBeyondPaper);
  } else {
    throw new Error(`unknown edgeScoreMode ${JSON.stringify(config.edgeScoreMode)}`);
  }
  const scores = computeBoxWindowScores(edgeFeature, numberOfSamples, numberOfOffsets, innerWindow, outerWindow);
  const numberOfCentres = numberOfOffsets - innerWindow - outerWindow;

  if (config.edgeScoreMode === "two_class" && config.contactLineHalfWidthPixels > 0) {
    replaceColourlessRowsWithContactScores(profiles, shape, paperColour, scores, numberOfCentres, innerWindow, config);
  }
  for (let sample = 0; sample < numberOfSamples; sample += 1) {
    for (let centreIndex = 0; centreIndex < numberOfCentres; centreIndex += 1) {
      const cell = sample * numberOfCentres + centreIndex;
      let windowInside = Number.isFinite(scores[cell]);
      for (let shift = 0; windowInside && shift <= innerWindow + outerWindow; shift += 1) {
        windowInside = insideImage[sample * numberOfOffsets + centreIndex + shift] === 1;
      }
      if (!windowInside) scores[cell] = -Infinity;
    }
  }
  return new SideScoreProfiles({
    sideStart: [sideStart[0], sideStart[1]],
    unitTangent: frame.unitTangent,
    unitNormal: frame.unitNormal,
    sideLength,
    positionsPixels,
    normalOffsets: sampledOffsets.slice(innerWindow, innerWindow + numberOfCentres),
    scores,
  });
}

/**
 * Rows with no colour evidence (white on white) or weak paper/background contrast score the
 * paper's contact shadow instead: a soft shadow halo brightens back THROUGH the paper colour
 * and would fake an outer colour edge.
 */
function replaceColourlessRowsWithContactScores(profiles, shape, paperColour, scores, numberOfCentres, firstCentre, config) {
  const { numberOfSamples, numberOfOffsets, numberOfChannels } = shape;
  const backgroundColours = computeBackgroundColours(profiles, numberOfSamples, numberOfOffsets, numberOfChannels, config.backgroundWindowPixels);
  for (let sample = 0; sample < numberOfSamples; sample += 1) {
    let squaredContrast = 0;
    for (let channel = 0; channel < numberOfChannels; channel += 1) {
      const difference = backgroundColours[sample * numberOfChannels + channel] - paperColour[channel];
      squaredContrast += difference * difference;
    }
    const lowContrast = Math.sqrt(squaredContrast) < config.contactLineBelowContrast;
    let anyFinite = false;
    for (let centreIndex = 0; centreIndex < numberOfCentres && !anyFinite; centreIndex += 1) {
      anyFinite = Number.isFinite(scores[sample * numberOfCentres + centreIndex]);
    }
    if (!anyFinite || lowContrast) {
      writeContactLineScoresForSample(profiles, shape, sample, firstCentre, numberOfCentres, config.contactLineHalfWidthPixels, config.contactLineContrastUnit, scores);
    }
  }
}
