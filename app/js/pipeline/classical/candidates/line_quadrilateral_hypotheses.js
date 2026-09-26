/**
 * Rectangle hypotheses from pairs of parallel edge segments (Python
 * `line_quadrilateral_hypotheses.py`).
 *
 * 1. Parallel pair: segments within `lineParallelToleranceDegrees`, a check-sized distance
 *    apart, whose extents overlap along their shared direction.
 * 2. End positions: the pair's own extent ends, plus every roughly perpendicular segment
 *    ("end cap") lying between the two lines (a seam between touching checks, a short border).
 * 3. Every (start, end) choice whose span has a check aspect with the pair's separation and
 *    which both segments cover becomes a quad: the pair lines intersected with the end lines.
 * Budgeted against straight-line textures (gingham, tile): pairs longest first up to
 * `lineMaximumPairs`, `MAXIMUM_CAPS_PER_PAIR` caps per pair, and the `lineMaximumHypotheses`
 * best-supported hypotheses after grid deduplication.
 */

import { numpyArgsort, roundHalfEven } from "../numeric/numpy_compatibility.js";
import { intersectLines, orderCornersClockwise } from "../geometry/quadrilateral_geometry.js";
import { segmentLength } from "./collinear_segment_merging.js";

const MINIMUM_PAIR_OVERLAP_FRACTION = 0.25; // of the shorter segment
const MINIMUM_SPAN_COVERAGE_FRACTION = 0.4; // each pair segment must cover this much of the span
const MINIMUM_CAP_COVERAGE_FRACTION = 0.5; // an end cap must span this much of the separation
const POSITION_DEDUPLICATION_PIXELS = 4.0;
const HYPOTHESIS_DEDUPLICATION_GRID_PIXELS = 6.0;
const MAXIMUM_CAPS_PER_PAIR = 6;
const CAP_SUPPORT_WEIGHT = 0.5; // a real cap line at an end adds this much support per end
const MINIMUM_DIRECTION_LENGTH = 1e-9;
const MINIMUM_ASPECT_DENOMINATOR = 1e-6;
const RIGHT_ANGLE_DEGREES = 90.0;
const HALF_TURN_DEGREES = 180.0;
const RADIANS_TO_DEGREES = 180.0 / Math.PI;

/** `[vx, vy, x0, y0]` line with unit direction, anchored at the segment start. */
function segmentLine(segment) {
  const directionX = segment[2] - segment[0];
  const directionY = segment[3] - segment[1];
  const length = Math.max(Math.sqrt(directionX * directionX + directionY * directionY), MINIMUM_DIRECTION_LENGTH);
  return [directionX / length, directionY / length, segment[0], segment[1]];
}

/** Length of the intersection of two 1-D intervals (0 when disjoint). */
function intervalOverlap(firstInterval, secondInterval) {
  return Math.max(0.0, Math.min(firstInterval[1], secondInterval[1]) - Math.max(firstInterval[0], secondInterval[0]));
}

/** `np.remainder(value, 180.0)` (result takes the divisor's sign). */
function remainderHalfTurn(value) {
  const remainder = value % HALF_TURN_DEGREES;
  return remainder !== 0 && remainder < 0 ? remainder + HALF_TURN_DEGREES : remainder;
}

/** Sorted interval of the two endpoints of `segment` projected onto `direction` from `origin`. */
function projectedInterval(segment, originX, originY, axisX, axisY) {
  const first = (segment[0] - originX) * axisX + (segment[1] - originY) * axisY;
  const second = (segment[2] - originX) * axisX + (segment[3] - originY) * axisY;
  return first <= second ? [first, second] : [second, first];
}

/** Pairwise undirected angle differences and the (i < j) parallel pairs in row-major order. */
function parallelPairsLongestFirst(segments, angles, lengths, config) {
  const segmentCount = segments.length;
  const angleDifference = Array.from({ length: segmentCount }, () => new Float64Array(segmentCount));
  const firstIndices = [];
  const secondIndices = [];
  for (let first = 0; first < segmentCount; first += 1) {
    for (let second = 0; second < segmentCount; second += 1) {
      const difference = Math.abs(angles[first] - angles[second]);
      angleDifference[first][second] = Math.min(difference, HALF_TURN_DEGREES - difference);
      if (second > first && angleDifference[first][second] < config.lineParallelToleranceDegrees) {
        firstIndices.push(first);
        secondIndices.push(second);
      }
    }
  }
  const pairOrder = numpyArgsort(firstIndices.map((first, index) => -(lengths[first] + lengths[secondIndices[index]])));
  const pairs = Array.from(pairOrder.subarray(0, Math.min(config.lineMaximumPairs, pairOrder.length)), (index) => [firstIndices[index], secondIndices[index]]);
  return { angleDifference, pairs };
}

/** Candidate end positions for one pair: extent ends plus accepted caps, deduplicated. */
function pairEndPositions(context, firstInterval, secondInterval, capIndices, separation, meanSecondOffset) {
  const { segments, originX, originY, directionX, directionY, normalX, normalY } = context;
  const endPositions = [
    [Math.min(firstInterval[0], secondInterval[0]), null],
    [Math.max(firstInterval[1], secondInterval[1]), null],
  ];
  const pairNormalExtent = meanSecondOffset >= 0 ? [0.0, meanSecondOffset] : [meanSecondOffset, 0.0];
  let acceptedCapCount = 0;
  for (const capIndex of capIndices) {
    if (acceptedCapCount >= MAXIMUM_CAPS_PER_PAIR) break;
    const capNormalExtent = projectedInterval(segments[capIndex], originX, originY, normalX, normalY);
    if (intervalOverlap(capNormalExtent, pairNormalExtent) < MINIMUM_CAP_COVERAGE_FRACTION * separation) continue;
    const capSegment = segments[capIndex];
    const capStartPosition = (capSegment[0] - originX) * directionX + (capSegment[1] - originY) * directionY;
    const capEndPosition = (capSegment[2] - originX) * directionX + (capSegment[3] - originY) * directionY;
    endPositions.push([(capStartPosition + capEndPosition) / 2, capIndex]);
    acceptedCapCount += 1;
  }
  endPositions.sort((first, second) => first[0] - second[0]);
  const deduplicatedPositions = [];
  for (const [position, capIndex] of endPositions) {
    const last = deduplicatedPositions[deduplicatedPositions.length - 1];
    if (last && position - last[0] < POSITION_DEDUPLICATION_PIXELS) {
      if (capIndex !== null) deduplicatedPositions[deduplicatedPositions.length - 1] = [position, capIndex]; // prefer a real cap line
      continue;
    }
    deduplicatedPositions.push([position, capIndex]);
  }
  return deduplicatedPositions;
}

/** Hypotheses `[support, corners]` from every start/end choice of one pair. */
function pairHypotheses(context, firstIndex, secondIndex, firstInterval, secondInterval, positions, separation, config) {
  const { lines, originX, originY, directionX, directionY, normalX, normalY } = context;
  const hypotheses = [];
  for (let startChoice = 0; startChoice < positions.length; startChoice += 1) {
    for (let endChoice = startChoice + 1; endChoice < positions.length; endChoice += 1) {
      const [startPosition, startCap] = positions[startChoice];
      const [endPosition, endCap] = positions[endChoice];
      const span = endPosition - startPosition;
      const aspect = Math.max(span, separation) / Math.max(Math.min(span, separation), MINIMUM_ASPECT_DENOMINATOR);
      if (!(config.minimumAspectRatio <= aspect && aspect <= config.maximumAspectRatio)) continue;
      const firstCoverage = intervalOverlap(firstInterval, [startPosition, endPosition]);
      const secondCoverage = intervalOverlap(secondInterval, [startPosition, endPosition]);
      if (Math.min(firstCoverage, secondCoverage) < MINIMUM_SPAN_COVERAGE_FRACTION * span) continue;
      const capCount = (startCap !== null ? 1 : 0) + (endCap !== null ? 1 : 0);
      const support = (firstCoverage + secondCoverage) / (2.0 * span) + CAP_SUPPORT_WEIGHT * capCount / 2.0;
      const endLines = [[startPosition, startCap], [endPosition, endCap]].map(([position, capIndex]) => (
        capIndex !== null ? lines[capIndex] : [normalX, normalY, originX + position * directionX, originY + position * directionY]
      ));
      const corners = [
        intersectLines(lines[firstIndex], endLines[0]),
        intersectLines(lines[firstIndex], endLines[1]),
        intersectLines(lines[secondIndex], endLines[1]),
        intersectLines(lines[secondIndex], endLines[0]),
      ];
      if (corners.some((corner) => corner === null)) continue;
      hypotheses.push([support, orderCornersClockwise(corners)]);
    }
  }
  return hypotheses;
}

/** Drop hypotheses whose corners all round to the same 6 px grid cell as an earlier one. */
export function deduplicateHypotheses(hypotheses) {
  const seenKeys = new Set();
  const uniqueHypotheses = [];
  for (const corners of hypotheses) {
    const key = corners.map(([x, y]) => `${roundHalfEven(x / HYPOTHESIS_DEDUPLICATION_GRID_PIXELS)},${roundHalfEven(y / HYPOTHESIS_DEDUPLICATION_GRID_PIXELS)}`).join(";");
    if (seenKeys.has(key)) continue;
    seenKeys.add(key);
    uniqueHypotheses.push(corners);
  }
  return uniqueHypotheses;
}

/** All rectangle hypotheses (four clockwise corners each) from the segment set. */
export function buildLineQuadrilateralHypotheses(segments, imageLongSide, config) {
  if (segments.length < 2) return [];
  const lines = segments.map(segmentLine);
  const angles = lines.map(([vx, vy]) => remainderHalfTurn(Math.atan2(vy, vx) * RADIANS_TO_DEGREES));
  const lengths = segments.map(segmentLength);
  const minimumSeparation = config.lineMinimumSeparationFraction * imageLongSide;
  const { angleDifference, pairs } = parallelPairsLongestFirst(segments, angles, lengths, config);
  const isPerpendicular = (first, second) => Math.abs(angleDifference[first][second] - RIGHT_ANGLE_DEGREES) < config.linePerpendicularToleranceDegrees;

  const hypotheses = [];
  for (const [firstIndex, secondIndex] of pairs) {
    const [directionX, directionY, originX, originY] = lines[firstIndex];
    const normalX = -directionY;
    const normalY = directionX;
    const secondSegment = segments[secondIndex];
    const secondStartOffset = (secondSegment[0] - originX) * normalX + (secondSegment[1] - originY) * normalY;
    const secondEndOffset = (secondSegment[2] - originX) * normalX + (secondSegment[3] - originY) * normalY;
    const meanSecondOffset = (secondStartOffset + secondEndOffset) / 2;
    const separation = Math.abs(meanSecondOffset);
    if (separation < minimumSeparation) continue;
    const firstInterval = projectedInterval(segments[firstIndex], originX, originY, directionX, directionY);
    const secondInterval = projectedInterval(secondSegment, originX, originY, directionX, directionY);
    if (intervalOverlap(firstInterval, secondInterval) < MINIMUM_PAIR_OVERLAP_FRACTION * Math.min(lengths[firstIndex], lengths[secondIndex])) continue;

    const capCandidates = [];
    for (let capIndex = 0; capIndex < segments.length; capIndex += 1) {
      if (isPerpendicular(firstIndex, capIndex) && isPerpendicular(secondIndex, capIndex)) capCandidates.push(capIndex);
    }
    const capOrder = numpyArgsort(capCandidates.map((capIndex) => -lengths[capIndex]));
    const capIndices = Array.from(capOrder, (index) => capCandidates[index]);
    const context = { segments, lines, originX, originY, directionX, directionY, normalX, normalY };
    const positions = pairEndPositions(context, firstInterval, secondInterval, capIndices, separation, meanSecondOffset);
    hypotheses.push(...pairHypotheses(context, firstIndex, secondIndex, firstInterval, secondInterval, positions, separation, config));
  }
  hypotheses.sort((first, second) => second[0] - first[0]);
  return deduplicateHypotheses(hypotheses.map(([, corners]) => corners)).slice(0, config.lineMaximumHypotheses);
}
