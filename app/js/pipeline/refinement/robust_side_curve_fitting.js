/**
 * Robust fit of a side's edge as offset = polynomial(position), and corner intersection.
 *
 * Port of experiments/detection/refinement/robust_side_curve_fitting.py:
 * RANSAC over point pairs for the dominant straight line (an overlapping check or a fold
 * owns the outliers), then Tukey-biweight IRLS for the requested degree (straight first,
 * the curvature term joins once the inlier set has settled). Corners are the Newton
 * intersection of the incoming side's END with the outgoing side's START.
 *
 * - RANSAC draws come from the shared `randomGenerator` (numpy-exact PCG64 by default), in
 *   numpy's row-major (iterations x 2) order, so hypotheses match the Python one for one.
 * - Least squares uses Householder QR (numpy uses SVD; both are exact to ~1e-15 relative on
 *   these full-rank, [-1, 1]-normalised systems).
 */

import { evaluatePolynomial, SideCurve } from "./side_curve.js";
import { median } from "./numpy_compatible_math.js";

const TUKEY_REWEIGHTING_ROUNDS = 5;
const TUKEY_SCALE_INLIER_MULTIPLE = 2.0;
const NEWTON_ITERATIONS = 8;
const NEWTON_MAXIMUM_STEP_PIXELS = 1e4;
const NEWTON_CONVERGED_STEP_PIXELS = 1e-4;
const MINIMUM_JACOBIAN_DETERMINANT = 1e-6;
const MINIMUM_POSITION_GAP = 1e-6;
const MINIMUM_POINT_WEIGHT = 1e-6;

/**
 * Weighted least squares for offset = sum_k c_k x^k, over the points with index in `rows`.
 *
 * Solves min || sqrt(w) * (A c - b) || with Householder QR; returns a Float64Array of degree+1.
 */
export function fitWeightedPolynomial(normalisedPositions, offsets, weights, degree, rows) {
  const numberOfTerms = degree + 1;
  const width = numberOfTerms + 1; // augmented [design | target]: the target is reflected like a column
  const numberOfRows = rows.length;
  const augmented = new Float64Array(numberOfRows * width);
  for (let row = 0; row < numberOfRows; row += 1) {
    const pointIndex = rows[row];
    const rootWeight = Math.sqrt(Math.max(weights[pointIndex], 0));
    let power = 1;
    for (let term = 0; term < numberOfTerms; term += 1) {
      augmented[row * width + term] = power * rootWeight;
      power *= normalisedPositions[pointIndex];
    }
    augmented[row * width + numberOfTerms] = offsets[pointIndex] * rootWeight;
  }
  for (let column = 0; column < numberOfTerms; column += 1) {
    let columnSquaredNorm = 0; // bounded values (|x| <= 1, weights <= 1, offsets ~1e2): no overflow risk
    for (let row = column; row < numberOfRows; row += 1) columnSquaredNorm += augmented[row * width + column] ** 2;
    if (columnSquaredNorm === 0) continue;
    const columnNorm = Math.sqrt(columnSquaredNorm);
    const pivot = augmented[column * width + column];
    const alpha = pivot > 0 ? -columnNorm : columnNorm;
    augmented[column * width + column] = pivot - alpha; // v = x - alpha e1, stored in place
    let vNormSquared = 0;
    for (let row = column; row < numberOfRows; row += 1) vNormSquared += augmented[row * width + column] ** 2;
    for (let other = column + 1; other < width; other += 1) {
      let projection = 0;
      for (let row = column; row < numberOfRows; row += 1) projection += augmented[row * width + column] * augmented[row * width + other];
      const scale = (2 * projection) / vNormSquared;
      for (let row = column; row < numberOfRows; row += 1) augmented[row * width + other] -= scale * augmented[row * width + column];
    }
    augmented[column * width + column] = alpha;
  }
  const coefficients = new Float64Array(numberOfTerms);
  for (let term = numberOfTerms - 1; term >= 0; term -= 1) {
    let remainder = augmented[term * width + numberOfTerms];
    for (let other = term + 1; other < numberOfTerms; other += 1) remainder -= augmented[term * width + other] * coefficients[other];
    const diagonal = augmented[term * width + term];
    coefficients[term] = diagonal === 0 ? 0 : remainder / diagonal; // rank-deficient: zero, never NaN
  }
  return coefficients;
}

/** Index of the RANSAC pair hypothesis with the largest inlier weight, and its inlier mask. */
function selectRansacInliers(normalisedPositions, offsets, weights, inlierDistance, iterations, randomGenerator) {
  const numberOfPoints = normalisedPositions.length;
  const pairIndices = randomGenerator.drawIntegersBelow(numberOfPoints, iterations * 2);
  let bestInliers = new Uint8Array(numberOfPoints);
  let bestWeight = -Infinity;
  const candidateInliers = new Uint8Array(numberOfPoints);
  for (let iteration = 0; iteration < iterations; iteration += 1) {
    const first = pairIndices[2 * iteration];
    const second = pairIndices[2 * iteration + 1];
    const positionGap = normalisedPositions[second] - normalisedPositions[first];
    const usable = Math.abs(positionGap) > MINIMUM_POSITION_GAP;
    const slope = usable ? (offsets[second] - offsets[first]) / positionGap : 0.0;
    let inlierWeight = 0;
    for (let point = 0; point < numberOfPoints; point += 1) {
      const predicted = offsets[first] + slope * (normalisedPositions[point] - normalisedPositions[first]);
      const isInlier = usable && Math.abs(offsets[point] - predicted) < inlierDistance;
      candidateInliers[point] = isInlier ? 1 : 0;
      if (isInlier) inlierWeight += weights[point];
    }
    if (inlierWeight > bestWeight) {
      bestWeight = inlierWeight;
      bestInliers = candidateInliers.slice();
    }
  }
  return bestInliers;
}

/**
 * RANSAC straight line, then Tukey IRLS polynomial of `degree`, in `curveFrame`'s frame.
 *
 * Args: points as parallel Float64Arrays (positions px along the side, offsets px, strengths).
 * Returns a new SideCurve with fit statistics, or null (< 2 points or no consensus).
 */
export function fitRobustSideCurve(curveFrame, positions, offsets, strengths, degree, inlierDistancePixels, ransacIterations, randomGenerator) {
  const numberOfPoints = positions.length;
  if (numberOfPoints < Math.max(2, degree + 1)) return null;
  const normalisedPositions = Float64Array.from(positions, (position) => curveFrame.normalised(position));
  const weights = Float64Array.from(strengths, (strength) => Math.max(strength, MINIMUM_POINT_WEIGHT));
  const bestInliers = selectRansacInliers(normalisedPositions, offsets, weights, inlierDistancePixels, ransacIterations, randomGenerator);
  const inlierRows = [];
  bestInliers.forEach((isInlier, point) => { if (isInlier) inlierRows.push(point); });
  if (inlierRows.length < degree + 1) return null;
  const allRows = Array.from({ length: numberOfPoints }, (_, point) => point);
  let coefficients = fitWeightedPolynomial(normalisedPositions, offsets, weights, 1, inlierRows);
  const tukeyScale = TUKEY_SCALE_INLIER_MULTIPLE * inlierDistancePixels;
  const combinedWeights = new Float64Array(numberOfPoints);
  for (let round = 0; round < TUKEY_REWEIGHTING_ROUNDS; round += 1) {
    const roundDegree = round >= 1 ? degree : 1;
    let nonzeroCount = 0;
    for (let point = 0; point < numberOfPoints; point += 1) {
      const residual = offsets[point] - evaluatePolynomial(coefficients, normalisedPositions[point]);
      const relative = residual / tukeyScale;
      const tukeyWeight = Math.abs(residual) < tukeyScale ? (1 - relative * relative) * (1 - relative * relative) : 0.0;
      if (tukeyWeight !== 0) nonzeroCount += 1;
      combinedWeights[point] = tukeyWeight * weights[point];
    }
    if (nonzeroCount < roundDegree + 1) break;
    coefficients = fitWeightedPolynomial(normalisedPositions, offsets, combinedWeights, roundDegree, allRows);
  }
  const inlierStrengths = [];
  let squaredResidualSum = 0;
  for (let point = 0; point < numberOfPoints; point += 1) {
    const residual = offsets[point] - evaluatePolynomial(coefficients, normalisedPositions[point]);
    if (Math.abs(residual) < inlierDistancePixels) {
      inlierStrengths.push(strengths[point]);
      squaredResidualSum += residual * residual;
    }
  }
  const inlierCount = inlierStrengths.length;
  return curveFrame.withFit({
    coefficients,
    inlierCount,
    medianInlierStrength: inlierCount > 0 ? median(inlierStrengths) : 0.0,
    inlierResidualRms: inlierCount > 0 ? Math.sqrt(squaredResidualSum / inlierCount) : 0.0,
  });
}

/** Solve the 2x2 system [[a, b], [c, d]] x = rhs by LU with partial pivoting (LAPACK order); null if singular-ish. */
function solveTwoByTwo(a, b, c, d, rhs) {
  const swap = Math.abs(c) > Math.abs(a);
  const [u00, u01, l10Numerator, l11, r0, r1] = swap ? [c, d, a, b, rhs[1], rhs[0]] : [a, b, c, d, rhs[0], rhs[1]];
  const lower = l10Numerator * (1 / u00);
  const u11 = l11 - lower * u01;
  const determinant = (swap ? -1 : 1) * u00 * u11;
  if (!(Math.abs(determinant) >= MINIMUM_JACOBIAN_DETERMINANT)) return null;
  const second = (r1 - lower * r0) / u11;
  return [(r0 - second * u01) / u00, second];
}

/** Corner where the incoming side's END meets the outgoing side's START, or null (parallel / diverged). */
export function intersectSideCurves(incomingCurve, outgoingCurve) {
  let incomingPosition = incomingCurve.sideLength;
  let outgoingPosition = 0.0;
  for (let iteration = 0; iteration < NEWTON_ITERATIONS; iteration += 1) {
    const incoming = incomingCurve.pointAndDerivative(incomingPosition);
    const outgoing = outgoingCurve.pointAndDerivative(outgoingPosition);
    const step = solveTwoByTwo(
      incoming.derivative[0], -outgoing.derivative[0], incoming.derivative[1], -outgoing.derivative[1],
      [outgoing.point[0] - incoming.point[0], outgoing.point[1] - incoming.point[1]],
    );
    if (step === null) return null;
    const largestStep = Math.max(Math.abs(step[0]), Math.abs(step[1]));
    if (!Number.isFinite(step[0]) || !Number.isFinite(step[1]) || largestStep > NEWTON_MAXIMUM_STEP_PIXELS) return null;
    incomingPosition += step[0];
    outgoingPosition += step[1];
    if (largestStep < NEWTON_CONVERGED_STEP_PIXELS) break;
  }
  return incomingCurve.pointAndDerivative(incomingPosition).point;
}
