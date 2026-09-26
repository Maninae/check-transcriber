/**
 * `cv2.fitLine(points, DIST_HUBER, 0, 0.01, 0.01)` for 2D float32 points, bit-exact with the
 * Python cv2 (OpenCV 5.0, arm64). A line-by-line port of `fitLine2D` in `imgproc/linefit.cpp`.
 *
 * The algorithm: 20 restarts, each seeded with 10 points drawn by a fixed-seed `cv::RNG`, then up
 * to 30 iteratively reweighted least-squares steps with Huber weights (c = 1.345). The line with
 * the smallest sum of point distances over every step of every restart wins. That final
 * comparison is a knife-edge: when two restarts converge to different lines with near-equal
 * error sums, one ulp decides the winner, and the losing line can sit a few tenths of a pixel
 * away. `cv.fitLine` in OpenCV.js rounds products that clang on arm64 fuses, so it picked a
 * different restart on eval_000207 (0.18 px on one quad side, which then flipped the detection).
 * The fused spots, all reproduced here with exact FMA emulation:
 * - weighted moments: `x2 - x * x`, `y2 - y * y`, `xy - x * y` (float64)
 * - point-to-line distance: `nx * x + ny * y` (float32)
 * - convergence test: `line . previousLine` (float32)
 *
 * Returns `[vx, vy, x0, y0]` (float32 values), like `cv2.fitLine(...).reshape(4)`.
 */

import { fusedMultiplyAddFloat32, fusedMultiplyAddFloat64 } from "./fused_multiply_add.js";
import { OpenCvRandomGenerator } from "./opencv_random_generator.js";

const fround = Math.fround;
const FLT_EPSILON = 1.1920928955078125e-7;
const RESTART_COUNT = 20;
const ITERATIONS_PER_RESTART = 30;
const SEED_POINT_COUNT = 10;
const HUBER_CONSTANT = fround(1.345); // OpenCV's default when param <= 0
const DISTANCE_DELTA = fround(0.01); // reps
const ANGLE_DELTA = fround(0.01); // aeps

/** Weighted least-squares line (`fitLine2D_wods`), written into `line`. */
function fitWeightedLine(points, count, weights, line) {
  let sumX = 0;
  let sumY = 0;
  let sumXX = 0;
  let sumYY = 0;
  let sumXY = 0;
  let weightSum = 0;
  for (let index = 0; index < count; index += 1) {
    const weight = weights[index];
    const pointX = points[2 * index];
    const pointY = points[2 * index + 1];
    const weightedX = fround(weight * pointX); // float32 products, accumulated in float64
    const weightedY = fround(weight * pointY);
    sumX += weightedX;
    sumY += weightedY;
    sumXX += fround(weightedX * pointX);
    sumYY += fround(weightedY * pointY);
    sumXY += fround(weightedX * pointY);
    weightSum += weight;
  }
  const meanX = sumX / weightSum;
  const meanY = sumY / weightSum;
  const varianceX = fusedMultiplyAddFloat64(-meanX, meanX, sumXX / weightSum);
  const varianceY = fusedMultiplyAddFloat64(-meanY, meanY, sumYY / weightSum);
  const covariance = fusedMultiplyAddFloat64(-meanX, meanY, sumXY / weightSum);
  const halfAngle = fround(fround(Math.atan2(2 * covariance, varianceX - varianceY)) / 2);
  line[0] = fround(Math.cos(halfAngle));
  line[1] = fround(Math.sin(halfAngle));
  line[2] = fround(meanX);
  line[3] = fround(meanY);
}

/** Per-point distance to `line` into `distances`; returns their float64 sum (`calcDist2D`). */
function measureDistances(points, count, line, distances) {
  const lineX = line[2];
  const lineY = line[3];
  const normalX = line[1];
  const normalY = -line[0];
  let distanceSum = 0;
  for (let index = 0; index < count; index += 1) {
    const offsetX = fround(points[2 * index] - lineX);
    const offsetY = fround(points[2 * index + 1] - lineY);
    distances[index] = fround(Math.abs(fusedMultiplyAddFloat32(normalX, offsetX, fround(normalY * offsetY))));
    distanceSum += distances[index];
  }
  return distanceSum;
}

/**
 * Huber line through the first `count` points of a flat float32 `[x0, y0, x1, y1, ...]` array.
 *
 * Returns `[vx, vy, x0, y0]`.
 */
export function fitLineHuberExact(points, count) {
  const errorFloor = count * FLT_EPSILON;
  const bestLine = new Float32Array(4);
  const line = new Float32Array(4);
  const previousLine = new Float32Array(4);
  const weights = new Float32Array(count);
  const distances = new Float32Array(count);
  const randomGenerator = new OpenCvRandomGenerator();
  let smallestError = Number.MAX_VALUE;
  let error = 0;
  for (let restart = 0; restart < RESTART_COUNT; restart += 1) {
    weights.fill(0);
    for (let seeded = 0; seeded < Math.min(count, SEED_POINT_COUNT);) {
      const pointIndex = randomGenerator.uniformBelow(count);
      if (weights[pointIndex] < FLT_EPSILON) {
        weights[pointIndex] = 1;
        seeded += 1;
      }
    }
    fitWeightedLine(points, count, weights, line);
    for (let iteration = 0; iteration < ITERATIONS_PER_RESTART; iteration += 1) {
      if (iteration > 0) {
        let cosine = fusedMultiplyAddFloat32(line[0], previousLine[0], fround(line[1] * previousLine[1]));
        cosine = Math.min(Math.max(cosine, -1), 1);
        if (Math.abs(Math.acos(cosine)) < ANGLE_DELTA) {
          const shiftX = fround(Math.abs(fround(line[2] - previousLine[2])));
          const shiftY = fround(Math.abs(fround(line[3] - previousLine[3])));
          if ((shiftX > shiftY ? shiftX : shiftY) < DISTANCE_DELTA) break;
        }
      }
      error = measureDistances(points, count, line, distances);
      if (error < smallestError) {
        smallestError = error;
        bestLine.set(line);
        if (error < errorFloor) break;
      }
      for (let index = 0; index < count; index += 1) {
        weights[index] = distances[index] < HUBER_CONSTANT ? 1 : fround(HUBER_CONSTANT / distances[index]);
      }
      let weightSum = 0;
      for (let index = 0; index < count; index += 1) weightSum += weights[index];
      if (Math.abs(weightSum) > FLT_EPSILON) {
        const inverseWeightSum = 1 / weightSum;
        for (let index = 0; index < count; index += 1) weights[index] = weights[index] * inverseWeightSum;
      } else {
        weights.fill(1);
      }
      previousLine.set(line);
      fitWeightedLine(points, count, weights, line);
    }
    if (error < smallestError) {
      smallestError = error;
      bestLine.set(line);
      if (error < errorFloor) break;
    }
  }
  return [bestLine[0], bestLine[1], bestLine[2], bestLine[3]];
}
