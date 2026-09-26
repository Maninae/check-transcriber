/**
 * Probabilistic Hough line segments, bit-exact with the Python cv2 (OpenCV 5.0, arm64).
 *
 * A line-by-line port of OpenCV's `HoughLinesProbabilistic` (what `cv2.HoughLinesP` runs).
 * OpenCV.js computes the same algorithm but its accumulator vote `cvRound(x * cos + y * sin)`
 * rounds both float32 products, while clang on arm64 contracts it into a fused multiply-add;
 * a vote landing in the neighbouring rho bin changes which points are consumed by which line,
 * and the output drifts (2970 vs 2966 raw segments on one scene). This port uses the fused
 * vote and OpenCV's fixed-seed multiply-with-carry `cv::RNG` so the random point order matches.
 *
 * Returns segments as `[x1, y1, x2, y2]` integer arrays, in OpenCV's output order.
 */

import { OpenCvRandomGenerator } from "../numeric/opencv_random_generator.js";

const fround = Math.fround;
const FIXED_POINT_SHIFT = 16;
const FIXED_POINT_ONE = 1 << FIXED_POINT_SHIFT;
const FIXED_POINT_HALF = 1 << (FIXED_POINT_SHIFT - 1);
// Float32 rounding moves a projection by < 5e-4 px while |projection| < 4096, so the float64
// value alone decides the bin unless it is within this margin of a half-integer.
const ROUNDING_AMBIGUITY_MARGIN = 2e-3;
const MAXIMUM_EXACT_SHORTCUT_EXTENT = 4096;
const CV_PI = 3.1415926535897932384626433832795;

/** OpenCV `cvRound` (round half to even) of a float value. */
function roundHalfEven(value) {
  const rounded = Math.round(value);
  return rounded - value === 0.5 && (rounded & 1) !== 0 ? rounded - 1 : rounded;
}

/** OpenCV `computeNumangle(0, pi, theta)`. */
function computeAngleCount(thetaStep) {
  let angleCount = Math.floor(CV_PI / thetaStep) + 1;
  if (angleCount > 1 && Math.abs(CV_PI - (angleCount - 1) * thetaStep) < thetaStep / 2) angleCount -= 1;
  return angleCount;
}

/** Vote bin (rho index + offset) of point (x, y) at one angle: cvRound(fma(x, cos, y * sin)) + offset. */
function exactVoteBin(x, y, cosine, sine, rhoOffset) {
  return roundHalfEven(fround(x * cosine + fround(y * sine))) + rhoOffset;
}

/**
 * Vote +1 along point (x, y)'s sinusoid; returns the first angle holding the running maximum
 * once it reaches `threshold` (OpenCV's `max_val`/`max_n` scan), or -1 when none does.
 *
 * Fast path: `| 0` floors the shifted float64 projection; within ROUNDING_AMBIGUITY_MARGIN of
 * a half-integer the exact float32 vote decides. Vote and unvote are separate functions so V8
 * keeps each loop monomorphic and tight.
 */
function voteAlongSinusoid(accumulator, cosines, sines, angleCount, rhoCount, rhoOffset, x, y, threshold) {
  const shiftedHalf = rhoOffset + 0.5; // keeps projections positive so `| 0` floors
  let maximumVotes = threshold - 1;
  let bestAngle = -1;
  let rowStart = 0;
  for (let angle = 0; angle < angleCount; angle += 1) {
    const shiftedProjection = x * cosines[angle] + y * sines[angle] + shiftedHalf;
    let bin = shiftedProjection | 0;
    const fraction = shiftedProjection - bin;
    if (fraction < ROUNDING_AMBIGUITY_MARGIN || fraction > 1 - ROUNDING_AMBIGUITY_MARGIN) {
      bin = exactVoteBin(x, y, cosines[angle], sines[angle], rhoOffset);
    }
    const index = rowStart + bin;
    const votes = accumulator[index] + 1;
    accumulator[index] = votes;
    if (votes > maximumVotes) {
      maximumVotes = votes;
      bestAngle = angle;
    }
    rowStart += rhoCount;
  }
  return bestAngle;
}

/** Withdraw point (x, y)'s votes (a pixel consumed by an accepted line). */
function unvoteAlongSinusoid(accumulator, cosines, sines, angleCount, rhoCount, rhoOffset, x, y) {
  const shiftedHalf = rhoOffset + 0.5;
  let rowStart = 0;
  for (let angle = 0; angle < angleCount; angle += 1) {
    const shiftedProjection = x * cosines[angle] + y * sines[angle] + shiftedHalf;
    let bin = shiftedProjection | 0;
    const fraction = shiftedProjection - bin;
    if (fraction < ROUNDING_AMBIGUITY_MARGIN || fraction > 1 - ROUNDING_AMBIGUITY_MARGIN) {
      bin = exactVoteBin(x, y, cosines[angle], sines[angle], rhoOffset);
    }
    accumulator[rowStart + bin] -= 1;
    rowStart += rhoCount;
  }
}

/**
 * `cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength, maxLineGap)`.
 *
 * Args:
 *   edgeBytes: Uint8Array edge map (nonzero = edge), `width` x `height`, row-major.
 */
export function houghLinesProbabilistic(edgeBytes, width, height, rho, theta, threshold, minimumLineLength, maximumLineGap) {
  const rhoStep = fround(rho);
  const thetaStep = fround(theta);
  const inverseRho = fround(1 / rhoStep);
  const lineLength = roundHalfEven(minimumLineLength);
  const lineGap = roundHalfEven(maximumLineGap);
  const angleCount = computeAngleCount(thetaStep);
  const rhoCount = roundHalfEven(((width + height) * 2 + 1) / rhoStep);
  const rhoOffset = Math.trunc((rhoCount - 1) / 2);
  const cosines = new Float32Array(angleCount);
  const sines = new Float32Array(angleCount);
  for (let angle = 0; angle < angleCount; angle += 1) {
    cosines[angle] = fround(Math.cos(angle * thetaStep) * inverseRho);
    sines[angle] = fround(Math.sin(angle * thetaStep) * inverseRho);
  }
  const accumulator = new Int32Array(angleCount * rhoCount);
  const cosines64 = Float64Array.from(cosines); // same float32 values, no per-read conversion
  const sines64 = Float64Array.from(sines);
  const mask = new Uint8Array(width * height);
  let edgePointCount = 0;
  for (let index = 0; index < width * height; index += 1) if (edgeBytes[index]) edgePointCount += 1;
  const pointXs = new Int32Array(edgePointCount);
  const pointYs = new Int32Array(edgePointCount);
  let pointCursor = 0;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      if (edgeBytes[y * width + x]) {
        mask[y * width + x] = 1;
        pointXs[pointCursor] = x;
        pointYs[pointCursor] = y;
        pointCursor += 1;
      }
    }
  }
  if (width + height >= MAXIMUM_EXACT_SHORTCUT_EXTENT) throw new Error(`houghLinesProbabilistic: ${width}x${height} exceeds the exact-rounding shortcut range`);
  const randomGenerator = new OpenCvRandomGenerator();
  const lines = [];
  let firstEndX = 0;
  let firstEndY = 0;
  let secondEndX = 0;
  let secondEndY = 0;
  for (let count = edgePointCount; count > 0; count -= 1) {
    const pickedIndex = randomGenerator.uniformBelow(count);
    const pointX = pointXs[pickedIndex];
    const pointY = pointYs[pickedIndex];
    pointXs[pickedIndex] = pointXs[count - 1];
    pointYs[pickedIndex] = pointYs[count - 1];
    if (!mask[pointY * width + pointX]) continue;
    const votedAngle = voteAlongSinusoid(accumulator, cosines64, sines64, angleCount, rhoCount, rhoOffset, pointX, pointY, threshold);
    if (votedAngle < 0) continue; // no angle reached the threshold
    const bestAngle = votedAngle;
    const directionA = -sines[bestAngle];
    const directionB = cosines[bestAngle];
    let startX = pointX;
    let startY = pointY;
    let stepX;
    let stepY;
    const walksAlongX = Math.abs(directionA) > Math.abs(directionB);
    if (walksAlongX) {
      stepX = directionA > 0 ? 1 : -1;
      stepY = roundHalfEven(fround(fround(directionB * FIXED_POINT_ONE) / Math.abs(directionA)));
      startY = startY * FIXED_POINT_ONE + FIXED_POINT_HALF;
    } else {
      stepY = directionB > 0 ? 1 : -1;
      stepX = roundHalfEven(fround(fround(directionA * FIXED_POINT_ONE) / Math.abs(directionB)));
      startX = startX * FIXED_POINT_ONE + FIXED_POINT_HALF;
    }
    // pass 1: walk both ways until the image border or a gap longer than lineGap
    for (let direction = 0; direction < 2; direction += 1) {
      let gap = 0;
      const deltaX = direction > 0 ? -stepX : stepX;
      const deltaY = direction > 0 ? -stepY : stepY;
      for (let fixedX = startX, fixedY = startY; ; fixedX += deltaX, fixedY += deltaY) {
        const column = walksAlongX ? fixedX : Math.floor(fixedX / FIXED_POINT_ONE);
        const row = walksAlongX ? Math.floor(fixedY / FIXED_POINT_ONE) : fixedY;
        if (column < 0 || column >= width || row < 0 || row >= height) break;
        if (mask[row * width + column]) {
          gap = 0;
          if (direction === 0) {
            firstEndX = column;
            firstEndY = row;
          } else {
            secondEndX = column;
            secondEndY = row;
          }
        } else if (++gap > lineGap) {
          break;
        }
      }
    }
    const isGoodLine = Math.abs(secondEndX - firstEndX) >= lineLength || Math.abs(secondEndY - firstEndY) >= lineLength;
    // pass 2: clear the walked pixels (and withdraw their votes when the line is kept)
    for (let direction = 0; direction < 2; direction += 1) {
      const deltaX = direction > 0 ? -stepX : stepX;
      const deltaY = direction > 0 ? -stepY : stepY;
      const endX = direction === 0 ? firstEndX : secondEndX;
      const endY = direction === 0 ? firstEndY : secondEndY;
      for (let fixedX = startX, fixedY = startY; ; fixedX += deltaX, fixedY += deltaY) {
        const column = walksAlongX ? fixedX : Math.floor(fixedX / FIXED_POINT_ONE);
        const row = walksAlongX ? Math.floor(fixedY / FIXED_POINT_ONE) : fixedY;
        if (mask[row * width + column]) {
          if (isGoodLine) unvoteAlongSinusoid(accumulator, cosines64, sines64, angleCount, rhoCount, rhoOffset, column, row);
          mask[row * width + column] = 0;
        }
        if (row === endY && column === endX) break;
      }
    }
    if (isGoodLine) lines.push([firstEndX, firstEndY, secondEndX, secondEndY]);
  }
  return lines;
}
