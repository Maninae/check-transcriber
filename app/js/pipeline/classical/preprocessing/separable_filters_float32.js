/**
 * Float32 Gaussian blur and 3x3 Sobel, bit-exact with the Python cv2 (OpenCV 5.0, arm64 NEON).
 *
 * OpenCV.js (WASM) rounds each multiply before adding; the arm64 build fuses them (NEON
 * `v_muladd` and clang-contracted scalar tails), so its float filters differ from ours by
 * ~1e-6. That is harmless for thresholds, except that the Lab a/b maps are later truncated to
 * uint8 for Canny, and a single flipped byte reshuffles the randomized HoughLinesP. So these
 * filters reproduce OpenCV's own kernels (`RowFilter<RowVec_32f>`, `SymmColumnFilter
 * <SymmColumnVec_32f>`, `SymmRowSmallFilter`/`SymmColumnSmallFilter` for Sobel) with the
 * arm64 operation order, and BORDER_REFLECT_101 like `BORDER_DEFAULT`:
 * - Gaussian row pass: `s = x0 * k0`, then `s = fma(x_k, k_k, s)` over taps left to right.
 * - Gaussian column pass: `s = S0 * k0`, then `s = fma(S[+k] + S[-k], k_k, s)`.
 * - Sobel [1, 2, 1]: `((a + c) + b) + b` on 4-lane NEON blocks, `(a + 2b) + c` in the scalar
 *   tail (the last `width % 4` columns); [-1, 0, 1]: `c - a`.
 * Maps are row-major Float32Arrays.
 */

const fround = Math.fround;
const NEON_FLOAT32_LANES = 4;

/** BORDER_REFLECT_101 index into [0, size). */
function reflect101(index, size) {
  if (size === 1) return 0;
  let reflected = index;
  while (reflected < 0 || reflected >= size) {
    reflected = reflected < 0 ? -reflected : 2 * size - 2 - reflected;
  }
  return reflected;
}

/** float32 fused multiply-add emulation: round(multiplier * multiplicand + addend) once. */
function fusedMultiplyAdd(multiplier, multiplicand, addend) {
  return fround(multiplier * multiplicand + addend);
}

/**
 * Float32 taps of OpenCV's Gaussian kernel for `sigma`, read from OpenCV.js itself.
 *
 * OpenCV builds the kernel with bit-exact soft-float math, so every platform agrees. Blurring
 * a single 1.0 pixel with an (n x 1) window yields the taps exactly (1.0 * k, then + 0).
 */
export function gaussianKernelTaps(cv, sigma) {
  const tapCount = Math.round(sigma * 4 * 2 + 1) | 1; // OpenCV float ksize: cvRound(sigma * 8 + 1) | 1
  const impulseWidth = 4 * tapCount;
  const impulseColumn = 2 * tapCount;
  const impulse = cv.Mat.zeros(1, impulseWidth, cv.CV_32F);
  const response = new cv.Mat();
  try {
    impulse.data32F[impulseColumn] = 1.0;
    cv.GaussianBlur(impulse, response, new cv.Size(tapCount, 1), sigma, sigma, cv.BORDER_DEFAULT);
    const responseValues = response.data32F;
    const anchor = tapCount >> 1;
    return Float32Array.from({ length: tapCount }, (_, tap) => responseValues[impulseColumn + anchor - tap]);
  } finally {
    impulse.delete();
    response.delete();
  }
}

/** Separable symmetric Gaussian blur of a float32 map with the given taps (arm64 order). */
export function gaussianBlurFloat32(values, width, height, taps) {
  const tapCount = taps.length;
  const anchor = tapCount >> 1;
  const rowFiltered = new Float32Array(width * height);
  const columnIndex = new Int32Array(width + tapCount);
  for (let offset = 0; offset < width + tapCount - 1; offset += 1) columnIndex[offset] = reflect101(offset - anchor, width);
  const interiorStart = Math.min(anchor, width);
  const interiorEnd = Math.max(interiorStart, width - anchor);
  for (let row = 0; row < height; row += 1) {
    const rowOffset = row * width;
    for (let column = 0; column < width; column += 1) {
      let sum;
      if (column >= interiorStart && column < interiorEnd) {
        // interior: taps read consecutive pixels, no border reflection needed
        const firstPixel = rowOffset + column - anchor;
        sum = fround(values[firstPixel] * taps[0]);
        for (let tap = 1; tap < tapCount; tap += 1) sum = fround(values[firstPixel + tap] * taps[tap] + sum);
      } else {
        sum = fround(values[rowOffset + columnIndex[column]] * taps[0]);
        for (let tap = 1; tap < tapCount; tap += 1) sum = fusedMultiplyAdd(values[rowOffset + columnIndex[column + tap]], taps[tap], sum);
      }
      rowFiltered[rowOffset + column] = sum;
    }
  }
  const blurred = new Float32Array(width * height);
  const centerTap = taps[anchor];
  for (let row = 0; row < height; row += 1) {
    const centerOffset = row * width;
    const outputRow = blurred.subarray(centerOffset, centerOffset + width);
    for (let column = 0; column < width; column += 1) outputRow[column] = fround(rowFiltered[centerOffset + column] * centerTap);
    for (let distance = 1; distance <= anchor; distance += 1) {
      const belowOffset = reflect101(row + distance, height) * width;
      const aboveOffset = reflect101(row - distance, height) * width;
      const tap = taps[anchor + distance];
      for (let column = 0; column < width; column += 1) {
        const pairSum = fround(rowFiltered[belowOffset + column] + rowFiltered[aboveOffset + column]);
        outputRow[column] = fusedMultiplyAdd(pairSum, tap, outputRow[column]);
      }
    }
  }
  return blurred;
}

/** First column index handled by OpenCV's scalar tail (after 4-lane blocks). */
function scalarTailStart(width) {
  return width - (width % NEON_FLOAT32_LANES);
}

/** [1, 2, 1] smoothing of three values: NEON block order or scalar-tail order. */
function smoothOneTwoOne(before, center, after, isVectorLane) {
  if (isVectorLane) return fround(fround(fround(before + after) + center) + center);
  return fround(fround(before + fround(center * 2)) + after);
}

/**
 * 3x3 Sobel (ksize 3, scale 1, delta 0) of a float32 map, as `cv2.Sobel(x, CV_32F, dx, dy, 3)`.
 *
 * Args:
 *   derivativeAxis: "x" for (dx=1, dy=0) or "y" for (dx=0, dy=1).
 */
export function sobelFloat32(values, width, height, derivativeAxis) {
  const tailStart = scalarTailStart(width);
  const rowFiltered = new Float32Array(width * height);
  for (let row = 0; row < height; row += 1) {
    const rowOffset = row * width;
    for (let column = 0; column < width; column += 1) {
      const before = values[rowOffset + reflect101(column - 1, width)];
      const center = values[rowOffset + column];
      const after = values[rowOffset + reflect101(column + 1, width)];
      rowFiltered[rowOffset + column] = derivativeAxis === "x"
        ? fround(after - before)
        : smoothOneTwoOne(before, center, after, column < tailStart);
    }
  }
  const filtered = new Float32Array(width * height);
  for (let row = 0; row < height; row += 1) {
    const aboveOffset = reflect101(row - 1, height) * width;
    const centerOffset = row * width;
    const belowOffset = reflect101(row + 1, height) * width;
    for (let column = 0; column < width; column += 1) {
      const above = rowFiltered[aboveOffset + column];
      const center = rowFiltered[centerOffset + column];
      const below = rowFiltered[belowOffset + column];
      filtered[centerOffset + column] = derivativeAxis === "x"
        ? smoothOneTwoOne(above, center, below, column < tailStart)
        : fround(below - above);
    }
  }
  return filtered;
}
