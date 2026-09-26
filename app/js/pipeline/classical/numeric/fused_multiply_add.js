/**
 * Correctly rounded fused multiply-add in plain JS (no `Math.fma` exists).
 *
 * clang on arm64 contracts `a * b + c` into one FMA (a single rounding); WASM rounds the product
 * first. Ports that must match the arm64 cv2 bit-for-bit use these instead of `a * b + c`.
 *
 * - `fusedMultiplyAddFloat64`: Boldo & Melquiond (2008), "Emulation of FMA and correctly rounded
 *   sums: proved algorithms using rounding to odd", Algorithm 5: exact product and sum as
 *   double-double pairs, then one round-to-odd add before the final round-to-nearest.
 * - `fusedMultiplyAddFloat32`: a float32 product is exact in float64; rounding the float64 sum
 *   to odd before `Math.fround` avoids the double-rounding trap (53 >= 2 * 24 + 2 bits).
 * - Valid for finite inputs away from overflow/underflow, which is all the detector feeds it.
 */

const fround = Math.fround;
const VELTKAMP_SPLITTER = 134217729; // 2^27 + 1: splits a double into two 26-bit halves
const float64Scratch = new Float64Array(1);
const wordScratch = new Uint32Array(float64Scratch.buffer); // [low, high] on little-endian hosts

/** Exact `a * b` as `[product, error]` (Dekker / Veltkamp). */
function exactProduct(a, b) {
  const product = a * b;
  const aScaled = VELTKAMP_SPLITTER * a;
  const aHigh = aScaled - (aScaled - a);
  const aLow = a - aHigh;
  const bScaled = VELTKAMP_SPLITTER * b;
  const bHigh = bScaled - (bScaled - b);
  const bLow = b - bHigh;
  return [product, ((aHigh * bHigh - product) + aHigh * bLow + aLow * bHigh) + aLow * bLow];
}

/** Exact `a + b` as `[sum, error]` (Knuth TwoSum). */
function exactSum(a, b) {
  const sum = a + b;
  const bVirtual = sum - a;
  return [sum, (a - (sum - bVirtual)) + (b - bVirtual)];
}

/** `a + b` rounded to odd: truncate, then force the last mantissa bit to 1 if inexact. */
function addRoundToOdd(a, b) {
  const [sum, error] = exactSum(a, b);
  if (error === 0) return sum;
  float64Scratch[0] = sum;
  if ((wordScratch[0] & 1) === 1) return sum;
  // even and inexact: step one ulp toward the exact value (away from zero iff error has sum's sign)
  const awayFromZero = (error > 0) === (sum > 0);
  if (awayFromZero) {
    wordScratch[0] = (wordScratch[0] + 1) >>> 0;
    if (wordScratch[0] === 0) wordScratch[1] += 1;
  } else {
    if (wordScratch[0] === 0) wordScratch[1] -= 1;
    wordScratch[0] = (wordScratch[0] - 1) >>> 0;
  }
  return float64Scratch[0];
}

/** float64 `round(a * b + c)` with a single rounding. */
export function fusedMultiplyAddFloat64(a, b, c) {
  const [productHigh, productLow] = exactProduct(a, b);
  const [sumHigh, sumLow] = exactSum(c, productHigh);
  return sumHigh + addRoundToOdd(sumLow, productLow);
}

/** float32 `round(a * b + c)` with a single rounding; inputs must already be float32 values. */
export function fusedMultiplyAddFloat32(a, b, c) {
  return fround(addRoundToOdd(a * b, c));
}
