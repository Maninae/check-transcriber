/**
 * `cv2.magnitude(x, y)` for float32 maps, bit-exact with the Python cv2 on arm64.
 *
 * That cv2 build routes `magnitude` to the carotene NEON HAL, which does not call `sqrt`: it
 * computes `1 / rsqrt(x*x + y*y)` from the ARM estimate instructions, each refined by two
 * Newton steps (`FRSQRTE` + 2x `FRSQRTS`, then `FRECPE` + 2x `FRECPS`). The estimates are
 * table lookups fully specified by the ARMv8 pseudocode (`RecipEstimate`,
 * `RecipSqrtEstimate`), reproduced here. The work is split into `parallel_for_` stripes of
 * `len / 65536` (OpenCV's deterministic stripe boundaries); within a stripe, all elements go
 * through the NEON path except the last one of an odd-length stripe, which uses `sqrtf`.
 */

const fround = Math.fround;
const float32Scratch = new Float32Array(1);
const uint32Scratch = new Uint32Array(float32Scratch.buffer);
const STRIPE_ELEMENT_COUNT = 1 << 16;
const NEON_BLOCK = 8;
const NEON_PAIR = 2;

/** Bits of a float32 value. */
function float32Bits(value) {
  float32Scratch[0] = value;
  return uint32Scratch[0];
}

/** float32 value of raw bits. */
function float32FromBits(bits) {
  uint32Scratch[0] = bits >>> 0;
  return float32Scratch[0];
}

/** ARM `RecipEstimate` (8-bit precision), for 9-bit inputs 256..511. */
function recipEstimate(scaled) {
  const rounded = scaled * 2 + 1;
  const quotient = Math.floor((1 << 19) / rounded);
  return (quotient + 1) >> 1;
}

/** ARM `RecipSqrtEstimate` (8-bit precision), for 9-bit inputs 128..511. */
function recipSqrtEstimate(scaled) {
  let input = scaled;
  if (input < 256) {
    input = input * 2 + 1; // 0.25 .. 0.5, in units of 1/512 rounded to nearest
  } else {
    input = ((input >> 1) << 1) + 1; // 0.5 .. 1.0: discard the bottom bit,
    input *= 2; // then units of 1/256 rounded to nearest
  }
  let estimate = 512;
  while (input * (estimate + 1) * (estimate + 1) < 2 ** 28) estimate += 1;
  return (estimate + 1) >> 1;
}

const RECIP_TABLE = Int32Array.from({ length: 512 }, (_, index) => (index >= 256 ? recipEstimate(index) : 0));
const RECIP_SQRT_TABLE = Int32Array.from({ length: 512 }, (_, index) => (index >= 128 ? recipSqrtEstimate(index) : 0));

/** ARM `FRECPE` on a positive finite float32 (the only case magnitude reaches). */
function reciprocalEstimate(value) {
  if (value === 0) return Infinity;
  if (value === Infinity) return 0;
  const bits = float32Bits(value);
  let exponent = (bits >>> 23) & 0xff;
  let fraction = bits & 0x7fffff; // 23 bits, the pseudocode's fraction<51:29>
  if (exponent === 0) {
    if ((fraction & 0x400000) === 0) {
      exponent = -1;
      fraction = (fraction << 2) & 0x7fffff;
    } else {
      fraction = (fraction << 1) & 0x7fffff;
    }
  }
  const scaled = 256 | (fraction >>> 15); // '1' : top 8 fraction bits
  let resultExponent = 253 - exponent;
  let resultFraction = (RECIP_TABLE[scaled] & 0xff) << 15; // estimate<7:0> as the top fraction bits
  if (resultExponent === 0) {
    resultFraction = 0x400000 | (resultFraction >>> 1);
  } else if (resultExponent === -1) {
    resultFraction = 0x200000 | (resultFraction >>> 2);
    resultExponent = 0;
  }
  if (resultExponent >= 255) return Infinity;
  return float32FromBits((resultExponent << 23) | resultFraction);
}

/** ARM `FRSQRTE` on a non-negative finite float32. */
function reciprocalSqrtEstimate(value) {
  if (value === 0) return Infinity;
  if (value === Infinity) return 0;
  const bits = float32Bits(value);
  let exponent = (bits >>> 23) & 0xff;
  let fraction = bits & 0x7fffff;
  if (exponent === 0) {
    while ((fraction & 0x400000) === 0) {
      fraction = (fraction << 1) & 0x7fffff;
      exponent -= 1;
    }
    fraction = (fraction << 1) & 0x7fffff;
  }
  const scaled = (exponent & 1) === 0 ? 256 | (fraction >>> 15) : 128 | (fraction >>> 16);
  const resultExponent = Math.floor((380 - exponent) / 2);
  return float32FromBits((resultExponent << 23) | ((RECIP_SQRT_TABLE[scaled] & 0xff) << 15));
}

/** ARM `FRSQRTS`: (3 - a * b) / 2 with one rounding; (inf, 0) gives 1.5. */
function reciprocalSqrtStep(first, second) {
  if ((first === Infinity && second === 0) || (first === 0 && second === Infinity)) return 1.5;
  return fround((3 - first * second) / 2);
}

/** ARM `FRECPS`: 2 - a * b with one rounding; (inf, 0) gives 2. */
function reciprocalStep(first, second) {
  if ((first === Infinity && second === 0) || (first === 0 && second === Infinity)) return 2;
  return fround(2 - first * second);
}

/** carotene `internal::vsqrtq_f32` of one lane: reciprocal of the refined reciprocal sqrt. */
function neonEstimatedSqrt(value) {
  let inverseRoot = reciprocalSqrtEstimate(value);
  inverseRoot = fround(reciprocalSqrtStep(fround(inverseRoot * inverseRoot), value) * inverseRoot);
  inverseRoot = fround(reciprocalSqrtStep(fround(inverseRoot * inverseRoot), value) * inverseRoot);
  let root = reciprocalEstimate(inverseRoot);
  root = fround(reciprocalStep(inverseRoot, root) * root);
  root = fround(reciprocalStep(inverseRoot, root) * root);
  return root;
}

/**
 * Element-wise magnitude of two float32 maps, as the arm64 cv2 computes it.
 *
 * Returns a new Float32Array.
 */
export function magnitudeFloat32(valuesX, valuesY) {
  const length = valuesX.length;
  const magnitudes = new Float32Array(length);
  const stripeCount = Math.max(1, Math.min(Math.round(length / STRIPE_ELEMENT_COUNT), length));
  const halfStripes = Math.floor(stripeCount / 2);
  for (let stripe = 0; stripe < stripeCount; stripe += 1) {
    const start = stripeCount === 1 ? 0 : Math.floor((stripe * length + halfStripes) / stripeCount);
    const end = stripe + 1 >= stripeCount ? length : Math.floor(((stripe + 1) * length + halfStripes) / stripeCount);
    const stripeLength = end - start;
    let offset = 0;
    while (offset + NEON_BLOCK <= stripeLength) offset += NEON_BLOCK;
    while (offset + NEON_PAIR <= stripeLength) offset += NEON_PAIR;
    const scalarStart = start + offset;
    for (let index = start; index < end; index += 1) {
      const valueX = valuesX[index];
      const valueY = valuesY[index];
      if (index < scalarStart) {
        magnitudes[index] = neonEstimatedSqrt(fround(fround(valueX * valueX) + fround(valueY * valueY)));
      } else {
        magnitudes[index] = fround(Math.sqrt(fround(valueX * valueX + fround(valueY * valueY)))); // sqrtf(fma(x, x, y * y))
      }
    }
  }
  return magnitudes;
}
