/**
 * OpenCV's `cv::RNG` (multiply-with-carry, 64-bit state), seeded `(uint64)-1`.
 *
 * `HoughLinesProbabilistic` and the robust `fitLine` both build one of these per call with the
 * fixed seed, so the random point order they draw is deterministic and must match exactly.
 */

const RNG_MULTIPLIER_HIGH = 4164903690 >>> 16; // cv::RNG coefficient 4164903690, split into 16-bit limbs
const RNG_MULTIPLIER_LOW = 4164903690 & 0xffff;
const TWO_TO_32 = 4294967296;

/** OpenCV `cv::RNG` with `uniform(0, n)`. */
export class OpenCvRandomGenerator {
  /** Seed `(uint64)-1`, the state both OpenCV callers start from. */
  constructor() {
    this.stateLow = 0xffffffff;
    this.stateHigh = 0xffffffff;
  }

  /** `next()`: state = low * 4164903690 + high; returns the new low 32 bits. */
  next() {
    const low = this.stateLow;
    const lowHigh16 = low >>> 16;
    const lowLow16 = low & 0xffff;
    const productLowLow = lowLow16 * RNG_MULTIPLIER_LOW;
    const middle = lowLow16 * RNG_MULTIPLIER_HIGH + lowHigh16 * RNG_MULTIPLIER_LOW;
    const productHighHigh = lowHigh16 * RNG_MULTIPLIER_HIGH;
    const lowSum = productLowLow + (middle % 65536) * 65536 + this.stateHigh;
    const carry = Math.floor(lowSum / TWO_TO_32);
    this.stateLow = lowSum - carry * TWO_TO_32;
    this.stateHigh = (productHighHigh + Math.floor(middle / 65536) + carry) % TWO_TO_32;
    return this.stateLow;
  }

  /** `uniform(0, count)` for count > 0. */
  uniformBelow(count) {
    return this.next() % count;
  }
}
