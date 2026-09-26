/**
 * Seeded integer draws for RANSAC, bit-exact with numpy's `default_rng(seed).integers`.
 *
 * The Python refinement threads ONE `np.random.default_rng(config.random_seed)` through
 * every RANSAC call of a check (all sides, both passes, the corner-local fits), so the
 * hypotheses a JS port samples decide which inlier set wins wherever two lines compete.
 * Reproducing numpy exactly removes that divergence entirely:
 *
 * - `SeedSequence(seed).generate_state(4, uint64)` (numpy/random/bit_generator.pyx),
 * - PCG64 (XSL-RR 128/64: step first, then output),
 * - `integers(0, n)` with int64 dtype = Lemire's bounded method on a buffered uint32 stream
 *   whose buffer is local to each `integers` call (low 32 bits of a uint64 first).
 *
 * The 128-bit LCG uses BigInt: a check needs a few thousand uint64 draws (~0.2 ms).
 * `createMulberry32Generator` is the cheap non-numpy alternative, kept for measurement.
 */

const MASK_32 = 0xffffffffn;
const MASK_64 = (1n << 64n) - 1n;
const MASK_128 = (1n << 128n) - 1n;
const PCG_DEFAULT_MULTIPLIER_128 = 0x2360ed051fc65da44385df649fccf645n;
const SEED_SEQUENCE_POOL_SIZE = 4;
const SEED_SEQUENCE_INIT_A = 0x43b0d7e5;
const SEED_SEQUENCE_MULT_A = 0x931e8875;
const SEED_SEQUENCE_INIT_B = 0x8b51f9dd;
const SEED_SEQUENCE_MULT_B = 0x58f38ded;
const SEED_SEQUENCE_MIX_MULT_L = 0xca01f9dd;
const SEED_SEQUENCE_MIX_MULT_R = 0x4973f715;
const SEED_SEQUENCE_XSHIFT = 16;
// Lemire's product u32 * n must stay exact in a double.
const MAXIMUM_EXACT_BOUND = 2 ** 20;

/** Split a non-negative integer seed into little-endian uint32 words (numpy `_int_to_uint32_array`). */
function seedToUint32Words(seed) {
  if (!Number.isInteger(seed) || seed < 0) throw new Error(`random seed must be a non-negative integer, got ${seed}`);
  let remaining = BigInt(seed);
  if (remaining === 0n) return [0];
  const words = [];
  while (remaining > 0n) {
    words.push(Number(remaining & MASK_32));
    remaining >>= 32n;
  }
  return words;
}

/** numpy SeedSequence(seed).generate_state(n, uint32) for an int seed (no spawn key). */
export function generateSeedSequenceWords(seed, numberOfWords) {
  const entropyWords = seedToUint32Words(seed);
  let hashConstant = SEED_SEQUENCE_INIT_A;
  const hashMix = (value) => {
    let mixed = (value ^ hashConstant) >>> 0;
    hashConstant = Math.imul(hashConstant, SEED_SEQUENCE_MULT_A) >>> 0;
    mixed = Math.imul(mixed, hashConstant) >>> 0;
    return (mixed ^ (mixed >>> SEED_SEQUENCE_XSHIFT)) >>> 0;
  };
  const mix = (first, second) => {
    const result = (Math.imul(SEED_SEQUENCE_MIX_MULT_L, first) - Math.imul(SEED_SEQUENCE_MIX_MULT_R, second)) >>> 0;
    return (result ^ (result >>> SEED_SEQUENCE_XSHIFT)) >>> 0;
  };
  const pool = new Array(SEED_SEQUENCE_POOL_SIZE);
  for (let index = 0; index < SEED_SEQUENCE_POOL_SIZE; index += 1) {
    pool[index] = hashMix(index < entropyWords.length ? entropyWords[index] : 0);
  }
  for (let sourceIndex = 0; sourceIndex < SEED_SEQUENCE_POOL_SIZE; sourceIndex += 1) {
    for (let destinationIndex = 0; destinationIndex < SEED_SEQUENCE_POOL_SIZE; destinationIndex += 1) {
      if (sourceIndex !== destinationIndex) pool[destinationIndex] = mix(pool[destinationIndex], hashMix(pool[sourceIndex]));
    }
  }
  for (let sourceIndex = SEED_SEQUENCE_POOL_SIZE; sourceIndex < entropyWords.length; sourceIndex += 1) {
    for (let destinationIndex = 0; destinationIndex < SEED_SEQUENCE_POOL_SIZE; destinationIndex += 1) {
      pool[destinationIndex] = mix(pool[destinationIndex], hashMix(entropyWords[sourceIndex]));
    }
  }
  let outputHashConstant = SEED_SEQUENCE_INIT_B;
  const words = [];
  for (let index = 0; index < numberOfWords; index += 1) {
    let value = (pool[index % SEED_SEQUENCE_POOL_SIZE] ^ outputHashConstant) >>> 0;
    outputHashConstant = Math.imul(outputHashConstant, SEED_SEQUENCE_MULT_B) >>> 0;
    value = Math.imul(value, outputHashConstant) >>> 0;
    words.push((value ^ (value >>> SEED_SEQUENCE_XSHIFT)) >>> 0);
  }
  return words;
}

/** numpy's PCG64 bit generator; `nextUint64()` returns a BigInt. */
export class NumpyPcg64 {
  /** Seed exactly like `np.random.PCG64(seed)` (via SeedSequence). */
  constructor(seed) {
    const words = generateSeedSequenceWords(seed, 8).map(BigInt);
    const seedWords = [0, 1, 2, 3].map((index) => words[2 * index] | (words[2 * index + 1] << 32n));
    const initialState = (seedWords[0] << 64n) | seedWords[1];
    const initialSequence = (seedWords[2] << 64n) | seedWords[3];
    this.increment = ((initialSequence << 1n) | 1n) & MASK_128;
    this.state = 0n;
    this.step();
    this.state = (this.state + initialState) & MASK_128;
    this.step();
  }

  /** Advance the 128-bit LCG. */
  step() {
    this.state = (this.state * PCG_DEFAULT_MULTIPLIER_128 + this.increment) & MASK_128;
  }

  /** Next 64-bit output (XSL-RR of the NEW state), as a BigInt. */
  nextUint64() {
    this.step();
    const xored = ((this.state >> 64n) ^ this.state) & MASK_64;
    const rotation = this.state >> 122n;
    return ((xored >> rotation) | (xored << ((64n - rotation) & 63n))) & MASK_64;
  }
}

/** `integers(0, bound, size=count)` draws, row-major, numpy semantics; `generator.randomSeed` for logs. */
export function createNumpyRandomGenerator(seed) {
  const bitGenerator = new NumpyPcg64(seed);
  return {
    kind: "numpy_pcg64",
    /** Int32Array of `count` draws uniform in [0, upperExclusive). */
    drawIntegersBelow(upperExclusive, count) {
      const draws = new Int32Array(count);
      const rangeInclusive = upperExclusive - 1;
      if (rangeInclusive === 0) return draws; // numpy consumes no bits for a single value
      if (!(upperExclusive > 1 && upperExclusive <= MAXIMUM_EXACT_BOUND)) throw new Error(`unsupported RANSAC bound ${upperExclusive}`);
      let buffer = 0n;
      let bufferHasHighHalf = false;
      const nextUint32 = () => {
        if (bufferHasHighHalf) {
          bufferHasHighHalf = false;
          return Number(buffer >> 32n);
        }
        buffer = bitGenerator.nextUint64();
        bufferHasHighHalf = true;
        return Number(buffer & MASK_32);
      };
      const threshold = (2 ** 32 - upperExclusive) % upperExclusive;
      for (let index = 0; index < count; index += 1) {
        let product = nextUint32() * upperExclusive;
        let leftover = product % 2 ** 32;
        if (leftover < upperExclusive) {
          while (leftover < threshold) {
            product = nextUint32() * upperExclusive;
            leftover = product % 2 ** 32;
          }
        }
        draws[index] = Math.floor(product / 2 ** 32);
      }
      return draws;
    },
  };
}

/** Cheap seeded alternative (mulberry32); NOT numpy's stream, kept to measure RNG sensitivity. */
export function createMulberry32Generator(seed) {
  let state = seed >>> 0;
  const nextFloat = () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let mixed = Math.imul(state ^ (state >>> 15), 1 | state);
    mixed = (mixed + Math.imul(mixed ^ (mixed >>> 7), 61 | mixed)) ^ mixed;
    return ((mixed ^ (mixed >>> 14)) >>> 0) / 2 ** 32;
  };
  return {
    kind: "mulberry32",
    /** Int32Array of `count` draws uniform in [0, upperExclusive). */
    drawIntegersBelow(upperExclusive, count) {
      const draws = new Int32Array(count);
      for (let index = 0; index < count; index += 1) draws[index] = Math.floor(nextFloat() * upperExclusive);
      return draws;
    },
  };
}

/**
 * The RANSAC generator for one check: numpy-exact PCG64 seeded with `config.randomSeed`.
 *
 * `config.randomGeneratorKind` (absent from the Python config) is a measurement hook only:
 * "mulberry32" swaps in the cheap generator to quantify how much the draws move corners.
 */
export function createRandomGeneratorForConfig(config) {
  const kind = config.randomGeneratorKind ?? "numpy_pcg64";
  if (kind === "numpy_pcg64") return createNumpyRandomGenerator(config.randomSeed);
  if (kind === "mulberry32") return createMulberry32Generator(config.randomSeed);
  throw new Error(`unknown randomGeneratorKind ${JSON.stringify(kind)}`);
}
