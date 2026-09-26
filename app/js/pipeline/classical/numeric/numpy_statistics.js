/**
 * numpy order statistics and reductions with numpy's exact arithmetic: `np.median` (float32
 * midpoint mean), `np.percentile` (linear, float64 lerp), float64 pairwise summation, and an
 * exact "median passes threshold" test that usually avoids computing the median at all.
 */

const fround = Math.fround;

/** k-th smallest value of `values` (reorders it in place): Hoare quickselect. */
function selectKthSmallest(values, rank) {
  let low = 0;
  let high = values.length - 1;
  while (high > low) {
    const pivot = values[(low + high) >> 1];
    let left = low;
    let right = high;
    while (left <= right) {
      while (values[left] < pivot) left += 1;
      while (values[right] > pivot) right -= 1;
      if (left <= right) {
        const saved = values[left];
        values[left] = values[right];
        values[right] = saved;
        left += 1;
        right -= 1;
      }
    }
    if (rank <= right) high = right;
    else if (rank >= left) low = left;
    else return values[rank];
  }
  return values[rank];
}

/** Smallest value at or above `rank` after `selectKthSmallest(values, rank - 1)` partitioned it. */
function minimumFrom(values, startIndex) {
  let minimum = values[startIndex];
  for (let index = startIndex + 1; index < values.length; index += 1) {
    if (values[index] < minimum) minimum = values[index];
  }
  return minimum;
}

/** The two order statistics at `lowerRank` and `lowerRank + 1` of a scratch copy. */
function adjacentOrderStatistics(scratch, lowerRank) {
  const lowerValue = selectKthSmallest(scratch, lowerRank);
  const upperValue = lowerRank + 1 < scratch.length ? minimumFrom(scratch, lowerRank + 1) : lowerValue;
  return [lowerValue, upperValue];
}

/** `np.median` of float32 values: float32 mean of the two middle values for even counts. */
export function medianOfFloat32(values) {
  return medianOfFloat32InPlace(Float32Array.from(values));
}

/** `medianOfFloat32` that reorders its (scratch) Float32Array argument instead of copying it. */
export function medianOfFloat32InPlace(scratch) {
  const count = scratch.length;
  const middle = count >> 1;
  if (count % 2 === 1) return selectKthSmallest(scratch, middle);
  const [lowerValue, upperValue] = adjacentOrderStatistics(scratch, middle - 1);
  return fround(fround(lowerValue + upperValue) / 2);
}

/** Distance from a float32 value to the next float32 away from zero (one ulp). */
function float32Ulp(value) {
  const magnitude = Math.abs(value);
  if (magnitude === 0) return 1.401298464324817e-45;
  const exponent = Math.floor(Math.log2(magnitude));
  return 2 ** (Math.max(exponent, -126) - 23);
}

/**
 * Whether `np.median(values) <= threshold` for float32 values (or `>=` when `atLeast`),
 * decided exactly, usually from one counting pass instead of a selection.
 *
 * The median never has to be computed unless the count lands on the ambiguous boundary:
 * - odd count: the median is an element, so the count of values on the passing side decides;
 * - even count, n/2 + 1 or more values pass: both middle values pass, so their float32 mean does;
 * - even count, the smallest failing value is 2+ ulps past the threshold and at most n/2 - 1
 *   values pass: both middles fail by more than rounding can undo;
 * otherwise the exact median is computed. `atLeast` negates values and threshold (exact).
 */
export function medianPassesThreshold(values, threshold, atLeast = false) {
  const sign = atLeast ? -1 : 1;
  const bound = sign * threshold;
  const count = values.length;
  let passingCount = 0;
  let smallestFailing = Infinity;
  for (let index = 0; index < count; index += 1) {
    const value = sign * values[index];
    if (value <= bound) passingCount += 1;
    else if (value < smallestFailing) smallestFailing = value;
  }
  if (count % 2 === 1) return passingCount >= (count + 1) / 2;
  if (passingCount >= count / 2 + 1) return true;
  if (passingCount <= count / 2 - 1 && smallestFailing >= bound + 2 * float32Ulp(bound)) return false;
  const median = medianOfFloat32(values);
  return atLeast ? median >= threshold : median <= threshold;
}

/** `np.median` of float64 values. */
export function medianOfFloat64(values) {
  const scratch = Float64Array.from(values);
  const count = scratch.length;
  const middle = count >> 1;
  if (count % 2 === 1) return selectKthSmallest(scratch, middle);
  const [lowerValue, upperValue] = adjacentOrderStatistics(scratch, middle - 1);
  return (lowerValue + upperValue) / 2;
}

/** Order-preserving uint32 key of a float32 (negative values flipped): key order == value order. */
function sortableFloat32Keys(values) {
  const keys = new Uint32Array(Float32Array.from(values).buffer);
  for (let index = 0; index < keys.length; index += 1) {
    const bits = keys[index];
    keys[index] = bits & 0x80000000 ? ~bits >>> 0 : (bits | 0x80000000) >>> 0;
  }
  return keys;
}

/** float32 value of a sortable key. */
function valueOfSortableKey(key) {
  const bits = key & 0x80000000 ? (key & 0x7fffffff) >>> 0 : ~key >>> 0;
  return new Float32Array(Uint32Array.of(bits).buffer)[0];
}

/**
 * The `rank`-th smallest float32 in `values` (0-based), by two 16-bit radix passes.
 *
 * O(n) with two histogram passes, instead of copying and partitioning ~2M floats.
 */
export function kthSmallestFloat32(values, rank) {
  const keys = sortableFloat32Keys(values);
  const highCounts = new Uint32Array(65536);
  for (let index = 0; index < keys.length; index += 1) highCounts[keys[index] >>> 16] += 1;
  let highBucket = 0;
  let remainingRank = rank;
  while (remainingRank >= highCounts[highBucket]) {
    remainingRank -= highCounts[highBucket];
    highBucket += 1;
  }
  const lowCounts = new Uint32Array(65536);
  for (let index = 0; index < keys.length; index += 1) {
    if (keys[index] >>> 16 === highBucket) lowCounts[keys[index] & 0xffff] += 1;
  }
  let lowBucket = 0;
  while (remainingRank >= lowCounts[lowBucket]) {
    remainingRank -= lowCounts[lowBucket];
    lowBucket += 1;
  }
  return valueOfSortableKey(((highBucket << 16) | lowBucket) >>> 0);
}

/**
 * `np.percentile(values, percent)` with the default linear method, on float32 input.
 *
 * numpy's `_lerp` runs in float64 (the gamma array is float64) and switches to the
 * "from the top" form for gamma >= 0.5, which is reproduced here.
 */
export function percentileLinear(values, percent) {
  const quantile = percent / 100;
  const count = values.length;
  const virtualIndex = count * quantile + (1 - quantile) - 1;
  const lowerRank = Math.max(0, Math.min(count - 1, Math.floor(virtualIndex)));
  const upperRank = Math.min(count - 1, lowerRank + 1);
  const gamma = virtualIndex - Math.floor(virtualIndex);
  const lowerValue = kthSmallestFloat32(values, lowerRank);
  const upperValue = upperRank === lowerRank ? lowerValue : kthSmallestFloat32(values, upperRank);
  const difference = fround(upperValue - lowerValue);
  if (gamma >= 0.5) return upperValue - difference * (1 - gamma);
  return lowerValue + difference * gamma;
}

/** numpy's float64 pairwise summation order (used by `.sum()` and `np.mean`). */
export function pairwiseSum(values, start = 0, count = values.length) {
  if (count < 8) {
    let total = 0;
    for (let index = start; index < start + count; index += 1) total += values[index];
    return total;
  }
  if (count <= 128) {
    const partial = [0, 0, 0, 0, 0, 0, 0, 0];
    for (let lane = 0; lane < 8; lane += 1) partial[lane] = values[start + lane];
    let index = 8;
    for (; index < count - (count % 8); index += 8) {
      for (let lane = 0; lane < 8; lane += 1) partial[lane] += values[start + index + lane];
    }
    let total = ((partial[0] + partial[1]) + (partial[2] + partial[3])) + ((partial[4] + partial[5]) + (partial[6] + partial[7]));
    for (; index < count; index += 1) total += values[start + index];
    return total;
  }
  let half = count >> 1;
  half -= half % 8;
  return pairwiseSum(values, start, half) + pairwiseSum(values, start + half, count - half);
}
