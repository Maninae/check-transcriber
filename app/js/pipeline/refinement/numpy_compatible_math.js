/**
 * Small numeric helpers that reproduce numpy's exact semantics where the port depends on them.
 *
 * Each one mirrors a numpy behaviour a naive JS version gets subtly wrong:
 * - `linspace`: i * step + start, last element forced to `stop`.
 * - `floatArange`: numpy fills a[i] = start + i * (a[1] - a[0]), not start + i * step.
 * - `roundHalfToEven`: np.round; Math.round rounds halves up.
 * - `median`: mean of the two middle values for even counts.
 * - `argmax` / `argmin`: FIRST extreme index; argmax of an all -inf row is 0.
 * - Radian/degree conversion multiplies by numpy's constant (x * (pi / 180)).
 */

const DEGREES_TO_RADIANS = Math.PI / 180.0;
const RADIANS_TO_DEGREES = 180.0 / Math.PI;

/** np.linspace(start, stop, count) as a Float64Array. */
export function linspace(start, stop, count) {
  const values = new Float64Array(count);
  if (count === 1) {
    values[0] = start;
    return values;
  }
  const step = (stop - start) / (count - 1);
  for (let index = 0; index < count; index += 1) values[index] = index * step + start;
  values[count - 1] = stop;
  return values;
}

/** np.arange(start, stop, step) for floats, with numpy's fill rule. */
export function floatArange(start, stop, step) {
  const count = Math.max(0, Math.ceil((stop - start) / step));
  const values = new Float64Array(count);
  if (count === 0) return values;
  values[0] = start;
  if (count === 1) return values;
  values[1] = start + step;
  const delta = values[1] - values[0];
  for (let index = 2; index < count; index += 1) values[index] = start + index * delta;
  return values;
}

/** np.round (banker's rounding to the nearest integer). */
export function roundHalfToEven(value) {
  const floored = Math.floor(value);
  const difference = value - floored;
  if (difference > 0.5) return floored + 1;
  if (difference < 0.5) return floored;
  return floored % 2 === 0 ? floored : floored + 1;
}

/** np.clip for a scalar. */
export function clip(value, lower, upper) {
  return Math.min(Math.max(value, lower), upper);
}

/** np.deg2rad. */
export function degreesToRadians(degrees) {
  return degrees * DEGREES_TO_RADIANS;
}

/** np.degrees. */
export function radiansToDegrees(radians) {
  return radians * RADIANS_TO_DEGREES;
}

/** np.median of a numeric array-like (copied, NaN-free input assumed). */
export function median(values) {
  const count = values.length;
  if (count === 0) return NaN;
  const sorted = Float64Array.from(values).sort();
  const middle = count >> 1;
  return count % 2 === 1 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

/** np.argmax over values[offset .. offset + length) (first maximum). */
export function argmax(values, offset = 0, length = values.length - offset) {
  let bestIndex = 0;
  let bestValue = values[offset];
  for (let index = 1; index < length; index += 1) {
    const value = values[offset + index];
    if (value > bestValue || (Number.isNaN(value) && !Number.isNaN(bestValue))) {
      bestValue = value;
      bestIndex = index;
    }
  }
  return bestIndex;
}

/** np.argmin of |values| (first minimum absolute value). */
export function argminAbsolute(values) {
  let bestIndex = 0;
  for (let index = 1; index < values.length; index += 1) {
    if (Math.abs(values[index]) < Math.abs(values[bestIndex])) bestIndex = index;
  }
  return bestIndex;
}

const VELTKAMP_SPLITTER = 134217729; // 2^27 + 1

/** Exact x * x as a double-double [high, low] (Dekker's product via Veltkamp splitting). */
function exactSquare(value) {
  const high = value * value;
  const scaled = VELTKAMP_SPLITTER * value;
  const upper = scaled - (scaled - value);
  const lower = value - upper;
  const low = ((upper * upper - high) + 2 * upper * lower) + lower * lower;
  return [high, low];
}

/** Error-free a + b as [sum, error] (Knuth's TwoSum). */
function twoSum(first, second) {
  const sum = first + second;
  const secondVirtual = sum - first;
  return [sum, (first - (sum - secondVirtual)) + (second - secondVirtual)];
}

/**
 * np.hypot exactly as numpy computes it on this platform: sqrt(fma(small, small, big^2)),
 * measured bit-exact on 20k random pairs. V8's Math.hypot differs in ~40% of cases and a
 * plain sqrt(x*x + y*y) in ~6%; side lengths feed every sample position.
 */
export function hypot(x, y) {
  const big = Math.max(Math.abs(x), Math.abs(y));
  const small = Math.min(Math.abs(x), Math.abs(y));
  const [smallSquaredHigh, smallSquaredLow] = exactSquare(small);
  const [sum, sumError] = twoSum(smallSquaredHigh, big * big);
  return Math.sqrt(sum + (sumError + smallSquaredLow));
}

/** np.linalg.norm of a 2-vector: sqrt(x * x + y * y), NOT libm hypot. */
export function vectorNorm(x, y) {
  return Math.sqrt(x * x + y * y);
}
