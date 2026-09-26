/**
 * numpy / Python semantics the detector relies on, reproduced exactly in plain JS.
 *
 * The parity target is the Python detector on arm64 with numpy 2.x:
 * - `roundHalfEven`: Python `round()` / `np.rint` (banker's rounding), not `Math.round`.
 * - `linspace`: `np.linspace` (i * step + start, last element pinned to stop).
 * - `formatGeneral(Signed)`: Python `f"{x:g}"` / `f"{x:+g}"` for mask names.
 * Sorting lives in `numpy_argsort.js`, order statistics in `numpy_statistics.js`; both are
 * re-exported here so callers import numpy semantics from one place.
 */

export { argmaxFirst, argminFirst, numpyArgsort } from "./numpy_argsort.js";
export {
  kthSmallestFloat32, medianOfFloat32, medianOfFloat32InPlace, medianOfFloat64, medianPassesThreshold,
  pairwiseSum, percentileLinear,
} from "./numpy_statistics.js";

/** Python `round(x)` / `np.rint(x)`: nearest integer, exact halves to even. */
export function roundHalfEven(value) {
  const floorValue = Math.floor(value);
  const fraction = value - floorValue;
  if (fraction > 0.5) return floorValue + 1;
  if (fraction < 0.5) return floorValue;
  return floorValue % 2 === 0 ? floorValue : floorValue + 1;
}

/** `np.linspace(start, stop, count)` as a Float64Array (i * step + start, last pinned to stop). */
export function linspace(start, stop, count) {
  const result = new Float64Array(count);
  if (count === 1) {
    result[0] = start;
    return result;
  }
  const step = (stop - start) / (count - 1);
  for (let index = 0; index < count; index += 1) result[index] = index * step + start;
  result[count - 1] = stop;
  return result;
}

/** Python `f"{value:g}"` for the simple config numbers used in mask names. */
export function formatGeneral(value) {
  return String(Number(value.toPrecision(6)));
}

/** Python `f"{value:+g}"`. */
export function formatGeneralSigned(value) {
  const text = formatGeneral(value);
  return value >= 0 && !Object.is(value, -0) ? `+${text}` : text;
}
