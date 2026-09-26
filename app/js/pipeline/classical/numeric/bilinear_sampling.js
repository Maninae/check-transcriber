/**
 * Bilinear sampling of a float32 map at arbitrary points, bit-exact with the Python cv2.
 *
 * The Python detector samples maps with `cv2.remap(map, map_x, map_y, INTER_LINEAR,
 * BORDER_REPLICATE)` on float32 point coordinates. OpenCV 5 computes that as two float32
 * lerps (`v0 = p00 + a * (p01 - p00)`, `v1` likewise, then `v0 + b * (v1 - v0)`), and on
 * arm64 NEON each lerp is a fused multiply-add. OpenCV.js (WASM, no FMA) rounds the
 * product separately and differs by up to ~3e-5, so this pure-JS sampler emulates the
 * fused form instead: the float32 product is exact in float64, so one float64 add then
 * `Math.fround` reproduces the FMA (verified: 0 of 2000 random samples differ from cv2).
 * It also avoids allocating three Mats per sampling call.
 */

const fround = Math.fround;

/** float32 fused multiply-add emulation: round(multiplier * multiplicand + addend) once. */
function fusedMultiplyAddFloat32(multiplier, multiplicand, addend) {
  return fround(multiplier * multiplicand + addend);
}

/**
 * Sample `mapValues` (row-major float32, `width` x `height`) at one point.
 *
 * Args:
 *   mapValues: Float32Array of the map.
 *   width, height: map size in pixels.
 *   pointX, pointY: coordinates (cast to float32 first, as the Python casts its maps).
 * Returns:
 *   The float32 bilinear value, with coordinates clamped to the edge (BORDER_REPLICATE).
 */
export function sampleBilinearReplicate(mapValues, width, height, pointX, pointY) {
  const sampleX = fround(pointX);
  const sampleY = fround(pointY);
  const leftColumn = Math.floor(sampleX);
  const topRow = Math.floor(sampleY);
  const alpha = fround(sampleX - leftColumn);
  const beta = fround(sampleY - topRow);
  if (leftColumn >= 0 && topRow >= 0 && leftColumn + 1 < width && topRow + 1 < height) {
    // interior fast path: all four neighbours in frame, no clamping
    const topLeftIndex = topRow * width + leftColumn;
    const topLeftValue = mapValues[topLeftIndex];
    const bottomLeftValue = mapValues[topLeftIndex + width];
    const topInterpolated = fusedMultiplyAddFloat32(alpha, fround(mapValues[topLeftIndex + 1] - topLeftValue), topLeftValue);
    const bottomInterpolated = fusedMultiplyAddFloat32(alpha, fround(mapValues[topLeftIndex + width + 1] - bottomLeftValue), bottomLeftValue);
    return fusedMultiplyAddFloat32(beta, fround(bottomInterpolated - topInterpolated), topInterpolated);
  }
  const column0 = leftColumn < 0 ? 0 : leftColumn >= width ? width - 1 : leftColumn;
  const column1 = leftColumn + 1 < 0 ? 0 : leftColumn + 1 >= width ? width - 1 : leftColumn + 1;
  const row0 = topRow < 0 ? 0 : topRow >= height ? height - 1 : topRow;
  const row1 = topRow + 1 < 0 ? 0 : topRow + 1 >= height ? height - 1 : topRow + 1;
  const topLeft = mapValues[row0 * width + column0];
  const topRight = mapValues[row0 * width + column1];
  const bottomLeft = mapValues[row1 * width + column0];
  const bottomRight = mapValues[row1 * width + column1];
  const topValue = fusedMultiplyAddFloat32(alpha, fround(topRight - topLeft), topLeft);
  const bottomValue = fusedMultiplyAddFloat32(alpha, fround(bottomRight - bottomLeft), bottomLeft);
  return fusedMultiplyAddFloat32(beta, fround(bottomValue - topValue), topValue);
}

/** A sampler bound to one map: `sample(x, y)`. Maps are `{ values, width, height }`. */
export function createMapSampler(signalMap) {
  const { values, width, height } = signalMap;
  return (pointX, pointY) => sampleBilinearReplicate(values, width, height, pointX, pointY);
}
