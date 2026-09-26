/**
 * INTER_AREA downscale of a uint8 image, bit-exact with the Python cv2 (OpenCV 5.0, arm64).
 *
 * The Python detector's first step is `cv2.resize(image, None, fx=s, fy=s, INTER_AREA)`. Its
 * generic (non-integer ratio) path accumulates float32 source * weight products per output
 * row (`buf += S * alpha`, scalar C++ that clang on arm64 contracts into a fused multiply-add)
 * and then blends rows (`sum += beta * buf`: an unfused NEON multiply then add for full
 * 4-lane blocks, a fused scalar tail for the last 1-4 values). OpenCV.js (WASM) rounds every
 * product separately, so ~1 in 300k output bytes land on the other side of .5 and differ by 1,
 * which is enough to reshuffle the randomized HoughLinesP downstream. This port reproduces the
 * arm64 arithmetic (OpenCV `computeResizeAreaTab` + `ResizeArea_Invoker`), including which
 * elements are fused. Integer ratios (OpenCV's separate "area fast" path) are delegated to
 * `cv.resize`.
 */

const fround = Math.fround;
const NEON_FLOAT32_LANES = 4;
const AREA_EPSILON = 1e-3;
const DOUBLE_EPSILON = 2.220446049250313e-16;

/** OpenCV `cvRound` (round half to even) of a float. */
function roundHalfEvenToInteger(value) {
  const floorValue = Math.floor(value);
  const fraction = value - floorValue;
  if (fraction > 0.5) return floorValue + 1;
  if (fraction < 0.5) return floorValue;
  return floorValue % 2 === 0 ? floorValue : floorValue + 1;
}

/** OpenCV `computeResizeAreaTab`: (destination index, source index, float32 weight) triples. */
function computeResizeAreaTable(sourceSize, destinationSize, channelCount, scale) {
  const destinationIndices = [];
  const sourceIndices = [];
  const weights = [];
  const push = (destinationIndex, sourceIndex, weight) => {
    destinationIndices.push(destinationIndex);
    sourceIndices.push(sourceIndex);
    weights.push(fround(weight));
  };
  for (let destination = 0; destination < destinationSize; destination += 1) {
    const cellStart = destination * scale;
    const cellEnd = cellStart + scale;
    const cellWidth = Math.min(scale, sourceSize - cellStart);
    let firstWhole = Math.ceil(cellStart);
    let lastWhole = Math.floor(cellEnd);
    lastWhole = Math.min(lastWhole, sourceSize - 1);
    firstWhole = Math.min(firstWhole, lastWhole);
    if (firstWhole - cellStart > AREA_EPSILON) push(destination * channelCount, (firstWhole - 1) * channelCount, (firstWhole - cellStart) / cellWidth);
    for (let source = firstWhole; source < lastWhole; source += 1) push(destination * channelCount, source * channelCount, 1.0 / cellWidth);
    if (cellEnd - lastWhole > AREA_EPSILON) {
      push(destination * channelCount, lastWhole * channelCount, Math.min(Math.min(cellEnd - lastWhole, 1.0), cellWidth) / cellWidth);
    }
  }
  return { destinationIndices: Int32Array.from(destinationIndices), sourceIndices: Int32Array.from(sourceIndices), weights: Float32Array.from(weights) };
}

/** Store the float row sums as saturated, half-even-rounded bytes. */
function storeRow(rowSums, destinationBytes, rowOffset) {
  for (let index = 0; index < rowSums.length; index += 1) {
    const value = rowSums[index];
    let rounded = Math.round(value);
    if (rounded - value === 0.5 && (rounded & 1) !== 0) rounded -= 1; // half to even
    destinationBytes[rowOffset + index] = rounded < 0 ? 0 : rounded > 255 ? 255 : rounded;
  }
}

/**
 * Downscale `sourceBytes` (row-major, `channelCount` interleaved uint8) by `inverseScale` (< 1).
 *
 * Returns `{ bytes, width, height }`, or null when OpenCV would take its integer-ratio fast
 * path (the caller then uses `cv.resize`).
 */
export function resizeAreaUint8(sourceBytes, sourceWidth, sourceHeight, channelCount, inverseScale) {
  const destinationWidth = roundHalfEvenToInteger(sourceWidth * inverseScale);
  const destinationHeight = roundHalfEvenToInteger(sourceHeight * inverseScale);
  const scale = 1.0 / inverseScale;
  const integerScale = roundHalfEvenToInteger(scale);
  if (Math.abs(scale - integerScale) < DOUBLE_EPSILON) return null;
  const columnTable = computeResizeAreaTable(sourceWidth, destinationWidth, channelCount, scale);
  const rowTable = computeResizeAreaTable(sourceHeight, destinationHeight, 1, scale);
  const rowWidth = destinationWidth * channelCount;
  let vectorEnd = 0;
  while (vectorEnd + NEON_FLOAT32_LANES < rowWidth) vectorEnd += NEON_FLOAT32_LANES;
  const destinationBytes = new Uint8Array(rowWidth * destinationHeight);
  const rowBuffer = new Float32Array(rowWidth);
  const rowSums = new Float32Array(rowWidth);
  const { destinationIndices, sourceIndices, weights } = columnTable;
  const entryCount = weights.length;
  let previousRow = rowTable.destinationIndices[0];
  for (let entry = 0; entry < rowTable.weights.length; entry += 1) {
    const beta = rowTable.weights[entry];
    const destinationRow = rowTable.destinationIndices[entry];
    const sourceRowOffset = rowTable.sourceIndices[entry] * sourceWidth * channelCount;
    rowBuffer.fill(0);
    if (channelCount === 3) {
      for (let column = 0; column < entryCount; column += 1) {
        const destinationIndex = destinationIndices[column];
        const sourceIndex = sourceRowOffset + sourceIndices[column];
        const alpha = weights[column];
        // fused: buf + S * alpha rounds once (the product is exact in float64)
        rowBuffer[destinationIndex] = fround(rowBuffer[destinationIndex] + sourceBytes[sourceIndex] * alpha);
        rowBuffer[destinationIndex + 1] = fround(rowBuffer[destinationIndex + 1] + sourceBytes[sourceIndex + 1] * alpha);
        rowBuffer[destinationIndex + 2] = fround(rowBuffer[destinationIndex + 2] + sourceBytes[sourceIndex + 2] * alpha);
      }
    } else {
      for (let column = 0; column < entryCount; column += 1) {
        const destinationIndex = destinationIndices[column];
        const sourceIndex = sourceRowOffset + sourceIndices[column];
        const alpha = weights[column];
        for (let channel = 0; channel < channelCount; channel += 1) {
          rowBuffer[destinationIndex + channel] = fround(rowBuffer[destinationIndex + channel] + sourceBytes[sourceIndex + channel] * alpha);
        }
      }
    }
    if (destinationRow !== previousRow) {
      storeRow(rowSums, destinationBytes, previousRow * rowWidth);
      for (let index = 0; index < rowWidth; index += 1) rowSums[index] = fround(beta * rowBuffer[index]);
      previousRow = destinationRow;
    } else {
      for (let index = 0; index < vectorEnd; index += 1) rowSums[index] = fround(rowSums[index] + fround(beta * rowBuffer[index]));
      for (let index = vectorEnd; index < rowWidth; index += 1) rowSums[index] = fround(rowSums[index] + beta * rowBuffer[index]);
    }
  }
  storeRow(rowSums, destinationBytes, previousRow * rowWidth);
  return { bytes: destinationBytes, width: destinationWidth, height: destinationHeight };
}
