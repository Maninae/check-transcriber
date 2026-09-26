/**
 * Exact morphological opening / closing of a binary (0 / nonzero) uint8 mask, bit-packed.
 *
 * `cv.morphologyEx(mask, MORPH_OPEN, ellipse)` visits every kernel point per pixel (~100 for
 * an 11x11 ellipse), over a second per photo in WASM across the candidate masks. For a binary
 * mask and a kernel whose rows are runs centered on the anchor (OpenCV's ellipse, rect and
 * cross all are), the operation separates per kernel row:
 * - horizontal: each row is packed 32 pixels per word and eroded (AND) / dilated (OR) over a
 *   window of +-halfWidth with log-step shifted combines;
 * - vertical: output row y combines kernel row i's horizontal result of source row y + i - a.
 * Borders follow OpenCV's morphology defaults: out-of-image pixels count as on for erode
 * and off for dilate. Output equals OpenCV's pixel for pixel (0 / 255).
 */

import { ALL_ONES, WORD_BITS, packRows, shiftTowardHigher, shiftTowardLower, unpackRows } from "./bit_packed_rows.js";

export { orBitmapInto } from "./bit_packed_rows.js";

/**
 * Half-width of each kernel row, from an OpenCV structuring element (CV_8U Mat).
 *
 * Returns `{ halfWidths: Int32Array, anchorRow }`; throws when a row is not a run centered on
 * the anchor column (such kernels need the generic OpenCV path).
 */
export function kernelRowHalfWidths(kernelMat) {
  const kernelRows = kernelMat.rows;
  const kernelColumns = kernelMat.cols;
  const anchorColumn = kernelColumns >> 1;
  const kernelBytes = kernelMat.data;
  const halfWidths = new Int32Array(kernelRows);
  for (let row = 0; row < kernelRows; row += 1) {
    let first = -1;
    let last = -1;
    for (let column = 0; column < kernelColumns; column += 1) {
      if (!kernelBytes[row * kernelColumns + column]) continue;
      if (first < 0) first = column;
      else if (column !== last + 1) throw new Error("kernelRowHalfWidths: kernel row is not one contiguous run");
      last = column;
    }
    if (first < 0 || anchorColumn - first !== last - anchorColumn) throw new Error("kernelRowHalfWidths: kernel row is not centered on the anchor");
    halfWidths[row] = last - anchorColumn;
  }
  return { halfWidths, anchorRow: kernelRows >> 1 };
}

/**
 * Horizontal window combine of one packed row: result(x) = op over row(x - h .. x + h).
 *
 * `isErosion` picks AND (pad on) or OR (pad off). The row is first shifted by h, so positions
 * left of the image read the pad; then doubling (`power` covers 2^k pixels from x) chains
 * pieces until the window length 2h + 1 is covered.
 */
function combineRowWindow(packed, rowStart, wordsPerRow, halfWidth, isErosion, scratch, result) {
  const padWord = isErosion ? ALL_ONES : 0;
  const { rowCopy, power, shifted } = scratch;
  const extendedWords = rowCopy.length; // spare pad words keep the bits the h-shift pushes past the row end
  for (let word = 0; word < wordsPerRow; word += 1) rowCopy[word] = packed[rowStart + word];
  rowCopy.fill(padWord, wordsPerRow);
  shiftTowardHigher(rowCopy, extendedWords, halfWidth, padWord, power);
  let windowLength = 2 * halfWidth + 1;
  let powerLength = 1;
  let accumulatedLength = 0;
  while (windowLength > 0) {
    if (windowLength & 1) {
      if (accumulatedLength === 0) {
        result.set(power);
      } else {
        shiftTowardLower(power, 0, extendedWords, accumulatedLength, padWord, shifted);
        for (let word = 0; word < extendedWords; word += 1) {
          result[word] = isErosion ? (result[word] & shifted[word]) >>> 0 : (result[word] | shifted[word]) >>> 0;
        }
      }
      accumulatedLength += powerLength;
    }
    windowLength >>>= 1;
    if (windowLength === 0) break;
    shiftTowardLower(power, 0, extendedWords, powerLength, padWord, shifted);
    for (let word = 0; word < extendedWords; word += 1) {
      power[word] = isErosion ? (power[word] & shifted[word]) >>> 0 : (power[word] | shifted[word]) >>> 0;
    }
    powerLength *= 2;
  }
}

/** Combine kernel rows with per-row horizontal results (any centered-run kernel). */
function combineKernelRows(horizontalByHalfWidth, halfWidths, anchorRow, height, wordsPerRow, isErosion, output) {
  if (isErosion) output.fill(ALL_ONES);
  for (let kernelRow = 0; kernelRow < halfWidths.length; kernelRow += 1) {
    const rowOffset = kernelRow - anchorRow;
    const horizontal = horizontalByHalfWidth.get(halfWidths[kernelRow]);
    const firstRow = Math.max(0, -rowOffset); // out-of-image source rows: on for erode, off for dilate
    const lastRow = Math.min(height, height - rowOffset);
    for (let row = firstRow; row < lastRow; row += 1) {
      const outputStart = row * wordsPerRow;
      const sourceStart = (row + rowOffset) * wordsPerRow;
      for (let word = 0; word < wordsPerRow; word += 1) {
        output[outputStart + word] = isErosion
          ? (output[outputStart + word] & horizontal[sourceStart + word]) >>> 0
          : (output[outputStart + word] | horizontal[sourceStart + word]) >>> 0;
      }
    }
  }
}

/** rows(y + shift) combined into target(y); rows past the bottom read `padWord`. */
function combineShiftedRows(target, source, height, wordsPerRow, shift, isErosion, padWord) {
  const validRows = Math.max(0, height - shift);
  const validWords = validRows * wordsPerRow;
  const shiftWords = shift * wordsPerRow;
  if (isErosion) {
    for (let index = 0; index < validWords; index += 1) target[index] = (target[index] & source[index + shiftWords]) >>> 0;
    if (padWord === 0) target.fill(0, validWords);
  } else {
    for (let index = 0; index < validWords; index += 1) target[index] = (target[index] | source[index + shiftWords]) >>> 0;
    if (padWord !== 0) target.fill(ALL_ONES, validWords);
  }
}

/**
 * Rect kernels: output(y) = op over rows(y - anchorRow .. y - anchorRow + kernelRows - 1), by
 * doubling (log2(kernelRows) passes instead of kernelRows). Out-of-image rows read the pad.
 */
function combineRowsVertically(rows, height, wordsPerRow, kernelRows, anchorRow, isErosion, output) {
  const padWord = isErosion ? ALL_ONES : 0;
  // padded(y) = rows(y - anchorRow), pad outside the image; tall enough for every window
  const paddedHeight = height + kernelRows;
  const padded = new Uint32Array(wordsPerRow * paddedHeight).fill(padWord);
  padded.set(rows.subarray(0, height * wordsPerRow), anchorRow * wordsPerRow);
  let power = padded;
  let powerLength = 1;
  let result = null;
  let resultLength = 0;
  let remaining = kernelRows;
  while (remaining > 0) {
    if (remaining & 1) {
      if (result === null) {
        result = Uint32Array.from(power);
      } else {
        combineShiftedRows(result, power, paddedHeight, wordsPerRow, resultLength, isErosion, padWord);
      }
      resultLength += powerLength;
    }
    remaining >>>= 1;
    if (remaining === 0) break;
    const doubled = Uint32Array.from(power);
    combineShiftedRows(doubled, power, paddedHeight, wordsPerRow, powerLength, isErosion, padWord);
    power = doubled;
    powerLength *= 2;
  }
  output.set(result.subarray(0, height * wordsPerRow));
}

/** Erode (AND) or dilate (OR) a packed mask; returns packed rows with tail bits = the next pad. */
export function morphPacked(packed, width, height, wordsPerRow, halfWidths, anchorRow, isErosion, outputPadWord) {
  const distinctHalfWidths = [...new Set(halfWidths)];
  const extendedWords = wordsPerRow + 1 + (Math.max(...distinctHalfWidths) >>> 5);
  const scratch = { rowCopy: new Uint32Array(extendedWords), power: new Uint32Array(extendedWords), shifted: new Uint32Array(extendedWords) };
  const horizontalByHalfWidth = new Map();
  for (const halfWidth of distinctHalfWidths) {
    const horizontal = new Uint32Array(wordsPerRow * height);
    const rowResult = new Uint32Array(extendedWords);
    for (let row = 0; row < height; row += 1) {
      combineRowWindow(packed, row * wordsPerRow, wordsPerRow, halfWidth, isErosion, scratch, rowResult);
      horizontal.set(rowResult.subarray(0, wordsPerRow), row * wordsPerRow);
    }
    horizontalByHalfWidth.set(halfWidth, horizontal);
  }
  const output = new Uint32Array(wordsPerRow * height);
  if (distinctHalfWidths.length === 1) {
    combineRowsVertically(horizontalByHalfWidth.get(distinctHalfWidths[0]), height, wordsPerRow, halfWidths.length, anchorRow, isErosion, output);
  } else {
    combineKernelRows(horizontalByHalfWidth, halfWidths, anchorRow, height, wordsPerRow, isErosion, output);
  }
  const tailBits = width % WORD_BITS;
  if (tailBits !== 0) {
    const tailMask = (ALL_ONES << tailBits) >>> 0;
    for (let row = 0; row < height; row += 1) {
      const lastWord = row * wordsPerRow + wordsPerRow - 1;
      output[lastWord] = outputPadWord ? (output[lastWord] | tailMask) >>> 0 : (output[lastWord] & ~tailMask) >>> 0;
    }
  }
  return output;
}

/** `cv2.erode(mask, kernel)` of a binary mask (0 / 255 output). */
export function erodeBinary(maskBytes, width, height, halfWidths, anchorRow) {
  const wordsPerRow = Math.ceil(width / WORD_BITS);
  const packed = packRows(maskBytes, width, height, wordsPerRow, ALL_ONES);
  return unpackRows(morphPacked(packed, width, height, wordsPerRow, halfWidths, anchorRow, true, 0), width, height, wordsPerRow);
}

/** `cv2.dilate(mask, kernel)` of a binary mask (0 / 255 output). */
export function dilateBinary(maskBytes, width, height, halfWidths, anchorRow) {
  const wordsPerRow = Math.ceil(width / WORD_BITS);
  const packed = packRows(maskBytes, width, height, wordsPerRow, 0);
  return unpackRows(morphPacked(packed, width, height, wordsPerRow, halfWidths, anchorRow, false, 0), width, height, wordsPerRow);
}

/** `cv2.morphologyEx(mask, MORPH_OPEN, kernel)` of a binary mask (0 / 255 output). */
export function openBinary(maskBytes, width, height, halfWidths, anchorRow) {
  const wordsPerRow = Math.ceil(width / WORD_BITS);
  const packed = packRows(maskBytes, width, height, wordsPerRow, ALL_ONES);
  const eroded = morphPacked(packed, width, height, wordsPerRow, halfWidths, anchorRow, true, 0); // tail = dilate's pad (off)
  return unpackRows(morphPacked(eroded, width, height, wordsPerRow, halfWidths, anchorRow, false, 0), width, height, wordsPerRow);
}

/**
 * `cv2.morphologyEx(mask, MORPH_CLOSE, kernel)` of an already packed mask (tail bits zero).
 *
 * Returns the 0 / 255 byte mask.
 */
export function closePackedToBytes(packed, width, height, halfWidths, anchorRow) {
  const wordsPerRow = Math.ceil(width / WORD_BITS);
  const dilated = morphPacked(packed, width, height, wordsPerRow, halfWidths, anchorRow, false, ALL_ONES); // tail = erode's pad (on)
  return unpackRows(morphPacked(dilated, width, height, wordsPerRow, halfWidths, anchorRow, true, 0), width, height, wordsPerRow);
}

/** `cv2.morphologyEx(mask, MORPH_CLOSE, kernel)` of a binary mask (0 / 255 output). */
export function closeBinary(maskBytes, width, height, halfWidths, anchorRow) {
  const wordsPerRow = Math.ceil(width / WORD_BITS);
  const packed = packRows(maskBytes, width, height, wordsPerRow, 0);
  const dilated = morphPacked(packed, width, height, wordsPerRow, halfWidths, anchorRow, false, ALL_ONES); // tail = erode's pad (on)
  return unpackRows(morphPacked(dilated, width, height, wordsPerRow, halfWidths, anchorRow, true, 0), width, height, wordsPerRow);
}
