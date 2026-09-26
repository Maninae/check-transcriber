/**
 * Bit-packed binary rows: 32 pixels per Uint32 word (bit b of word w = pixel 32w + b).
 *
 * Primitives for `binary_morphology.js` and the cell-pair unions: pack / unpack a 0 / nonzero
 * byte mask, shift a packed row left or right with a pad word for the out-of-row positions,
 * and OR a packed bitmap into a larger canvas at a pixel offset.
 */

export const WORD_BITS = 32;
export const ALL_ONES = 0xffffffff;
const ON_VALUE = 255;

/** Pack a byte mask into rows of 32-bit words (bit b of word w = pixel 32w + b); tail bits = pad. */
export function packRows(maskBytes, width, height, wordsPerRow, padWord) {
  const packed = new Uint32Array(wordsPerRow * height);
  const tailBits = width % WORD_BITS;
  for (let row = 0; row < height; row += 1) {
    const byteStart = row * width;
    const wordStart = row * wordsPerRow;
    for (let word = 0; word < wordsPerRow; word += 1) {
      const firstColumn = word * WORD_BITS;
      const bitCount = Math.min(WORD_BITS, width - firstColumn);
      let packedWord = 0;
      for (let bit = 0; bit < bitCount; bit += 1) packedWord |= (maskBytes[byteStart + firstColumn + bit] !== 0 ? 1 : 0) << bit;
      packed[wordStart + word] = packedWord >>> 0;
    }
    if (tailBits !== 0 && padWord !== 0) packed[wordStart + wordsPerRow - 1] |= (ALL_ONES << tailBits) >>> 0;
  }
  return packed;
}

/** destination(x) = source(x + shift) for shift >= 0; positions past the row read `padWord`. */
export function shiftTowardLower(source, sourceStart, wordsPerRow, shift, padWord, destination) {
  const wordShift = shift >>> 5;
  const bitShift = shift & 31;
  for (let word = 0; word < wordsPerRow; word += 1) {
    const lowIndex = word + wordShift;
    const low = lowIndex < wordsPerRow ? source[sourceStart + lowIndex] : padWord;
    if (bitShift === 0) {
      destination[word] = low;
    } else {
      const high = lowIndex + 1 < wordsPerRow ? source[sourceStart + lowIndex + 1] : padWord;
      destination[word] = ((low >>> bitShift) | (high << (WORD_BITS - bitShift))) >>> 0;
    }
  }
}

/** destination(x) = source(x - shift) for shift >= 0; positions before the row read `padWord`. */
export function shiftTowardHigher(source, wordsPerRow, shift, padWord, destination) {
  const wordShift = shift >>> 5;
  const bitShift = shift & 31;
  for (let word = 0; word < wordsPerRow; word += 1) {
    const currentIndex = word - wordShift;
    const current = currentIndex >= 0 ? source[currentIndex] : padWord;
    if (bitShift === 0) {
      destination[word] = current;
    } else {
      const lower = currentIndex - 1 >= 0 ? source[currentIndex - 1] : padWord;
      destination[word] = ((current << bitShift) | (lower >>> (WORD_BITS - bitShift))) >>> 0;
    }
  }
}

/** Unpack packed rows into a 0 / 255 byte mask. */
export function unpackRows(packed, width, height, wordsPerRow) {
  const maskBytes = new Uint8Array(width * height);
  for (let row = 0; row < height; row += 1) {
    const wordStart = row * wordsPerRow;
    const byteStart = row * width;
    for (let word = 0; word < wordsPerRow; word += 1) {
      const packedWord = packed[wordStart + word];
      if (packedWord === 0) continue;
      const firstColumn = word * WORD_BITS;
      const bitCount = Math.min(WORD_BITS, width - firstColumn);
      for (let bit = 0; bit < bitCount; bit += 1) maskBytes[byteStart + firstColumn + bit] = ((packedWord >>> bit) & 1) * ON_VALUE;
    }
  }
  return maskBytes;
}

/** OR `bitmap` (rows of `bitmapWords` words) into `target` at a (column, row) offset. */
export function orBitmapInto(target, targetWordsPerRow, bitmap, bitmapWords, bitmapRows, columnOffset, rowOffset) {
  const wordShift = columnOffset >>> 5;
  const bitShift = columnOffset & 31;
  for (let row = 0; row < bitmapRows; row += 1) {
    const targetStart = (row + rowOffset) * targetWordsPerRow + wordShift;
    const sourceStart = row * bitmapWords;
    for (let word = 0; word < bitmapWords; word += 1) {
      const sourceWord = bitmap[sourceStart + word];
      if (sourceWord === 0) continue;
      target[targetStart + word] = (target[targetStart + word] | (sourceWord << bitShift)) >>> 0;
      if (bitShift !== 0) {
        const carry = sourceWord >>> (WORD_BITS - bitShift);
        if (carry !== 0) target[targetStart + word + 1] = (target[targetStart + word + 1] | carry) >>> 0;
      }
    }
  }
}

