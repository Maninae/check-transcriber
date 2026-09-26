/**
 * Bit-exact ports of the three Pillow 12 operations the handwriting reader's input recipe
 * uses (experiments/field_reading/learned/trocr_reader.py `trocr_pixel_values` with grey
 * input): RGB -> L conversion, `ImageOps.autocontrast(cutoff=2)`, and `Image.resize` with
 * BICUBIC. TrOCR collapses on real crops without the grey + autocontrast step (README:
 * 0.82 -> 0.09 on amounts), so this is load-bearing, not cosmetic.
 *
 * All images here are single-channel Uint8Arrays in row-major order.
 */

// Pillow's Convert.c L24: ITU-R 601-2 luma in 16-bit fixed point, rounded.
const LUMA_RED_WEIGHT = 19595;
const LUMA_GREEN_WEIGHT = 38470;
const LUMA_BLUE_WEIGHT = 7471;
const LUMA_ROUNDING = 0x8000;
const LUMA_SHIFT_DIVISOR = 65536;

// Resample.c: 8-bit fixed-point coefficients with 22 fractional bits.
const RESAMPLE_PRECISION_BITS = 22;
const RESAMPLE_ONE = 2 ** RESAMPLE_PRECISION_BITS;
const BICUBIC_A = -0.5;
const BICUBIC_SUPPORT = 2.0;

/** RGB(A) pixels -> Pillow "L" (`image.convert("L")`). `channelCount` is 3 or 4. */
export function convertRgbToLuma(pixels, width, height, channelCount) {
  const luma = new Uint8Array(width * height);
  for (let index = 0; index < luma.length; index += 1) {
    const offset = index * channelCount;
    const weighted = pixels[offset] * LUMA_RED_WEIGHT + pixels[offset + 1] * LUMA_GREEN_WEIGHT
      + pixels[offset + 2] * LUMA_BLUE_WEIGHT + LUMA_ROUNDING;
    luma[index] = Math.floor(weighted / LUMA_SHIFT_DIVISOR);
  }
  return luma;
}

/** `ImageOps.autocontrast(image, cutoff=cutoffPercent)` for an L image. */
export function autocontrastLuma(luma, cutoffPercent) {
  const histogram = new Array(256).fill(0);
  for (const value of luma) histogram[value] += 1;
  const pixelCount = luma.length;
  let cut = Math.floor((pixelCount * cutoffPercent) / 100);
  for (let low = 0; low < 256; low += 1) {
    if (cut > histogram[low]) {
      cut -= histogram[low];
      histogram[low] = 0;
    } else {
      histogram[low] -= cut;
      cut = 0;
    }
    if (cut <= 0) break;
  }
  cut = Math.floor((pixelCount * cutoffPercent) / 100);
  for (let high = 255; high >= 0; high -= 1) {
    if (cut > histogram[high]) {
      cut -= histogram[high];
      histogram[high] = 0;
    } else {
      histogram[high] -= cut;
      cut = 0;
    }
    if (cut <= 0) break;
  }
  // Python's for/break leaves the loop variable at the last index when nothing breaks.
  let lowest = 0;
  while (lowest < 255 && !histogram[lowest]) lowest += 1;
  let highest = 255;
  while (highest > 0 && !histogram[highest]) highest -= 1;
  const lookup = new Uint8Array(256);
  if (highest <= lowest) {
    for (let value = 0; value < 256; value += 1) lookup[value] = value;
  } else {
    const scale = 255.0 / (highest - lowest);
    const offset = -lowest * scale;
    for (let value = 0; value < 256; value += 1) {
      lookup[value] = Math.min(255, Math.max(0, Math.trunc(value * scale + offset)));
    }
  }
  return luma.map((value) => lookup[value]);
}

function bicubicFilter(distance) {
  const x = Math.abs(distance);
  if (x < 1.0) return ((BICUBIC_A + 2.0) * x - (BICUBIC_A + 3.0)) * x * x + 1;
  if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * BICUBIC_A;
  return 0.0;
}

/**
 * Resample.c `precompute_coeffs` + `normalize_coeffs_8bpc` for one axis:
 * returns `{ kernelSize, bounds: Int32Array(2 * outSize), coefficients: Float64Array }`,
 * coefficients already in 22-bit fixed point.
 */
function precomputeFixedPointCoefficients(inSize, outSize) {
  const scale = inSize / outSize;
  const filterScale = Math.max(1.0, scale);
  const support = BICUBIC_SUPPORT * filterScale;
  const kernelSize = Math.ceil(support) * 2 + 1;
  const bounds = new Int32Array(outSize * 2);
  const coefficients = new Float64Array(outSize * kernelSize);
  for (let outIndex = 0; outIndex < outSize; outIndex += 1) {
    const center = (outIndex + 0.5) * scale;
    const inverseScale = 1.0 / filterScale;
    let minimum = Math.trunc(center - support + 0.5);
    if (minimum < 0) minimum = 0;
    let maximum = Math.trunc(center + support + 0.5);
    if (maximum > inSize) maximum = inSize;
    maximum -= minimum;
    const weights = new Float64Array(maximum);
    let weightSum = 0.0;
    for (let x = 0; x < maximum; x += 1) {
      weights[x] = bicubicFilter((x + minimum - center + 0.5) * inverseScale);
      weightSum += weights[x];
    }
    for (let x = 0; x < maximum; x += 1) {
      const normalized = weightSum !== 0.0 ? weights[x] / weightSum : weights[x];
      coefficients[outIndex * kernelSize + x] = Math.trunc(normalized * RESAMPLE_ONE + (normalized < 0 ? -0.5 : 0.5));
    }
    bounds[outIndex * 2] = minimum;
    bounds[outIndex * 2 + 1] = maximum;
  }
  return { kernelSize, bounds, coefficients };
}

function clipFixedPointToByte(accumulator) {
  if (accumulator >= RESAMPLE_ONE * 256) return 255;
  if (accumulator <= 0) return 0;
  return Math.floor(accumulator / RESAMPLE_ONE);
}

/** `image.resize((outWidth, outHeight), Image.BICUBIC)` for an L image (horizontal pass first, like Pillow). */
export function resizeLumaBicubic(luma, width, height, outWidth, outHeight) {
  const horizontal = precomputeFixedPointCoefficients(width, outWidth);
  const vertical = precomputeFixedPointCoefficients(height, outHeight);
  const needHorizontal = outWidth !== width;
  const needVertical = outHeight !== height;
  let source = luma;
  let sourceWidth = width;
  if (needHorizontal) {
    const firstRow = vertical.bounds[0];
    const lastRow = vertical.bounds[outHeight * 2 - 2] + vertical.bounds[outHeight * 2 - 1];
    for (let row = 0; row < outHeight; row += 1) vertical.bounds[row * 2] -= firstRow;
    const rowCount = lastRow - firstRow;
    const temporary = new Uint8Array(outWidth * rowCount);
    for (let row = 0; row < rowCount; row += 1) {
      const sourceRowOffset = (row + firstRow) * width;
      for (let column = 0; column < outWidth; column += 1) {
        const minimum = horizontal.bounds[column * 2];
        const count = horizontal.bounds[column * 2 + 1];
        const kernelOffset = column * horizontal.kernelSize;
        let accumulator = RESAMPLE_ONE / 2;
        for (let x = 0; x < count; x += 1) {
          accumulator += source[sourceRowOffset + minimum + x] * horizontal.coefficients[kernelOffset + x];
        }
        temporary[row * outWidth + column] = clipFixedPointToByte(accumulator);
      }
    }
    source = temporary;
    sourceWidth = outWidth;
  }
  if (!needVertical) return source;
  const output = new Uint8Array(sourceWidth * outHeight);
  for (let row = 0; row < outHeight; row += 1) {
    const minimum = vertical.bounds[row * 2];
    const count = vertical.bounds[row * 2 + 1];
    const kernelOffset = row * vertical.kernelSize;
    for (let column = 0; column < sourceWidth; column += 1) {
      let accumulator = RESAMPLE_ONE / 2;
      for (let y = 0; y < count; y += 1) {
        accumulator += source[(minimum + y) * sourceWidth + column] * vertical.coefficients[kernelOffset + y];
      }
      output[row * sourceWidth + column] = clipFixedPointToByte(accumulator);
    }
  }
  return output;
}
