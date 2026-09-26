/**
 * Turns a check's upright crop (straight from the pipeline worker) into what the page
 * shows: rotated as the operator chose, with the MICR band blurred (spec section 5, stage
 * 8, and the privacy rules in section 7).
 *
 * The MICR line (routing and account numbers) sits 0.19-0.31 in above the bottom edge,
 * inside the bottom 5/8 in clear band (ANSI X9); on 2.75-3.5 in tall checks that band is
 * at most 23% of the height, so the bottom `MICR_BAND_HEIGHT_FRACTION` of the displayed
 * crop is blurred. When the orientation call was close, the top band is blurred too,
 * because on a wrongly-oriented crop the MICR line is at the top.
 *
 * Canvases only (never <img> or object URLs): pixels stay in memory, no URL to leak.
 */

const MICR_BAND_HEIGHT_FRACTION = 0.23;
const MICR_BLUR_RADIUS_FRACTION_OF_WIDTH = 0.012; // ~19 px on a 1600 px crop; digits are ~30 px tall
// Orientation calls between these upside-down probabilities blur both bands.
const UNSURE_ORIENTATION_PROBABILITY_RANGE = [0.15, 0.85];

/** A canvas holding the worker's RGBA crop pixels. */
export function createCanvasFromRgbaPixels(width, height, rgbaBuffer) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  canvas.getContext("2d").putImageData(new ImageData(new Uint8ClampedArray(rgbaBuffer), width, height), 0, 0);
  return canvas;
}

/** Whether the orientation classifier's call was too close to trust for the blur band. */
export function isOrientationUnsure(upsideDownProbability) {
  return upsideDownProbability > UNSURE_ORIENTATION_PROBABILITY_RANGE[0]
    && upsideDownProbability < UNSURE_ORIENTATION_PROBABILITY_RANGE[1];
}

/**
 * Whether the top band must be blurred too: whenever the crop as displayed has more than a
 * small chance of being upside down (its MICR line at the top). Without a rotation that is the
 * unsure range; after the operator's half turn the classifier's call flips, so a check the
 * classifier was sure about gets its (now top) MICR band blurred as well. A rotation that fixes
 * a confidently wrong call also blurs the payer line; that rare case is the price of never
 * showing bank numbers by accident ("Show bottom line" unblurs both).
 */
export function shouldBlurTopBand(upsideDownProbability, rotatedHalfTurn) {
  // The worker already turned the crop when p > 0.5, so the upright crop is wrong with min(p, 1 - p).
  const uprightCropWrongProbability = Math.min(upsideDownProbability, 1 - upsideDownProbability);
  const displayedUpsideDownProbability = rotatedHalfTurn ? 1 - uprightCropWrongProbability : uprightCropWrongProbability;
  return displayedUpsideDownProbability > UNSURE_ORIENTATION_PROBABILITY_RANGE[0];
}

function blurHorizontalBand(context, sourceCanvas, bandTop, bandHeight, blurRadius) {
  const { width } = sourceCanvas;
  context.save();
  context.beginPath();
  context.rect(0, bandTop, width, bandHeight);
  context.clip();
  context.filter = `blur(${blurRadius}px)`;
  // Drawing the band's own pixels blurred over itself; the clip keeps the blur inside it.
  context.drawImage(sourceCanvas, 0, 0);
  context.restore();
}

/**
 * Composes the displayed crop at full crop resolution.
 * `options`: `{ rotatedHalfTurn, micrBlurred, blurTopBandToo }`. Returns a new canvas.
 */
export function composeDisplayedCrop(uprightCropCanvas, { rotatedHalfTurn, micrBlurred, blurTopBandToo }) {
  const { width, height } = uprightCropCanvas;
  const rotatedCanvas = document.createElement("canvas");
  rotatedCanvas.width = width;
  rotatedCanvas.height = height;
  const rotatedContext = rotatedCanvas.getContext("2d");
  if (rotatedHalfTurn) {
    rotatedContext.translate(width, height);
    rotatedContext.rotate(Math.PI);
  }
  rotatedContext.drawImage(uprightCropCanvas, 0, 0);
  if (!micrBlurred) return rotatedCanvas;

  const displayedCanvas = document.createElement("canvas");
  displayedCanvas.width = width;
  displayedCanvas.height = height;
  const displayedContext = displayedCanvas.getContext("2d");
  displayedContext.drawImage(rotatedCanvas, 0, 0);
  const bandHeight = Math.round(height * MICR_BAND_HEIGHT_FRACTION);
  const blurRadius = Math.max(4, Math.round(width * MICR_BLUR_RADIUS_FRACTION_OF_WIDTH));
  blurHorizontalBand(displayedContext, rotatedCanvas, height - bandHeight, bandHeight, blurRadius);
  if (blurTopBandToo) blurHorizontalBand(displayedContext, rotatedCanvas, 0, bandHeight, blurRadius);
  rotatedCanvas.width = 0; // release the intermediate's pixels now rather than at GC
  return displayedCanvas;
}

/** Draws `sourceCanvas` into `targetCanvas`, resizing the target to `displayWidth` x device pixels. */
export function drawCanvasScaledToWidth(targetCanvas, sourceCanvas, displayWidth) {
  const devicePixelRatio = window.devicePixelRatio || 1;
  const targetWidth = Math.round(displayWidth * devicePixelRatio);
  const targetHeight = Math.round((targetWidth * sourceCanvas.height) / sourceCanvas.width);
  targetCanvas.width = targetWidth;
  targetCanvas.height = targetHeight;
  const context = targetCanvas.getContext("2d");
  context.imageSmoothingQuality = "high";
  context.drawImage(sourceCanvas, 0, 0, targetWidth, targetHeight);
}
