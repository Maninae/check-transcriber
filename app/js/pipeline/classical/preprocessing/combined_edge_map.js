/**
 * Canny edges from lightness OR color: the edge map every Canny-based generator shares
 * (Python `combined_edge_map.py`).
 *
 * On a white or cream bedsheet the lit sides of a check have no lightness step; only hue
 * survives (a beige, pink or blue check on a neutral sheet). Canny on Lab a and b, amplified
 * by `chromaEdgeGain` around mid-gray, finds those sides; the union with lightness Canny
 * gives closed outlines. Used by line extraction, Canny cells and hole-filled Canny masks.
 */

const NEUTRAL_CHROMA_LEVEL = 128.0; // a/b are centered on 0; Canny needs uint8 around mid-gray
const fround = Math.fround;

/** `np.clip(value, 0, 255).astype(np.uint8)` of a float (truncation toward zero). */
function clipToUint8(value) {
  if (value <= 0) return 0;
  if (value >= 255) return 255;
  return Math.trunc(value);
}

/** Canny (L2 gradient) of a uint8 plane given as JS bytes; returns a new CV_8U Mat. */
function cannyOfBytes(cv, planeBytes, width, height, lowThreshold, highThreshold) {
  const planeMat = new cv.Mat(height, width, cv.CV_8U);
  try {
    planeMat.data.set(planeBytes);
    const edgesMat = new cv.Mat();
    cv.Canny(planeMat, edgesMat, lowThreshold, highThreshold, 3, true);
    return edgesMat;
  } finally {
    planeMat.delete();
  }
}

/**
 * uint8 0/255 Canny edges of lightness, OR'd with Canny of each amplified color channel.
 *
 * Returns a new CV_8U Mat of the working size; the caller deletes it.
 */
export function buildCombinedCannyEdges(cv, channels, lowThreshold, highThreshold, config) {
  const { width, height } = channels;
  const pixelCount = width * height;
  const planeBytes = new Uint8Array(pixelCount);
  const lightnessValues = channels.lightness.values;
  for (let index = 0; index < pixelCount; index += 1) planeBytes[index] = clipToUint8(lightnessValues[index]);
  const edgesMat = cannyOfBytes(cv, planeBytes, width, height, lowThreshold, highThreshold);
  if (!config.useChromaEdges) return edgesMat;
  const gain = fround(config.chromaEdgeGain);
  for (const colorChannel of [channels.labA, channels.labB]) {
    const colorValues = colorChannel.values;
    for (let index = 0; index < pixelCount; index += 1) {
      planeBytes[index] = clipToUint8(fround(NEUTRAL_CHROMA_LEVEL + fround(gain * colorValues[index])));
    }
    const colorEdgesMat = cannyOfBytes(cv, planeBytes, width, height, lowThreshold, highThreshold);
    const colorEdgeBytes = colorEdgesMat.data;
    const combinedBytes = edgesMat.data;
    for (let index = 0; index < pixelCount; index += 1) combinedBytes[index] |= colorEdgeBytes[index];
    colorEdgesMat.delete();
  }
  return edgesMat;
}
