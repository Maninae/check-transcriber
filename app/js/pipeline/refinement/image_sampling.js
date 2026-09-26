/**
 * The one place refinement touches OpenCV: bilinear sampling of the uint8 image at float32 maps.
 *
 * Mirrors `cv2.remap(image, map_x, map_y, INTER_LINEAR, borderMode=BORDER_REPLICATE)` on the
 * full-resolution image, so there is no crop to copy. Same OpenCV version (5.0.0) as the
 * Python reference, so the fixed-point bilinear weights and rounding are identical.
 *
 * - Every Mat created here is deleted in `finally`; the caller owns `image`.
 * - The output is COPIED out of the WASM heap before returning (a later allocation may grow
 *   the heap and detach any typed-array view into it).
 */

/**
 * Returns a Uint8Array (rows x cols x channels) of `image` sampled at (mapX, mapY).
 *
 * Args:
 *   cv: the OpenCV.js module.
 *   image: cv.Mat CV_8UC1/CV_8UC3.
 *   mapX, mapY: Float32Array of rows * cols image coordinates.
 */
export function sampleImageBilinear(cv, image, mapX, mapY, rows, cols) {
  const mapXMat = new cv.Mat(rows, cols, cv.CV_32FC1);
  const mapYMat = new cv.Mat(rows, cols, cv.CV_32FC1);
  const sampled = new cv.Mat();
  try {
    mapXMat.data32F.set(mapX);
    mapYMat.data32F.set(mapY);
    cv.remap(image, sampled, mapXMat, mapYMat, cv.INTER_LINEAR, cv.BORDER_REPLICATE, new cv.Scalar());
    return new Uint8Array(sampled.data);
  } finally {
    mapXMat.delete();
    mapYMat.delete();
    sampled.delete();
  }
}
