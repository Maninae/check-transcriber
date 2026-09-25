/**
 * Turns an uploaded/pasted/dropped image file into two in-memory canvases: a
 * full-resolution one (kept for legible crops in a later milestone) and a working
 * copy downscaled to a manageable size (for the detection stages a later milestone
 * adds). This module owns the only place EXIF orientation is read and the only
 * place the original File/Blob is touched.
 *
 * Privacy invariant this module exists to uphold: EXIF can carry GPS coordinates.
 * `createImageBitmap` decodes pixels without exposing EXIF to our code at all, and
 * the canvases below hold pixels only, so once a canvas exists there is no EXIF left
 * anywhere in memory. Callers must not hold onto the original File/Blob after calling
 * this function returns; letting it fall out of scope is the whole mitigation.
 */

// Long-edge target for the working copy. Matches the spec's detection-stage budget
// (a later milestone runs contour-finding on this copy, not the full-res original).
export const WORKING_COPY_LONG_EDGE_PX = 2500;

/**
 * Decodes `file` into an oriented ImageBitmap, honoring EXIF orientation so a
 * portrait photo isn't processed sideways. Chrome/Edge (this app's only supported
 * browsers) have supported `imageOrientation: "from-image"` since 2020, but a
 * defensive fallback keeps a photo viewable even if a future browser regresses this.
 */
async function createOrientedImageBitmap(file) {
  try {
    return await createImageBitmap(file, { imageOrientation: "from-image" });
  } catch (orientationError) {
    console.warn(
      "createImageBitmap with imageOrientation failed, falling back to unoriented decode:",
      orientationError,
    );
    return createImageBitmap(file);
  }
}

/**
 * Scales (width, height) down to fit within `maxLongEdge` on its longer side.
 * Returns the original size unchanged if it's already within budget (never upscale).
 */
function computeDownscaledDimensions(width, height, maxLongEdge) {
  const longEdge = Math.max(width, height);
  if (longEdge <= maxLongEdge) {
    return { width, height };
  }
  const scale = maxLongEdge / longEdge;
  return {
    width: Math.round(width * scale),
    height: Math.round(height * scale),
  };
}

/**
 * Draws `bitmap` onto a new canvas at (width, height), resampling if they differ
 * from the bitmap's native size.
 */
function drawBitmapToCanvas(bitmap, width, height) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  context.drawImage(bitmap, 0, 0, width, height);
  return canvas;
}

/**
 * Decodes and orients `file`, producing:
 * - `fullResCanvas`: full resolution, upright, for legible crops later.
 * - `workingCanvas`: downscaled to `WORKING_COPY_LONG_EDGE_PX` on the long edge.
 * - `width`/`height`: the full-resolution oriented dimensions (post-rotation).
 *
 * The caller owns `file` and must drop its reference once this resolves; see the
 * module docstring above for why.
 */
export async function decodeAndOrientImage(file) {
  const bitmap = await createOrientedImageBitmap(file);
  const { width, height } = bitmap;

  const fullResCanvas = drawBitmapToCanvas(bitmap, width, height);
  const workingDimensions = computeDownscaledDimensions(width, height, WORKING_COPY_LONG_EDGE_PX);
  const workingCanvas = drawBitmapToCanvas(bitmap, workingDimensions.width, workingDimensions.height);

  // The pixels are safely copied onto both canvases now; release the decoder's copy.
  bitmap.close();

  return { fullResCanvas, workingCanvas, width, height };
}
