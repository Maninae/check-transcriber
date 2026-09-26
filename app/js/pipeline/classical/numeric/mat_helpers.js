/**
 * Small OpenCV.js Mat plumbing shared by every stage: point Mats, kernels, scoped cleanup.
 *
 * Every `cv.Mat` created in this pipeline must be deleted (WASM heap leaks accumulate across
 * photos). `withMats` runs a body and deletes whatever it registered, even on a throw.
 * Typed-array views (`mat.data`, `mat.data32F`) detach when the WASM heap grows, so read
 * them right after the producing cv call, before any other cv call, and copy what you keep.
 */

/** Run `body(track)`; every Mat passed to `track` is deleted afterwards. Returns body's value. */
export function withMats(body) {
  const trackedMats = [];
  const track = (mat) => {
    trackedMats.push(mat);
    return mat;
  };
  try {
    return body(track);
  } finally {
    for (const mat of trackedMats) {
      if (!mat.isDeleted()) mat.delete();
    }
  }
}

/** A CV_32FC2 N x 1 Mat of `[[x, y], ...]` points (caller deletes). */
export function createFloat32PointMat(cv, points) {
  const pointMat = new cv.Mat(points.length, 1, cv.CV_32FC2);
  const pointValues = pointMat.data32F;
  for (let index = 0; index < points.length; index += 1) {
    pointValues[2 * index] = points[index][0];
    pointValues[2 * index + 1] = points[index][1];
  }
  return pointMat;
}

/** A CV_32SC2 N x 1 Mat of integer `[[x, y], ...]` points (caller deletes). */
export function createInt32PointMat(cv, points) {
  const pointMat = new cv.Mat(points.length, 1, cv.CV_32SC2);
  const pointValues = pointMat.data32S;
  for (let index = 0; index < points.length; index += 1) {
    pointValues[2 * index] = points[index][0];
    pointValues[2 * index + 1] = points[index][1];
  }
  return pointMat;
}

/** A CV_32FC2 Mat of a flat `[x0, y0, x1, y1, ...]` point list (caller deletes). */
export function createFloat32PointMatFromFlat(cv, flatPoints, pointCount = flatPoints.length >> 1) {
  const pointMat = new cv.Mat(pointCount, 1, cv.CV_32FC2);
  pointMat.data32F.set(flatPoints.subarray ? flatPoints.subarray(0, 2 * pointCount) : flatPoints.slice(0, 2 * pointCount));
  return pointMat;
}

/** `cv2.getStructuringElement(shape, (size, size))` (caller deletes). */
export function createSquareKernel(cv, shape, size) {
  return cv.getStructuringElement(shape, new cv.Size(size, size));
}

/** A CV_32F Mat holding a copy of `values` (caller deletes). */
export function createFloat32MatFromValues(cv, values, width, height) {
  const mat = new cv.Mat(height, width, cv.CV_32F);
  mat.data32F.set(values);
  return mat;
}

/** A CV_8U Mat holding a copy of `values` (caller deletes). */
export function createUint8MatFromValues(cv, values, width, height) {
  const mat = new cv.Mat(height, width, cv.CV_8U);
  mat.data.set(values);
  return mat;
}

/**
 * External contours of a uint8 mask as flat Int32Arrays (cv.findContours).
 *
 * Args:
 *   maskMat: CV_8U Mat (not modified in OpenCV >= 3.2).
 *   approximation: cv.CHAIN_APPROX_NONE or cv.CHAIN_APPROX_SIMPLE.
 * Returns:
 *   Array of Int32Array `[x0, y0, ...]`, in OpenCV's order.
 */
export function findExternalContours(cv, maskMat, approximation) {
  return withMats((track) => {
    const contours = new cv.MatVector();
    const hierarchy = track(new cv.Mat());
    try {
      cv.findContours(maskMat, contours, hierarchy, cv.RETR_EXTERNAL, approximation);
      const flatContours = [];
      for (let index = 0; index < contours.size(); index += 1) {
        const contour = contours.get(index);
        flatContours.push(Int32Array.from(contour.data32S));
        contour.delete();
      }
      return flatContours;
    } finally {
      contours.delete();
    }
  });
}
