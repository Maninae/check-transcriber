/**
 * The inline magnifier and the box outline for unsure and blank fields (spec 4.3 "Blank"):
 * while such a field has focus, its crop region is outlined on the row's crop and drawn
 * large right under the crop, in the row (never a modal), so the answer is beside the input.
 *
 * - Region: the field's read box plus a margin, or, with no box, the field's usual place on
 *   a personal check (`FALLBACK_FIELD_REGIONS`, fractions of the upright crop).
 * - PRIVACY: always drawn from the row's DISPLAYED crop, which has the MICR band blurred
 *   unless the operator chose "Show bottom line"; never from the unblurred upright canvas.
 * - Boxes and regions are in the displayed crop's full-resolution pixels; the row canvas
 *   shows that crop scaled to `CROP_DISPLAY_WIDTH_PX`.
 */

import { FALLBACK_FIELD_REGIONS } from "./field_definitions.js";

// Padding around a read box: half its height above and below, a bit more sideways,
// so the whole handwritten word and its neighbours are in view.
const BOX_MARGIN_FRACTION_OF_BOX_HEIGHT = 0.5;
const BOX_SIDE_MARGIN_FRACTION_OF_CROP_WIDTH = 0.02;
const MAGNIFIER_MAXIMUM_DISPLAY_HEIGHT_PX = 170;

/** The crop region to magnify for `fieldKey`, `[x0, y0, x1, y1]` in crop pixels. */
export function magnifierRegionForField(fieldKey, box, cropWidth, cropHeight) {
  let region;
  if (box) {
    const [x0, y0, x1, y1] = box;
    const verticalMargin = (y1 - y0) * BOX_MARGIN_FRACTION_OF_BOX_HEIGHT;
    const sideMargin = verticalMargin + cropWidth * BOX_SIDE_MARGIN_FRACTION_OF_CROP_WIDTH;
    region = [x0 - sideMargin, y0 - verticalMargin, x1 + sideMargin, y1 + verticalMargin];
  } else {
    const [fx0, fy0, fx1, fy1] = FALLBACK_FIELD_REGIONS[fieldKey];
    region = [fx0 * cropWidth, fy0 * cropHeight, fx1 * cropWidth, fy1 * cropHeight];
  }
  const [x0, y0, x1, y1] = region;
  return [Math.max(0, x0), Math.max(0, y0), Math.min(cropWidth, x1), Math.min(cropHeight, y1)].map(Math.round);
}

/**
 * Draws `region` of `displayedCropCanvas` into `magnifierCanvas`, as large as fits in
 * `maximumDisplayWidth` x `MAGNIFIER_MAXIMUM_DISPLAY_HEIGHT_PX` CSS pixels.
 */
export function drawMagnifiedRegion(magnifierCanvas, displayedCropCanvas, region, maximumDisplayWidth) {
  const [x0, y0, x1, y1] = region;
  const regionWidth = Math.max(1, x1 - x0);
  const regionHeight = Math.max(1, y1 - y0);
  const displayScale = Math.min(maximumDisplayWidth / regionWidth, MAGNIFIER_MAXIMUM_DISPLAY_HEIGHT_PX / regionHeight);
  const displayWidth = Math.round(regionWidth * displayScale);
  const displayHeight = Math.round(regionHeight * displayScale);
  const devicePixelRatio = window.devicePixelRatio || 1;
  magnifierCanvas.width = Math.round(displayWidth * devicePixelRatio);
  magnifierCanvas.height = Math.round(displayHeight * devicePixelRatio);
  magnifierCanvas.style.width = `${displayWidth}px`;
  magnifierCanvas.style.height = `${displayHeight}px`;
  const context = magnifierCanvas.getContext("2d");
  context.imageSmoothingQuality = "high";
  context.drawImage(displayedCropCanvas, x0, y0, regionWidth, regionHeight, 0, 0, magnifierCanvas.width, magnifierCanvas.height);
}

/** Places `outlineElement` over the row canvas around `box` (crop pixels); hides it for no box. */
export function positionBoxOutline(outlineElement, box, cropWidth, displayWidth) {
  if (!box) {
    outlineElement.hidden = true;
    return;
  }
  const scale = displayWidth / cropWidth;
  const [x0, y0, x1, y1] = box;
  outlineElement.style.left = `${x0 * scale}px`;
  outlineElement.style.top = `${y0 * scale}px`;
  outlineElement.style.width = `${(x1 - x0) * scale}px`;
  outlineElement.style.height = `${(y1 - y0) * scale}px`;
  outlineElement.hidden = false;
}
