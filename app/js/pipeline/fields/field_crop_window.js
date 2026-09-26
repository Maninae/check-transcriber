/**
 * The field-crop framing every reader was trained and scored with (a port of
 * experiments/field_reading/data_access/field_crop.py): the box plus 15% of its height
 * (at least 6 px) on every side, clipped to the check. Pure, no OpenCV.
 */

import { FIELD_CROP_MARGIN_FRACTION, FIELD_CROP_MINIMUM_MARGIN_PX } from "./field_reading_config.js";

/** Integer [x0, y0, x1, y1] window for a box ([x0, y0, x1, y1] crop pixels) on a check of the given size. */
export function fieldCropWindow(box, checkWidth, checkHeight) {
  const [x0, y0, x1, y1] = box;
  const margin = Math.max(FIELD_CROP_MINIMUM_MARGIN_PX, FIELD_CROP_MARGIN_FRACTION * (y1 - y0));
  return [
    Math.max(0, Math.floor(x0 - margin)),
    Math.max(0, Math.floor(y0 - margin)),
    Math.min(checkWidth, Math.ceil(x1 + margin)),
    Math.min(checkHeight, Math.ceil(y1 + margin)),
  ];
}
