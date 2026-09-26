/**
 * "Add a check" (spec 4.2): the operator drags a rough rectangle; this re-fits it to the
 * nearest strong edges so a rough drag becomes a tight outline.
 *
 * Strategy, most specific first:
 * 1. Run the classical detector on a crop around the drawn rectangle (padded so the whole
 *    check is inside even when the drag was tight) and take the detection that best
 *    overlaps the drag, when the overlap is convincing.
 * 2. Otherwise refine the drawn rectangle itself with the corner refinement.
 * 3. If refinement wanders away from the drag, keep the drag as drawn: the operator's
 *    box is never replaced by something that disagrees with it.
 * Either way the result goes through corner refinement against the full photo, with the
 * other checks masked out.
 */

import { detectChecksClassical } from "./classical/detect_checks_classical.js";
import { refineCheckQuadrilateral } from "./refinement/quadrilateral_refinement.js";
import { computeQuadBounds, computeQuadIoU, orderCornersClockwise } from "./quadrilateral_math.js";

const SEARCH_PADDING_FRACTION = 0.25; // of the drawn box's width/height, each side
const MINIMUM_DETECTION_IOU_WITH_DRAWN = 0.5;
const MINIMUM_REFINED_IOU_WITH_DRAWN = 0.6;

function detectInsidePaddedRegion(cv, imageBgr, drawnCorners) {
  const bounds = computeQuadBounds(drawnCorners);
  const padX = (bounds.maxX - bounds.minX) * SEARCH_PADDING_FRACTION;
  const padY = (bounds.maxY - bounds.minY) * SEARCH_PADDING_FRACTION;
  const left = Math.max(0, Math.floor(bounds.minX - padX));
  const top = Math.max(0, Math.floor(bounds.minY - padY));
  const right = Math.min(imageBgr.cols, Math.ceil(bounds.maxX + padX));
  const bottom = Math.min(imageBgr.rows, Math.ceil(bounds.maxY + padY));
  if (right - left < 16 || bottom - top < 16) return [];
  const regionView = imageBgr.roi(new cv.Rect(left, top, right - left, bottom - top));
  // A ROI is a strided view; the detector reads `.data` directly, so give it a continuous copy.
  const regionCopy = regionView.clone();
  try {
    return detectChecksClassical(cv, regionCopy).map(({ corners }) => corners.map(([x, y]) => [x + left, y + top]));
  } finally {
    regionView.delete();
    regionCopy.delete();
  }
}

/** Returns `{ corners, refitSource }` where refitSource is "detected", "refined" or "as-drawn". */
export function refitDrawnRectangle(cv, imageBgr, drawnCorners, otherCornerSets) {
  const drawnClockwise = orderCornersClockwise(drawnCorners);
  const candidates = detectInsidePaddedRegion(cv, imageBgr, drawnClockwise);
  let bestDetection = null;
  let bestIoU = 0;
  for (const candidate of candidates) {
    const overlap = computeQuadIoU(candidate, drawnClockwise);
    if (overlap > bestIoU) {
      bestIoU = overlap;
      bestDetection = candidate;
    }
  }
  const startingCorners = bestIoU >= MINIMUM_DETECTION_IOU_WITH_DRAWN ? bestDetection : drawnClockwise;
  const { corners: refinedCorners } = refineCheckQuadrilateral(cv, imageBgr, startingCorners, undefined, otherCornerSets);
  const refitSource = startingCorners === drawnClockwise ? "refined" : "detected";
  if (computeQuadIoU(refinedCorners, drawnClockwise) < MINIMUM_REFINED_IOU_WITH_DRAWN && refitSource === "refined") {
    return { corners: drawnClockwise, refitSource: "as-drawn" };
  }
  return { corners: orderCornersClockwise(refinedCorners), refitSource };
}
