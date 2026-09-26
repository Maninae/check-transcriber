/**
 * Numbers checks in reading order (spec 4.2): top-to-bottom by row, then left-to-right
 * within a row, matching how the operator scans the bedsheet.
 *
 * Rows are formed greedily from vertical centres: walking the checks from the top, a
 * check joins the current row while its centre sits within half a typical check height
 * of the row's first check. Checks lie at any angle, so "height" is each quad's
 * vertical extent in the photo, and the typical value is their median.
 */

import { computeQuadBounds, computeQuadCentre } from "./quadrilateral_math.js";

const ROW_TOLERANCE_FRACTION_OF_MEDIAN_HEIGHT = 0.5;

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

/**
 * Returns a new array of the same items sorted into reading order.
 * `getCorners(item)` extracts the quad from each item (defaults to the item itself).
 */
export function sortIntoReadingOrder(items, getCorners = (item) => item) {
  if (items.length <= 1) return [...items];
  const described = items.map((item) => {
    const corners = getCorners(item);
    const bounds = computeQuadBounds(corners);
    const [centreX, centreY] = computeQuadCentre(corners);
    return { item, centreX, centreY, height: bounds.maxY - bounds.minY };
  });
  const rowTolerance = ROW_TOLERANCE_FRACTION_OF_MEDIAN_HEIGHT * median(described.map(({ height }) => height));
  described.sort((a, b) => a.centreY - b.centreY);

  const rows = [];
  for (const entry of described) {
    const currentRow = rows[rows.length - 1];
    if (currentRow && entry.centreY - currentRow[0].centreY <= rowTolerance) {
      currentRow.push(entry);
    } else {
      rows.push([entry]);
    }
  }
  return rows.flatMap((row) => row.sort((a, b) => a.centreX - b.centreX).map(({ item }) => item));
}
