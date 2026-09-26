/**
 * Greedy fusion of collinear edge segments (Python `line_segment_extraction.merge_collinear_segments`).
 *
 * Longest first, each segment is tested against every merged line so far: nearly the same
 * direction (within 2 degrees), both endpoints within 3 px of the line, and extents that
 * overlap or nearly touch (within `maximumGapPixels`). The first mergeable line absorbs it
 * (its extent grows along its own direction); otherwise the segment starts a new line.
 */

import { numpyArgsort } from "../numeric/numpy_compatibility.js";

const MERGE_ANGLE_TOLERANCE_DEGREES = 2.0;
const MERGE_OFFSET_TOLERANCE_PIXELS = 3.0;
const MINIMUM_SEGMENT_LENGTH = 1e-6;
const MINIMUM_COSINE = Math.cos(MERGE_ANGLE_TOLERANCE_DEGREES * (Math.PI / 180));

/** Length of a `[x1, y1, x2, y2]` segment (np.hypot). */
export function segmentLength(segment) {
  const deltaX = segment[2] - segment[0];
  const deltaY = segment[3] - segment[1];
  return Math.sqrt(deltaX * deltaX + deltaY * deltaY);
}

/** Fuse collinear, overlapping or nearly touching segments; returns merged `[x1, y1, x2, y2]`s. */
export function mergeCollinearSegments(segments, maximumGapPixels) {
  const order = numpyArgsort(segments.map((segment) => -segmentLength(segment)));
  const mergedStarts = [];
  const mergedEnds = [];
  for (const segmentIndex of order) {
    const segment = segments[segmentIndex];
    const startX = segment[0];
    const startY = segment[1];
    const endX = segment[2];
    const endY = segment[3];
    const directionX = endX - startX;
    const directionY = endY - startY;
    const length = Math.sqrt(directionX * directionX + directionY * directionY);
    if (length < MINIMUM_SEGMENT_LENGTH) continue;
    const unitSegmentX = directionX / length;
    const unitSegmentY = directionY / length;
    let mergedIndex = -1;
    let mergedGeometry = null;
    for (let candidate = 0; candidate < mergedStarts.length; candidate += 1) {
      const [anchorX, anchorY] = mergedStarts[candidate];
      const vectorX = mergedEnds[candidate][0] - anchorX;
      const vectorY = mergedEnds[candidate][1] - anchorY;
      const mergedLength = Math.sqrt(vectorX * vectorX + vectorY * vectorY);
      const unitX = vectorX / mergedLength;
      const unitY = vectorY / mergedLength;
      const normalX = -unitY;
      const normalY = unitX;
      const cosine = Math.abs(unitX * unitSegmentX + unitY * unitSegmentY);
      const startOffset = (startX - anchorX) * normalX + (startY - anchorY) * normalY;
      const endOffset = (endX - anchorX) * normalX + (endY - anchorY) * normalY;
      const startPosition = (startX - anchorX) * unitX + (startY - anchorY) * unitY;
      const endPosition = (endX - anchorX) * unitX + (endY - anchorY) * unitY;
      if (
        cosine >= MINIMUM_COSINE
        && Math.max(Math.abs(startOffset), Math.abs(endOffset)) <= MERGE_OFFSET_TOLERANCE_PIXELS
        && Math.min(startPosition, endPosition) <= mergedLength + maximumGapPixels
        && Math.max(startPosition, endPosition) >= -maximumGapPixels
      ) {
        mergedIndex = candidate;
        mergedGeometry = { anchorX, anchorY, unitX, unitY, mergedLength, startPosition, endPosition };
        break;
      }
    }
    if (mergedIndex >= 0) {
      const { anchorX, anchorY, unitX, unitY, mergedLength, startPosition, endPosition } = mergedGeometry;
      const lowPosition = Math.min(0.0, startPosition, endPosition);
      const highPosition = Math.max(mergedLength, startPosition, endPosition);
      mergedStarts[mergedIndex] = [anchorX + lowPosition * unitX, anchorY + lowPosition * unitY];
      mergedEnds[mergedIndex] = [anchorX + highPosition * unitX, anchorY + highPosition * unitY];
      continue;
    }
    mergedStarts.push([startX, startY]);
    mergedEnds.push([endX, endY]);
  }
  return mergedStarts.map((start, index) => Float64Array.of(start[0], start[1], mergedEnds[index][0], mergedEnds[index][1]));
}
