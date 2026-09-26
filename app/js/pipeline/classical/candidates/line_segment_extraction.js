/**
 * Straight edge segments for the line-based hypothesis generator (Python
 * `line_segment_extraction.py`).
 *
 * Combined Canny (lightness OR color), minus edges deep inside textured surfaces (a min filter
 * of the texture map stays high inside carpet or crochet but is low next to any smooth paper),
 * then `HoughLinesP` (the arm64-exact JS port, `probabilistic_hough_lines.js`), then a greedy merge of collinear pieces so a border broken by a shadow
 * comes back as one segment. Segments are `Float64Array`s `[x1, y1, x2, y2]`, longest first.
 * Every sort uses numpy's argsort so equal lengths keep the Python's order.
 */

import { oddKernelSize } from "../classical_detector_config.js";
import { buildCombinedCannyEdges } from "../preprocessing/combined_edge_map.js";
import { createFloat32MatFromValues, withMats } from "../numeric/mat_helpers.js";
import { numpyArgsort } from "../numeric/numpy_compatibility.js";
import { mergeCollinearSegments, segmentLength } from "./collinear_segment_merging.js";
import { houghLinesProbabilistic } from "./probabilistic_hough_lines.js";

const HOUGH_ANGLE_RESOLUTION_RADIANS = Math.PI / 360;
const HOUGH_DISTANCE_RESOLUTION_PIXELS = 1;
const HOUGH_VOTE_FRACTION_OF_MINIMUM_LENGTH = 0.6;
const HOUGH_PIECE_FRACTION_OF_MINIMUM_LENGTH = 0.6; // pieces this short may still merge into a long line
const MAXIMUM_RAW_SEGMENTS_TO_MERGE = 400;
const fround = Math.fround;

/** Segments reordered longest first (numpy `argsort(-lengths)`), truncated to `limit`. */
export function sortSegmentsLongestFirst(segments, limit = segments.length) {
  const order = numpyArgsort(segments.map((segment) => -segmentLength(segment)));
  return Array.from(order.subarray(0, Math.min(limit, order.length)), (index) => segments[index]);
}

/** Canny edges with texture-interior edges removed, then HoughLinesP; raw integer segments. */
function detectRawSegments(cv, channels, config) {
  const { width, height, longSidePixels } = channels;
  const minimumLength = config.lineMinimumLengthFraction * longSidePixels;
  return withMats((track) => {
    const edges = track(buildCombinedCannyEdges(cv, channels, config.lineCannyLow, config.lineCannyHigh, config));
    const textureWindow = oddKernelSize(config.textureWindowFraction, longSidePixels);
    const textureMat = track(createFloat32MatFromValues(cv, channels.textureStd.values, width, height));
    const erosionKernel = track(cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(2 * textureWindow + 1, 2 * textureWindow + 1)));
    const nearestSmoothTexture = track(new cv.Mat());
    cv.erode(textureMat, nearestSmoothTexture, erosionKernel);
    const smoothestNearby = nearestSmoothTexture.data32F;
    const edgeBytes = edges.data;
    const suppressionThreshold = fround(config.lineTextureSuppressionStd);
    for (let index = 0; index < width * height; index += 1) {
      if (smoothestNearby[index] > suppressionThreshold) edgeBytes[index] = 0;
    }
    const rawSegments = houghLinesProbabilistic(
      edgeBytes, width, height, HOUGH_DISTANCE_RESOLUTION_PIXELS, HOUGH_ANGLE_RESOLUTION_RADIANS,
      Math.trunc(minimumLength * HOUGH_VOTE_FRACTION_OF_MINIMUM_LENGTH),
      minimumLength * HOUGH_PIECE_FRACTION_OF_MINIMUM_LENGTH,
      config.lineMaximumGapFraction * longSidePixels,
    ).map((segment) => Float64Array.from(segment));
    return rawSegments;
  });
}

/** Long straight edge segments, longest first, capped at `lineMaximumSegments`. */
export function extractLineSegments(cv, channels, config) {
  const minimumLength = config.lineMinimumLengthFraction * channels.longSidePixels;
  const rawSegments = detectRawSegments(cv, channels, config);
  if (rawSegments.length === 0) return [];
  const longestRawSegments = sortSegmentsLongestFirst(rawSegments, MAXIMUM_RAW_SEGMENTS_TO_MERGE);
  const mergedSegments = mergeCollinearSegments(longestRawSegments, config.lineMergeGapFraction * channels.longSidePixels);
  const longEnoughSegments = mergedSegments.filter((segment) => segmentLength(segment) >= minimumLength);
  return sortSegmentsLongestFirst(longEnoughSegments, config.lineMaximumSegments);
}
