/**
 * Python-vs-JS parity for the classical check detector (app/js/pipeline/classical/).
 *
 *     node app/tests/parity/run_classical_parity.mjs [--scenes-dir D] [--reference-dir D] [--output F] [scene_id ...]
 *
 * With no scene ids, runs the ids in `<parity root>/final_scene_ids.txt`. Inputs come from
 * `dump_scene_pixels.py` (raw BGR) and `dump_classical_reference.py` (Python detections).
 * Per scene: detection count equal?, and for detections matched greedily (closest first; IoU
 * >= 0.5 or near-identical corners), corner distances after the best cyclic roll. Prints a per-scene table and the
 * summary (count match rate, corner distance median / p95 / max, JS wall and CPU ms per scene
 * with the 1-minute load average, since the machine is shared; WASM heap size before and after
 * every scene, which must stay flat).
 */

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { loadOpenCvForNode, readBgrSceneMat } from "./load_opencv_node.mjs";
import { detectChecksClassical, DEFAULT_CLASSICAL_DETECTOR_CONFIG } from "../../js/pipeline/classical/detect_checks_classical.js";

const PARITY_ROOT = "/Volumes/vega/datasets/check-transcriber/tools/app-parity/classical";
const MATCH_IOU_THRESHOLD = 0.5;
const NEAR_CORNER_FRACTION_OF_DIAGONAL = 0.02;

/** Parse `--flag value` options and positional scene ids. */
function parseArguments(argv) {
  const options = { scenesDir: path.join(PARITY_ROOT, "scenes"), referenceDir: path.join(PARITY_ROOT, "python_reference"), output: null, sceneIds: [] };
  for (let index = 0; index < argv.length; index += 1) {
    if (argv[index] === "--scenes-dir") options.scenesDir = argv[++index];
    else if (argv[index] === "--reference-dir") options.referenceDir = argv[++index];
    else if (argv[index] === "--output") options.output = argv[++index];
    else options.sceneIds.push(argv[index]);
  }
  if (options.sceneIds.length === 0) {
    options.sceneIds = fs.readFileSync(path.join(PARITY_ROOT, "final_scene_ids.txt"), "utf8").split(/\s+/).filter(Boolean);
  }
  return options;
}

/** Per-corner distances between two quads after the cyclic roll of `candidate` with the smallest total. */
function cornerDistancesAfterBestRoll(reference, candidate) {
  let bestDistances = null;
  for (let roll = 0; roll < 4; roll += 1) {
    const distances = reference.map(([x, y], index) => {
      const [otherX, otherY] = candidate[(index + roll) % 4];
      return Math.hypot(x - otherX, y - otherY);
    });
    if (bestDistances === null || distances.reduce((a, b) => a + b) < bestDistances.reduce((a, b) => a + b)) bestDistances = distances;
  }
  return bestDistances;
}

/** Signed shoelace area of `[[x, y], ...]`. */
function polygonArea(points) {
  let doubledArea = 0;
  points.forEach(([x, y], index) => {
    const [nextX, nextY] = points[(index + 1) % points.length];
    doubledArea += x * nextY - nextX * y;
  });
  return doubledArea / 2;
}

/**
 * IoU of two convex quads by Sutherland-Hodgman clipping.
 *
 * Not cv.intersectConvexConvex: it under-reports the overlap of nearly coincident quads (two
 * detections 1e-4 px apart score IoU 0.37), which would hide exact matches.
 */
function convexQuadrilateralIouByClipping(firstCorners, secondCorners) {
  const orientation = Math.sign(polygonArea(secondCorners)) || 1;
  let clipped = firstCorners;
  secondCorners.forEach(([edgeStartX, edgeStartY], edgeIndex) => {
    const [edgeEndX, edgeEndY] = secondCorners[(edgeIndex + 1) % secondCorners.length];
    const inside = ([x, y]) => orientation * ((edgeEndX - edgeStartX) * (y - edgeStartY) - (edgeEndY - edgeStartY) * (x - edgeStartX)) >= 0;
    const crossing = ([x1, y1], [x2, y2]) => {
      const denominator = (edgeEndX - edgeStartX) * (y2 - y1) - (edgeEndY - edgeStartY) * (x2 - x1);
      const along = ((edgeEndX - edgeStartX) * (edgeStartY - y1) - (edgeEndY - edgeStartY) * (edgeStartX - x1)) / denominator;
      return [x1 + along * (x2 - x1), y1 + along * (y2 - y1)];
    };
    const input = clipped;
    clipped = [];
    input.forEach((point, pointIndex) => {
      const previous = input[(pointIndex + input.length - 1) % input.length];
      if (inside(point)) {
        if (!inside(previous)) clipped.push(crossing(previous, point));
        clipped.push(point);
      } else if (inside(previous)) {
        clipped.push(crossing(previous, point));
      }
    });
  });
  const intersectionArea = clipped.length >= 3 ? Math.abs(polygonArea(clipped)) : 0;
  const unionArea = Math.abs(polygonArea(firstCorners)) + Math.abs(polygonArea(secondCorners)) - intersectionArea;
  return unionArea > 0 ? intersectionArea / unionArea : 0;
}

/**
 * Greedy matching, closest pairs first (mean corner distance after the best cyclic roll).
 *
 * A pair is accepted when the quads overlap (IoU >= 0.5) or their corners agree to within
 * 2% of the reference diagonal; the second test covers near-coincident quads, where polygon
 * clipping is numerically fragile.
 */
function matchDetections(referenceDetections, jsDetections) {
  const candidatePairs = [];
  referenceDetections.forEach((reference, referenceIndex) => {
    const [[x0, y0], , [x2, y2]] = reference.corners;
    const diagonal = Math.hypot(x2 - x0, y2 - y0);
    jsDetections.forEach((detection, jsIndex) => {
      const distances = cornerDistancesAfterBestRoll(reference.corners, detection.corners);
      const meanDistance = distances.reduce((a, b) => a + b) / 4;
      const closeCorners = meanDistance <= NEAR_CORNER_FRACTION_OF_DIAGONAL * diagonal;
      if (closeCorners || convexQuadrilateralIouByClipping(reference.corners, detection.corners) >= MATCH_IOU_THRESHOLD) {
        candidatePairs.push({ referenceIndex, jsIndex, meanDistance });
      }
    });
  });
  candidatePairs.sort((first, second) => first.meanDistance - second.meanDistance);
  const usedReference = new Set();
  const usedJs = new Set();
  const matches = [];
  for (const pair of candidatePairs) {
    if (usedReference.has(pair.referenceIndex) || usedJs.has(pair.jsIndex)) continue;
    usedReference.add(pair.referenceIndex);
    usedJs.add(pair.jsIndex);
    matches.push(pair);
  }
  return matches;
}

/** Value at quantile `fraction` of a sorted copy (nearest rank). */
function quantile(values, fraction) {
  if (values.length === 0) return NaN;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor(fraction * (sorted.length - 1) + 0.5))];
}

/** WASM heap size in bytes (the buffer every Mat's typed-array view is a window on). */
function wasmHeapBytes(cv) {
  const probe = new cv.Mat(1, 1, cv.CV_8U);
  const heapBytes = probe.data.buffer.byteLength;
  probe.delete();
  return heapBytes;
}

loadOpenCvForNode((cv) => {
  const options = parseArguments(process.argv.slice(2));
  const rows = [];
  const allCornerDistances = [];
  const heapBefore = wasmHeapBytes(cv);
  for (const sceneId of options.sceneIds) {
    const reference = JSON.parse(fs.readFileSync(path.join(options.referenceDir, `${sceneId}.json`), "utf8"));
    const imageBgr = readBgrSceneMat(cv, options.scenesDir, sceneId);
    const startTime = performance.now();
    const startCpu = process.cpuUsage();
    const jsDetections = detectChecksClassical(cv, imageBgr, DEFAULT_CLASSICAL_DETECTOR_CONFIG);
    const elapsedMs = performance.now() - startTime;
    const cpuUsage = process.cpuUsage(startCpu);
    const cpuMs = (cpuUsage.user + cpuUsage.system) / 1000;
    imageBgr.delete();
    const matches = matchDetections(reference.detections, jsDetections);
    const sceneDistances = matches.flatMap(({ referenceIndex, jsIndex }) => (
      cornerDistancesAfterBestRoll(reference.detections[referenceIndex].corners, jsDetections[jsIndex].corners)
    ));
    allCornerDistances.push(...sceneDistances);
    const row = {
      sceneId,
      pythonCount: reference.detections.length,
      jsCount: jsDetections.length,
      matched: matches.length,
      maxCornerDistance: sceneDistances.length ? Math.max(...sceneDistances) : 0,
      jsMs: elapsedMs,
      jsCpuMs: cpuMs,
      pythonMs: reference.seconds * 1000,
      heapBytes: wasmHeapBytes(cv),
    };
    rows.push(row);
    console.log(
      `${sceneId}  py ${row.pythonCount}  js ${row.jsCount}  matched ${row.matched}  max corner ${row.maxCornerDistance.toFixed(3)} px`
      + `  js ${elapsedMs.toFixed(0)} ms (cpu ${cpuMs.toFixed(0)})  py ${row.pythonMs.toFixed(0)} ms  heap ${(row.heapBytes / 2 ** 20).toFixed(0)} MiB`,
    );
  }
  const countMatchRate = rows.filter((row) => row.pythonCount === row.jsCount).length / rows.length;
  const fullyMatchedRate = rows.filter((row) => row.pythonCount === row.jsCount && row.matched === row.pythonCount).length / rows.length;
  const jsTimes = rows.map((row) => row.jsMs);
  const summary = {
    scenes: rows.length,
    countMatchRate,
    fullyMatchedRate,
    matchedDetections: allCornerDistances.length / 4,
    cornerDistanceMedian: quantile(allCornerDistances, 0.5),
    cornerDistanceP95: quantile(allCornerDistances, 0.95),
    cornerDistanceMax: allCornerDistances.length ? Math.max(...allCornerDistances) : 0,
    exactCornerFraction: allCornerDistances.filter((distance) => distance < 1e-6).length / Math.max(1, allCornerDistances.length),
    jsMsMedian: quantile(jsTimes, 0.5),
    jsMsMax: Math.max(...jsTimes),
    jsCpuMsMedian: quantile(rows.map((row) => row.jsCpuMs), 0.5),
    jsCpuMsMax: Math.max(...rows.map((row) => row.jsCpuMs)),
    loadAverage: os.loadavg()[0],
    pythonMsMedian: quantile(rows.map((row) => row.pythonMs), 0.5),
    heapBytesBefore: heapBefore,
    heapBytesAfter: wasmHeapBytes(cv),
  };
  console.log(JSON.stringify(summary, null, 2));
  if (options.output) fs.writeFileSync(options.output, JSON.stringify({ summary, rows }, null, 2));
});
