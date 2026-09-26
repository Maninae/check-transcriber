/**
 * Stage-by-stage diff of the JS classical detector against a Python `--debug` reference, to find
 * where a scene's outputs first diverge.
 *
 *     node app/tests/parity/compare_classical_stages.mjs [--reference-dir D] [--scenes-dir D] scene_id ...
 *
 * Needs references written with `dump_classical_reference.py --debug`. Reports, per scene:
 * mask nonzero counts and region counts per mask, fitted-quad count and the largest corner
 * difference over index-aligned fitted quads, line segments (count, max endpoint difference),
 * hypotheses (count, first differing index), verified candidates (count, sources, max corner
 * difference) and the selected set.
 */

import fs from "node:fs";
import path from "node:path";
import { loadOpenCvForNode, readBgrSceneMat } from "./load_opencv_node.mjs";
import { DEFAULT_CLASSICAL_DETECTOR_CONFIG as config, verifyFittedQuadrilateral } from "../../js/pipeline/classical/detect_checks_classical.js";
import { forEachCandidateMask } from "../../js/pipeline/classical/candidates/candidate_masks.js";
import { extractRegionsFromMask } from "../../js/pipeline/classical/candidates/candidate_regions.js";
import { buildLineQuadrilateralHypotheses } from "../../js/pipeline/classical/candidates/line_quadrilateral_hypotheses.js";
import { extractLineSegments } from "../../js/pipeline/classical/candidates/line_segment_extraction.js";
import { fitQuadrilateralToContour } from "../../js/pipeline/classical/geometry/quadrilateral_fitting.js";
import { buildWorkingImageChannels } from "../../js/pipeline/classical/preprocessing/working_image_channels.js";
import { selectNonOverlappingCandidates } from "../../js/pipeline/classical/verification/candidate_selection.js";

const PARITY_ROOT = "/Volumes/vega/datasets/check-transcriber/tools/app-parity/classical";

/** Largest absolute difference between two equally shaped nested number arrays. */
function maximumDifference(first, second) {
  if (Array.isArray(first) || ArrayBuffer.isView(first)) {
    let largest = 0;
    for (let index = 0; index < first.length; index += 1) largest = Math.max(largest, maximumDifference(first[index], second[index]));
    return largest;
  }
  return Math.abs(first - second);
}

/** Compare one list of corner sets index by index; returns { count, firstMismatchIndex, maximumDifference }. */
function compareCornerLists(pythonList, jsList, tolerance = 1e-3) {
  let firstMismatchIndex = -1;
  let largest = 0;
  for (let index = 0; index < Math.min(pythonList.length, jsList.length); index += 1) {
    const difference = maximumDifference(pythonList[index], jsList[index]);
    largest = Math.max(largest, difference);
    if (firstMismatchIndex < 0 && difference > tolerance) firstMismatchIndex = index;
  }
  return { pythonCount: pythonList.length, jsCount: jsList.length, firstMismatchIndex, maximumDifference: largest };
}

loadOpenCvForNode((cv) => {
  const argv = process.argv.slice(2);
  let referenceDir = path.join(PARITY_ROOT, "python_reference");
  let scenesDir = path.join(PARITY_ROOT, "scenes");
  const sceneIds = [];
  for (let index = 0; index < argv.length; index += 1) {
    if (argv[index] === "--reference-dir") referenceDir = argv[++index];
    else if (argv[index] === "--scenes-dir") scenesDir = argv[++index];
    else sceneIds.push(argv[index]);
  }
  const aspectRange = [
    Math.min(config.minimumAspectRatio, config.borderTruncatedAspectRange[0]),
    Math.max(config.maximumAspectRatio, config.borderTruncatedAspectRange[1]),
  ];
  for (const sceneId of sceneIds) {
    const debug = JSON.parse(fs.readFileSync(path.join(referenceDir, `${sceneId}.json`), "utf8")).debug;
    const imageBgr = readBgrSceneMat(cv, scenesDir, sceneId);
    const channels = buildWorkingImageChannels(cv, imageBgr, config);
    imageBgr.delete();
    console.log(`== ${sceneId}`);
    const jsFitted = [];
    const regionFitted = [];
    forEachCandidateMask(cv, channels, config, (name, maskMat) => {
      const nonzero = cv.countNonZero(maskMat);
      const regions = extractRegionsFromMask(cv, maskMat, name, channels, config);
      const flag = nonzero !== debug.mask_nonzero_counts[name] || regions.length !== debug.regions_per_mask[name] ? "  <-- differs" : "";
      console.log(`  mask ${name}: nonzero ${nonzero}/${debug.mask_nonzero_counts[name]} regions ${regions.length}/${debug.regions_per_mask[name]}${flag}`);
      for (const region of regions) {
        const fitted = fitQuadrilateralToContour(cv, region.contour, region.sourceName, config.minimumRegionRectangularity, aspectRange);
        regionFitted.push(fitted === null ? null : fitted.corners);
        if (fitted !== null && fitted.rectangularity >= config.minimumRegionRectangularity) jsFitted.push(fitted);
      }
    });
    const pythonRegionFitted = debug.fitted_quads.map((quad) => quad.corners);
    const bothFitted = pythonRegionFitted.map((corners, index) => [corners, regionFitted[index]]).filter(([a, b]) => a && b);
    console.log("  fitted (all regions):", compareCornerLists(bothFitted.map(([a]) => a), bothFitted.map(([, b]) => b)),
      "null mismatches", pythonRegionFitted.filter((corners, index) => (corners === null) !== (regionFitted[index] === null || regionFitted[index] === undefined)).length);
    const segments = extractLineSegments(cv, channels, config);
    console.log("  segments:", compareCornerLists(debug.line_segments, segments.map((segment) => Array.from(segment))));
    const hypotheses = buildLineQuadrilateralHypotheses(segments, channels.longSidePixels, config);
    console.log("  hypotheses:", compareCornerLists(debug.line_hypotheses, hypotheses));
    for (const corners of hypotheses) jsFitted.push({ corners, rectangularity: config.lineHypothesisRectangularity, sourceName: "lines" });
    const verified = jsFitted.map((quad) => verifyFittedQuadrilateral(cv, quad, channels, config)).filter(Boolean);
    console.log("  verified:", compareCornerLists(debug.verified_candidates.map((c) => c.corners), verified.map((c) => c.corners)));
    const pythonSources = debug.verified_candidates.map((c) => c.source_name).join(",");
    const jsSources = verified.map((c) => c.sourceName).join(",");
    if (pythonSources !== jsSources) console.log(`  verified sources differ:\n    py ${pythonSources}\n    js ${jsSources}`);
    const selected = selectNonOverlappingCandidates(cv, verified, config, channels.width, channels.height);
    console.log("  selected:", compareCornerLists(debug.selected_candidates.map((c) => c.corners), selected.map((c) => c.corners)),
      "sources py", debug.selected_candidates.map((c) => c.source_name).join(","), "js", selected.map((c) => c.sourceName).join(","));
  }
});
