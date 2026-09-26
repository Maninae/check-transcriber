/**
 * JS-vs-Python parity (and JS-vs-GT accuracy) of app/js/pipeline/refinement/.
 *
 *   node app/tests/parity/run_refinement_parity.mjs [--inputs yolo_val,classical_val,classical_eval]
 *        [--rng numpy_pcg64|mulberry32] [--limit N] [--worst K]
 *
 * Reads `reference__<input>.json` (dump_refinement_reference.py) and the matching scene
 * pixels (dump_scene_pixels.py) from REFERENCE_DIRECTORY, runs refineDetectedChecks on the
 * identical inputs, and prints per input set:
 * - JS-vs-Python per-corner distance: median / p95 / max, share within 0.1 px, share bit-equal;
 * - corner error vs GT (best cyclic roll, in-frame corners) for input, Python and JS;
 * - diagnostics agreement (quad reverted, narrow-band acceptance), ms per check, WASM heap.
 */

import fs from "node:fs";
import path from "node:path";
import { loadOpenCvForNode, readBgrSceneMat } from "./load_opencv_node.mjs";
import { DEFAULT_CORNER_REFINEMENT_CONFIG, refineCheckQuadrilateral } from "../../js/pipeline/refinement/quadrilateral_refinement.js";

const REFERENCE_DIRECTORY = "/Volumes/vega/datasets/check-transcriber/tools/app-parity/refinement";
const SCENE_DIRECTORY = path.join(REFERENCE_DIRECTORY, "scenes");
const WITHIN_TOLERANCE_PIXELS = 0.1;

/**
 * { heapMegabytes, probeOffsetMegabytes }: WASM memory size (never shrinks) and where a fresh
 * 32 MB Mat lands. This build exposes no HEAPU8, so a probe Mat's buffer stands in; leaked
 * Mats would push the probe higher and grow the heap across scenes.
 */
function measureWasmHeap(cv) {
  const probe = new cv.Mat(4096, 8192, cv.CV_8UC1);
  try {
    return { heapMegabytes: probe.data.buffer.byteLength / 2 ** 20, probeOffsetMegabytes: probe.data.byteOffset / 2 ** 20 };
  } finally {
    probe.delete();
  }
}

/** Parse --key value flags. */
function parseArguments(argv) {
  const options = { inputs: "yolo_val,classical_val,classical_eval", rng: "numpy_pcg64", limit: Infinity, worst: 5 };
  for (let index = 0; index < argv.length; index += 2) options[argv[index].replace(/^--/, "")] = argv[index + 1];
  return { ...options, inputs: options.inputs.split(","), limit: Number(options.limit), worst: Number(options.worst) };
}

/** numpy.percentile (linear interpolation) of a numeric array. */
function percentile(values, percent) {
  if (values.length === 0) return NaN;
  const sorted = Float64Array.from(values).sort();
  const rank = (percent / 100) * (sorted.length - 1);
  const lower = Math.floor(rank);
  const upper = Math.min(lower + 1, sorted.length - 1);
  return sorted[lower] + (rank - lower) * (sorted[upper] - sorted[lower]);
}

/** Per-corner errors of the in-frame GT corners after the best cyclic roll (lowest mean). */
function inFrameCornerErrors(predicted, groundTruth) {
  let best = null;
  for (let shift = 0; shift < 4; shift += 1) {
    const errors = [];
    groundTruth.corners.forEach((corner, index) => {
      if (!groundTruth.in_frame_mask[index]) return;
      const candidate = predicted[(index + shift) % 4];
      errors.push(Math.hypot(candidate[0] - corner[0], candidate[1] - corner[1]));
    });
    const mean = errors.reduce((sum, error) => sum + error, 0) / errors.length;
    if (best === null || mean < best.mean) best = { mean, errors };
  }
  return best.errors;
}

/** Four-number summary string. */
function summarize(values, digits = 3) {
  const format = (value) => value.toFixed(digits);
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  return `median ${format(percentile(values, 50))}  mean ${format(mean)}  p95 ${format(percentile(values, 95))}  max ${format(Math.max(...values))}`;
}

/** Run one input set and print its report; returns the per-check timings. */
function runInputSet(cv, inputName, options) {
  const reference = JSON.parse(fs.readFileSync(path.join(REFERENCE_DIRECTORY, `reference__${inputName}.json`), "utf8"));
  const config = { ...DEFAULT_CORNER_REFINEMENT_CONFIG, randomGeneratorKind: options.rng };
  const parityDistances = [];
  const errors = { input: [], python: [], js: [] };
  const worstChecks = [];
  let bitEqualCorners = 0;
  let revertedAgreement = 0;
  let narrowAgreement = 0;
  let numberOfChecks = 0;
  let jsMilliseconds = 0;
  let pythonMilliseconds = 0;
  const heapAtStart = measureWasmHeap(cv);
  let heapAfterFirstScene = null;
  const sceneIds = Object.keys(reference.scenes).slice(0, options.limit);
  for (const sceneId of sceneIds) {
    const scene = reference.scenes[sceneId];
    const image = readBgrSceneMat(cv, SCENE_DIRECTORY, sceneId);
    try {
      const startTime = performance.now();
      const results = scene.input_quads.map((corners, checkIndex) => refineCheckQuadrilateral(
        cv, image, corners, config, scene.input_quads.filter((_, otherIndex) => otherIndex !== checkIndex),
      ));
      jsMilliseconds += performance.now() - startTime;
      pythonMilliseconds += scene.python_milliseconds;
      results.forEach(({ corners, diagnostics }, checkIndex) => {
        numberOfChecks += 1;
        const pythonCorners = scene.python_refined[checkIndex];
        const distances = corners.map((corner, index) => Math.hypot(corner[0] - pythonCorners[index][0], corner[1] - pythonCorners[index][1]));
        parityDistances.push(...distances);
        bitEqualCorners += corners.filter((corner, index) => corner[0] === pythonCorners[index][0] && corner[1] === pythonCorners[index][1]).length;
        const pythonDiagnostics = scene.python_diagnostics[checkIndex];
        revertedAgreement += Number(diagnostics.quadReverted === pythonDiagnostics.quad_reverted);
        narrowAgreement += Number(JSON.stringify(diagnostics.passes[0].sidesAcceptedInNarrowBand) === JSON.stringify(pythonDiagnostics.sides_accepted_in_narrow_band));
        worstChecks.push({ sceneId, checkIndex, maximum: Math.max(...distances) });
        const groundTruth = scene.ground_truth[checkIndex];
        if (groundTruth === null) return;
        errors.input.push(...inFrameCornerErrors(scene.input_quads[checkIndex], groundTruth));
        errors.python.push(...inFrameCornerErrors(pythonCorners, groundTruth));
        errors.js.push(...inFrameCornerErrors(corners, groundTruth));
      });
    } finally {
      image.delete();
    }
    heapAfterFirstScene ??= measureWasmHeap(cv);
  }
  const withinTolerance = parityDistances.filter((distance) => distance <= WITHIN_TOLERANCE_PIXELS).length / parityDistances.length;
  console.log(`\n== ${inputName} (${sceneIds.length} scenes, ${numberOfChecks} checks, rng ${options.rng}) ==`);
  console.log(`JS vs Python corner distance px: ${summarize(parityDistances, 4)}`);
  console.log(`  within ${WITHIN_TOLERANCE_PIXELS} px: ${(100 * withinTolerance).toFixed(1)}%   bit-equal: ${(100 * bitEqualCorners / parityDistances.length).toFixed(1)}%`);
  console.log(`  quad_reverted agree: ${revertedAgreement}/${numberOfChecks}   narrow-band acceptance agree: ${narrowAgreement}/${numberOfChecks}`);
  console.log(`GT corner error px (${errors.js.length} in-frame corners of matched checks):`);
  console.log(`  input : ${summarize(errors.input)}`);
  console.log(`  python: ${summarize(errors.python)}`);
  console.log(`  js    : ${summarize(errors.js)}`);
  console.log(`ms per check: js ${(jsMilliseconds / numberOfChecks).toFixed(1)}   python ${(pythonMilliseconds / numberOfChecks).toFixed(1)}`);
  const heapAtEnd = measureWasmHeap(cv);
  const describeHeap = (heap) => `${heap.heapMegabytes.toFixed(0)} (probe at ${heap.probeOffsetMegabytes.toFixed(1)})`;
  console.log(`WASM heap MB: start ${describeHeap(heapAtStart)}  after first scene ${describeHeap(heapAfterFirstScene)}  end ${describeHeap(heapAtEnd)}`);
  worstChecks.sort((first, second) => second.maximum - first.maximum);
  console.log(`worst checks: ${worstChecks.slice(0, options.worst).map((check) => `${check.sceneId}#${check.checkIndex} ${check.maximum.toFixed(3)}`).join(", ")}`);
}

const options = parseArguments(process.argv.slice(2));
loadOpenCvForNode((cv) => {
  for (const inputName of options.inputs) runInputSet(cv, inputName, options);
});
