/**
 * JS refinement on the synthetic test scenes: parity with Python and the Python tests' tolerances.
 *
 *   node app/tests/parity/run_refinement_synthetic_parity.mjs
 *
 * Reads what dump_refinement_synthetic_cases.py wrote; exits non-zero if any case fails.
 */

import fs from "node:fs";
import path from "node:path";
import { loadOpenCvForNode } from "./load_opencv_node.mjs";
import { refineCheckQuadrilateral } from "../../js/pipeline/refinement/quadrilateral_refinement.js";

const CASE_DIRECTORY = "/tmp/cts-parity/refinement-synthetic";
const PARITY_TOLERANCE_PIXELS = 0.01;

loadOpenCvForNode((cv) => {
  const cases = JSON.parse(fs.readFileSync(path.join(CASE_DIRECTORY, "synthetic_cases.json"), "utf8"));
  let failures = 0;
  for (const testCase of cases) {
    const image = new cv.Mat(testCase.height, testCase.width, testCase.channels === 1 ? cv.CV_8UC1 : cv.CV_8UC3);
    try {
      image.data.set(fs.readFileSync(path.join(CASE_DIRECTORY, `${testCase.name}.pixels`)));
      const { corners, diagnostics } = refineCheckQuadrilateral(cv, image, testCase.corners, undefined, testCase.other_quads);
      const parity = Math.max(...corners.map((corner, index) => Math.hypot(corner[0] - testCase.python_refined[index][0], corner[1] - testCase.python_refined[index][1])));
      const truthError = testCase.truth && Math.max(...corners.map((corner, index) => Math.hypot(corner[0] - testCase.truth[index][0], corner[1] - testCase.truth[index][1])));
      const supportAgrees = JSON.stringify(diagnostics.passes[0].sidesWithoutSupport) === JSON.stringify(testCase.python_sides_without_support);
      const passed = parity <= PARITY_TOLERANCE_PIXELS && supportAgrees && (testCase.truth === null || truthError < testCase.tolerance);
      failures += passed ? 0 : 1;
      console.log(`${passed ? "PASS" : "FAIL"} ${testCase.name.padEnd(28)} js-vs-python ${parity.toExponential(1)} px  js-vs-truth ${truthError === null ? "n/a" : truthError.toFixed(3)} (< ${testCase.tolerance})  sides-without-support agree ${supportAgrees}`);
    } finally {
      image.delete();
    }
  }
  process.exitCode = failures === 0 ? 0 : 1;
});
