/**
 * WASM heap leak check for the classical detector: run scenes in one process and watch the heap.
 *
 *     node app/tests/parity/check_classical_wasm_leaks.mjs [scene_id ...]
 *
 * OpenCV.js exposes no live-object counter, so after each scene this measures free WASM memory
 * directly: it `_malloc`s 256 KiB blocks until the free space below the initial heap top is
 * used up, counts the blocks that fit there, and frees them all. A leak of N bytes per scene
 * lowers that count by about N / 256 KiB every scene; a flat count means nothing leaked.
 * (A single large probe block is not enough: dlmalloc tucks leaked blocks into lower free space.)
 */

import path from "node:path";
import { loadOpenCvForNode, readBgrSceneMat } from "./load_opencv_node.mjs";
import { detectChecksClassical } from "../../js/pipeline/classical/detect_checks_classical.js";

const SCENES_DIRECTORY = "/Volumes/vega/datasets/check-transcriber/tools/app-parity/classical/scenes";
const PROBE_BLOCK_BYTES = 256 * 2 ** 10;
const PROBE_BLOCK_LIMIT = 1200; // 300 MiB of probes: enough to exhaust the space below the initial top
const DEFAULT_SCENE_IDS = Array.from({ length: 20 }, (_, index) => `val_${String(index).padStart(6, "0")}`);

/** WASM heap size and the number of 256 KiB blocks that still fit below `initialHeapBytes`. */
function heapSnapshot(cv, initialHeapBytes) {
  const probeAddresses = [];
  let blocksBelowInitialTop = 0;
  for (let block = 0; block < PROBE_BLOCK_LIMIT; block += 1) {
    const address = cv._malloc(PROBE_BLOCK_BYTES);
    if (address === 0) break;
    probeAddresses.push(address);
    if (address + PROBE_BLOCK_BYTES <= initialHeapBytes) blocksBelowInitialTop += 1;
  }
  for (const address of probeAddresses) cv._free(address);
  const probeMat = new cv.Mat(1, 1, cv.CV_8U);
  const heapBytes = probeMat.data.buffer.byteLength;
  probeMat.delete();
  return { heapMiB: heapBytes / 2 ** 20, freeMiBBelowInitialTop: (blocksBelowInitialTop * PROBE_BLOCK_BYTES) / 2 ** 20 };
}

loadOpenCvForNode((cv) => {
  const sceneIds = process.argv.length > 2 ? process.argv.slice(2) : DEFAULT_SCENE_IDS;
  const initialProbe = new cv.Mat(1, 1, cv.CV_8U);
  const initialHeapBytes = initialProbe.data.buffer.byteLength;
  initialProbe.delete();
  console.log("before", JSON.stringify(heapSnapshot(cv, initialHeapBytes)));
  for (const sceneId of sceneIds) {
    const imageBgr = readBgrSceneMat(cv, path.join(SCENES_DIRECTORY), sceneId);
    const detections = detectChecksClassical(cv, imageBgr);
    imageBgr.delete();
    console.log(sceneId, `${detections.length} checks`, JSON.stringify(heapSnapshot(cv, initialHeapBytes)));
  }
});
