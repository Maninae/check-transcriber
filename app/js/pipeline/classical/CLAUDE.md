# classical/: the no-model check detector, in the browser

A faithful OpenCV.js port of the Python detector in `experiments/detection/classical/` (read its `CLAUDE.md` for the algorithm and tuning history). Same stages, same config, same output; on the 67-scene parity set every scene has the same detection count as Python and the median corner difference is 0 px.

## Contract

```js
import { detectChecksClassical, DEFAULT_CLASSICAL_DETECTOR_CONFIG } from "./detect_checks_classical.js";
const checks = detectChecksClassical(cv, imageBgr /* CV_8UC3, full res, caller owns it */, config);
// -> [{ corners: [[x, y] x4], score, sourceName, sideSupports }]
```

- `corners`: full resolution, clockwise in image coordinates (y down), pixel-center convention `(x + 0.5) * scale - 0.5`, smallest x + y corner first, after the full-resolution side refinement. Same as the Python `DetectedCheck.corners`.
- `DEFAULT_CLASSICAL_DETECTOR_CONFIG` is deep-frozen and mirrors every `ClassicalDetectorConfig` default (snake_case -> camelCase). Override with `{ ...DEFAULT_CLASSICAL_DETECTOR_CONFIG, key: value }`. Change a threshold only together with the Python config.
- Every function takes `cv` as an argument. Plain synchronous code: no DOM, no worker globals, no `async`/`await` near `cv` (see `app/js/engine_loader.js` for the thenable hang). The same files run in Node (parity) and in a Web Worker.
- Plain ES modules with relative imports; no bundler.
- Every `cv.Mat` / `cv.MatVector` created is deleted (`numeric/mat_helpers.js` `withMats`). Channel maps are JS-owned `Float32Array`s (`{ values, width, height }`), so nothing WASM-side outlives a call. Typed-array views of a Mat (`mat.data`, `mat.data32F`) detach when the WASM heap grows: read them right after the producing cv call.

## Module map

| Path | Responsibility |
|------|----------------|
| `detect_checks_classical.js` | Entry point; wires the stages, cheapest gates first (mirrors Python `detect_checks_classical.py`). |
| `classical_detector_config.js` | Frozen defaults (every Python field) and `oddKernelSize`. |
| `preprocessing/working_image_channels.js` | Working copy + the 10 float32 signal maps (lightness, chroma, Lab a/b, paper score, print residue, texture std, gradients). |
| `preprocessing/area_resize.js` | INTER_AREA downscale, arm64-exact port of OpenCV `ResizeArea_Invoker`. |
| `preprocessing/separable_filters_float32.js` | Float32 Gaussian blur and 3x3 Sobel, arm64-exact (NEON FMA order). |
| `preprocessing/neon_magnitude.js` | `cv2.magnitude`, arm64-exact (carotene HAL: ARM `FRSQRTE`/`FRECPE` estimates + Newton steps). |
| `preprocessing/combined_edge_map.js` | Canny of lightness OR amplified Lab a/b. |
| `candidates/candidate_masks.js` | The 10 candidate masks (smooth, textured, paper Otsu, edge cells, Canny cells, hole-filled Canny), one alive at a time. |
| `candidates/candidate_regions.js` | Opening, connected components, per-component contours. |
| `candidates/adjacent_cell_merging.js` | Unions of neighbouring edge cells (per-label bitmaps + exact binary closing). |
| `candidates/line_segment_extraction.js` | Texture-suppressed Canny -> Hough -> collinear merge, longest first. |
| `candidates/probabilistic_hough_lines.js` | `HoughLinesP`, arm64-exact port (fused vote, OpenCV's fixed-seed `cv::RNG`). |
| `candidates/collinear_segment_merging.js` | Greedy fusion of collinear segments. |
| `candidates/line_quadrilateral_hypotheses.js` | Parallel pairs + end caps -> budgeted rectangle hypotheses. |
| `geometry/quadrilateral_geometry.js` | Ordering, area, aspect, angles, convexity, IoU, line intersection. |
| `geometry/quadrilateral_fitting.js` | Region contour -> approxPolyN quad + Huber side refit. |
| `geometry/edge_line_snapping.js` | Snap each side to the strongest nearby step (profiles, sub-pixel peak, fitLine). |
| `geometry/full_resolution_edge_refinement.js` | Final snap on a full-res gray crop. |
| `verification/quadrilateral_verification.js` | Shape gates and per-side border evidence score. |
| `verification/interior_appearance.js` | Interior print / texture / paper / chroma gate (lazy exact median tests). |
| `verification/interior_seam_detection.js` | Rejects quads with a straight border across the interior. |
| `verification/candidate_selection.js` | Greedy NMS: duplicates, containment, union coverage, fragment floor. |
| `numeric/numpy_compatibility.js` | numpy semantics: `roundHalfEven`, `linspace`, `:g` formatting; re-exports the two below. |
| `numeric/numpy_argsort.js` | numpy's non-stable introsort argsort (tie order matters), argmax/argmin. |
| `numeric/numpy_statistics.js` | `np.median`, `np.percentile`, pairwise sum, exact median-vs-threshold test. |
| `numeric/bilinear_sampling.js` | `cv2.remap` bilinear point sampling, arm64-exact (fused lerps). |
| `numeric/binary_morphology.js` | Exact bit-packed binary erode/dilate/open/close for centered-run kernels. |
| `numeric/bit_packed_rows.js` | Pack/unpack/shift primitives for 32-pixel words. |
| `numeric/opencv_geometry_formulas.js` | `contourArea` and `isContourConvex`, ported (no Mat per call). |
| `numeric/mat_helpers.js` | Mat creation, `withMats` cleanup, `fitLineHuber`, `findExternalContours`. |

## Why some OpenCV calls are JS ports

The parity target is the Python cv2 on arm64 (OpenCV 5.0 with the KleidiCV/carotene HALs). There, clang contracts `a + b * c` into a fused multiply-add and NEON kernels use `vfma`; OpenCV.js (WASM) rounds the product separately. Most of the time that is a 1-ulp difference, but two places turn it into different detections: a uint8 truncation of an amplified Lab channel feeding Canny, and `HoughLinesP`, whose randomized point order reshuffles on a single differing edge pixel or vote. So every OpenCV call whose float result feeds a threshold or Canny was replaced by a JS port that reproduces the arm64 arithmetic, each verified bit-for-bit against cv2:

- resize INTER_AREA, float Gaussian blur, Sobel, `magnitude`, bilinear `remap`, `HoughLinesP`: 0 differing values on real scenes.
- Integer or bit-exact OpenCV paths stay on `cv.*`: `cvtColor` (Lab, gray), uint8 morphology and blur, Canny, connected components, contours, convex hull, `approxPolyN`, `fillConvexPoly`, box filters (float64 sums).
- Binary morphology was ported for speed, not exactness (WASM ellipse morphology was ~1 s per photo); it is pixel-identical to `cv.morphologyEx`.
- When you add a `cv.*` call on float data, check it against cv2 on arm64 before trusting it.

## Invariants (from the Python detector, still binding)

- Output corners are full resolution, clockwise, pixel-center convention.
- A side lying on the image border counts as supported, and such quads get the looser `borderTruncatedAspectRange`.
- Generators are generous; precision comes from verification + selection. A new generator only needs to emit `{ corners, rectangularity, sourceName }` into `collectFittedQuadrilaterals`.
- Anything that can grow combinatorially is budgeted (line pairs, caps, hypotheses).
- Sorting must reproduce numpy's order, ties included: use `numpyArgsort` wherever the Python calls `np.argsort`, and a stable JS sort wherever it calls `sorted`/`list.sort`.

## Known residual differences

- `cv.fitLine` (Huber) differs from cv2 by 1 float32 ulp in ~12% of calls (the Python side fuses a float64 `x2 - x*x`). Effect: 53% of corners are bit-identical, the rest differ by a median of ~1e-4 px, max 0.9 px on the parity set. No detection count changes.
- `cv.intersectConvexConvex` under-reports the overlap of nearly coincident quads (two quads 1e-4 px apart score IoU 0.37). Python has the same behaviour, so parity holds, but it weakens the duplicate test; the union-coverage test catches those duplicates instead.

## Parity and checks

Scene pixels and references live on vega (`/Volumes/vega/datasets/check-transcriber/tools/app-parity/classical/`). From the repo root, one process at a time:

```
/Volumes/vega/datasets/check-transcriber/venv/bin/python app/tests/parity/dump_scene_pixels.py \
    --split val --output /Volumes/vega/datasets/check-transcriber/tools/app-parity/classical/scenes val_000000 ...
PYTHONPATH=. /Volumes/vega/datasets/check-transcriber/venv/bin/python app/tests/parity/dump_classical_reference.py \
    --debug val_000000 ...
node app/tests/parity/run_classical_parity.mjs            # final_scene_ids.txt: 60 val + 7 eval scenes
node app/tests/parity/compare_classical_stages.mjs val_000032   # where does a scene first diverge?
node app/tests/parity/check_classical_wasm_leaks.mjs     # free WASM memory must stay flat
```

## Performance

Median ~1.6-2.0 s per photo in Node on the shared Mac mini under load (Python: ~0.75-0.9 s in the same conditions); a 12 MP photo costs ~0.2 s more than a 5 MP one (resize and refinement scale with resolution, everything else runs at 1600 px). Largest stages: exact `HoughLinesP` (~0.2-0.9 s, edge-count dependent; `cv.HoughLinesP` is ~2x faster but breaks count parity on ~3% of scenes), masks + regions (~0.6 s), channels (~0.4 s), verification (~0.2 s).
