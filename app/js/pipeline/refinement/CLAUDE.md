# refinement/: sub-pixel corner refinement of detected check quads (OpenCV.js)

Faithful port of `experiments/detection/refinement/` (read its `CLAUDE.md` for the algorithm's lessons). Takes the full-resolution photo and approximate quads (YOLO boxes, classical contour quads) and moves each corner onto the paper's physical corner, so the perspective warp is exact. Measured parity with the Python on 662 real checks: median corner difference 0.0000 px, worst 0.0002 px, 90-97% of corners bit-identical, identical corner error vs GT.

## Contract

```js
import { DEFAULT_CORNER_REFINEMENT_CONFIG, refineCheckQuadrilateral, refineDetectedChecks } from "./quadrilateral_refinement.js";
refineCheckQuadrilateral(cv, imageBgr, corners, config?, otherQuads?)  // -> { corners, diagnostics }
refineDetectedChecks(cv, imageBgr, cornerSets, config?)                // -> refined corner sets, same order
```

- `imageBgr`: `cv.Mat` CV_8UC3 (CV_8UC1 also works), full resolution, owned by the caller. Corners are `[[x, y] x4]`, either winding; the corner ORDER is preserved (corners only move).
- `cv` is always an argument: no globals, no DOM, no async anywhere (awaiting OpenCV.js's thenable hangs the browser). The same files run in Node (parity) and in a Web Worker.
- Config keys are the Python `CornerRefinementConfig` fields in camelCase with identical defaults. `randomGeneratorKind` (not in the Python) is a measurement hook only.

## Module map

| Module | Responsibility |
|---|---|
| `quadrilateral_refinement.js` | Coordinator: bands, passes, per-corner curve pairs, intersections, `refineDetectedChecks` |
| `side_refinement.js` | One side: score + mask, line search then curve growing, narrow-first, tracking-pass local fits |
| `edge_profile_sampling.js` | One `remap` per side into (sample x normal offset) profiles, then the score grid (`SideScoreProfiles`) |
| `edge_feature_scoring.js` | Two-class paperness / paper distance features, box-window scores, contact-line valley scores |
| `side_line_search.js` | Pass-1 Radon line search (OUTERMOST strong line, keep-input gate), sub-pixel peaks near a curve |
| `robust_side_curve_fitting.js` | RANSAC + Tukey IRLS polynomial fit (Householder QR), Newton corner intersection |
| `side_curve.js` | `SideCurve`: offset = poly(normalised position) in the side's frame |
| `side_edge_tracking.js` | Viterbi edge path for the final pass |
| `overlap_masking.js` | Nearby-quad selection, dilation, point-in-quad, masking other detections' cells |
| `paper_colour_estimation.js` | Median paper colour over a 24x24 grid inside the quad |
| `quadrilateral_geometry.js` | Short side, signed area, convexity, guard rails vs the input quad |
| `image_sampling.js` | The only `cv.Mat` code: bilinear `remap` with border replicate, output copied off the heap |
| `seeded_random_generator.js` | numpy-exact `default_rng(seed).integers` (SeedSequence + PCG64 + Lemire), mulberry32 for measurement |
| `numpy_compatible_math.js` | numpy semantics: linspace/arange fill rules, half-to-even round, median, argmax, `np.hypot` |
| `refinement_config.js` | `DEFAULT_CORNER_REFINEMENT_CONFIG` (frozen) |

## Invariants (parity depends on them; each was measured)

- RANSAC must use the numpy-exact generator. With mulberry32 the median is still 0 but 4% of corners move > 0.1 px and the worst moves 23 px (hypotheses decide between competing lines); PCG64 via BigInt costs nothing measurable.
- `np.hypot` here is `sqrt(fma(small, small, big^2))`, bit-exact on 20k pairs; `Math.hypot` differs from it in ~40% of cases. Side lengths feed every sample position. `np.linalg.norm` is plain `sqrt(x*x + y*y)` (`vectorNorm`).
- Emulate numpy float32 exactly where the Python is float32: remap output means, background-window means and luminance are float32 with left-to-right sums (`Math.fround` after each op). Everything else is float64.
- Build remap maps in float64 in numpy's operation order, then store to Float32Array: OpenCV.js 5.0.0 then returns bit-identical pixels.
- Line shifts use `roundHalfToEven` (np.round), argmax/argmin take the FIRST extreme, `floatArange` fills like numpy (`start + i * (a[1] - a[0])`).
- Every `cv.Mat` is deleted in `finally`; read `mat.data` once and copy (no `ucharPtr` loops). The WASM heap stays flat across all parity scenes.
- Known knife-edge (in the Python itself): the corner-local fit keeps points with `position / length >= 1 - 0.2`, and a 95-sample side puts sample 77 at exactly 0.8. Which side of 0.8 it lands on depends on the last bit of the pass-1 corners (numpy's SVD lstsq + FMA vs our QR), so that one point can flip; it was seen once in 662 checks (0.27 px on one corner).
- The algorithm lessons (outermost line, two-sided clip, narrow-first, keep-input gate, overlap masking with 0 px dilation, tent paperness, contact line below 25 units, pass 2 only grows the pass-1 curve) live in `experiments/detection/refinement/CLAUDE.md`; change the Python first, then port.

## Parity

Run from the repository root, one process at a time (shared machine):

```
PYTHONPATH=. <venv>/bin/python app/tests/parity/dump_refinement_reference.py --input-name yolo_val   # also classical_val, classical_eval
node app/tests/parity/run_refinement_parity.mjs [--inputs yolo_val,classical_val,classical_eval] [--rng mulberry32] [--limit N]
PYTHONPATH=. <venv>/bin/python app/tests/parity/dump_refinement_synthetic_cases.py
node app/tests/parity/run_refinement_synthetic_parity.mjs
```

References and scene pixels live in `/Volumes/vega/datasets/check-transcriber/tools/app-parity/refinement/` (scene pixels: `dump_scene_pixels.py --output .../refinement/scenes`). `<venv>` = `/Volumes/vega/datasets/check-transcriber/venv/bin/python`.
