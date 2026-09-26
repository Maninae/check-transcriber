# refinement/: sub-pixel corner refinement of detected check quads

Takes a full-resolution photo and an approximate quad (a YOLO oriented box, a classical contour quad) and moves each corner onto the paper's physical corner, so the perspective warp for OCR is exact. Only uses operations available in OpenCV.js (`remap`) plus small-array numpy math, so it ports to the browser.

Entry points: `refine_check_quadrilateral(image, corners, config) -> (corners, diagnostics)` and `refine_detected_checks(image, list[DetectedCheck], config)` in `quadrilateral_refinement.py`. Corner order is preserved (corners only move). Evaluate with `python -m experiments.detection.refinement.evaluate_refinement --split val --perturbation obb|jitter|scale [--amount N] [--predictions file.json]`; results go to `/Volumes/vega/datasets/check-transcriber/experiments/detection/refinement/<split>__<input>__n=<scenes>__<tag>/`.

## Modules

| Module | Responsibility |
|---|---|
| `quadrilateral_refinement.py` | Coordinator: paper colour, passes, per-side fit, corner intersection, guard rails |
| `refinement_config.py` | `CornerRefinementConfig`, every knob with its tuned default |
| `edge_profile_sampling.py` | One `remap` per side → (sample × normal offset) edge-score grid (`two_class` paperness step or `paper_distance`) |
| `side_line_search.py` | Pass 1: Radon-style line integral over angles, OUTERMOST strong line wins; sub-pixel peak extraction near a curve |
| `robust_side_curve_fitting.py` | Offset-vs-position polynomial per side (RANSAC + Tukey IRLS), Newton intersection of adjacent curves |
| `side_edge_tracking.py` | Optional final pass (`final_pass_mode="track"`): Viterbi edge path, corner-local fits where the edge bends |
| `detector_input_simulation.py` | Simulated detector inputs from GT (`obb`, `jitter`, `scale`) and corner error |
| `refinement_evaluation_records.py` | Per-scene worker, prediction-to-GT IoU matching, summaries and breakdown groups |
| `evaluate_refinement.py` | CLI: parallel eval (≤2 workers, shared machine), summary.md/json, records.jsonl, debug panels |
| `debug_crop_rendering.py` | 2×2 panels of 3× zoomed corner crops: input red, refined green, GT blue |

## Invariants and lessons (each cost a val run to learn)

- Pick the OUTERMOST strong line, not the strongest: checks carry printed borders, rules and text a few px inside the paper edge that out-score a faint paper edge.
- Clip scores to [-c, +c] before integrating. A positive-only clip turns zero-mean background texture into fake lines, which the outermost rule then selects.
- Search mostly inward (`outward_band` ≈ 1% of the short side): an enclosing box's sides are at or outside the paper, and a wide outward band finds neighbouring checks and background stripes.
- Pass 2 only grows the pass-1 curve (no second line search), otherwise it can jump a second time to an outward edge.
- `two_class` paperness (per-sample background colour from the profile's outer end, paper colour from the quad interior, projected and clipped to [0, 1]) is what makes faint white-on-white edges competitive with printed ink.
- Sample the uint8 image directly with `remap`; converting a float crop per check dominated runtime.
- A printed line inside the 4 px inner window biases the edge outward by up to ~0.8 px; synthetic tests draw the border 8 px inside like real checks.
- Tune on val only. The eval split is scored once per final configuration.
