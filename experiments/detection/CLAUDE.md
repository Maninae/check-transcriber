# experiments/detection: agent map

Check detection for Check Transcriber: photo in, one quadrilateral per check out (corners TL, TR, BR, BL of the check itself, full-resolution pixels). Read `REPORT.md` for the current numbers and recommendation, and `PLAN.md` for the build log.

## Contracts (everything plugs into these)

- Ground truth: `dataset/scene_annotations.py` loads the v1 synthetic scenes from vega (`SceneAnnotation`, `CheckAnnotation`).
- Detector output: `predictions/detected_check.py` (`DetectedCheck`, one predictions JSON per run). Corners are clockwise; `orientation_known=True` means corner 0 is the check's own top-left.
- Paths: `config/paths.py`. Heavy files (data, runs, weights, images) never go on the internal disk.

## Packages (dependencies flow downward: tools → stages → contracts)

| Folder | Responsibility | Own CLAUDE.md |
|---|---|---|
| `metrics/` | IoU matching, corner error, orientation, breakdowns, report CLI | yes |
| `classical/` | OpenCV detector (the milestone-2 detector; must stay OpenCV.js-portable) | yes |
| `learned/` | YOLO dataset copies, training, inference wrappers | no |
| `learned/centernet/` | permissive CenterNet-style detector (trained on v1 + close-up; best counter, coarse corners) | yes |
| `refinement/` | edge-based corner refinement for any detector's quads (OpenCV.js-portable) | yes |
| `orientation/` | upside-down classifier + corner reordering for unordered detections | no |
| `pipeline/` | post-processing for any predictions file: duplicate suppression, hybrid close-up fit, refine, orient; hybrid tuner | no |
| `export/` | ONNX export, parity and latency; `BROWSER_NOTE.md` | no |
| `visualization/` | overlays of hard scenes and failure galleries | no |

## Rules

- Tune on val; score eval once per frozen pipeline and record it in `REPORT.md`.
- One GPU training job at a time on the shared 16 GB Mac. YOLO at 1024 needs batch 4 (about 3.5 GB); batch 8 thrashed swap.
- Ultralytics is AGPL-3.0; anything trained with it inherits that (see REPORT.md, "AGPL note").
- Datasets: scene ids repeat across synthetic sets (every set has `val_000000`); never pool scenes from two sets into one scene-id-keyed dict. Pick a set with `dataset_name=` or `CHECK_DETECTION_DATASET_ROOT`.
- Ship YOLO's one-to-many head plus rotated NMS, not the `nms=False` one-to-one head (measured worse on val).
- `refinement/` and `classical/` must only use operations OpenCV.js exports.
