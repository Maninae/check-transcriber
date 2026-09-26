# Check detection: plan and log

Goal: from a phone photo of several checks on a household surface, return every check as a precise quadrilateral with its corners in the check's own TL/TR/BR/BL order. Owner: detector branch `owner/detector`. Data: v1 synthetic set on vega (train 3,500 / val 750 / eval 750 scenes, held out by template and background). Eval is scored once per final detector; all tuning and model selection happen on val.

## Contracts (fixed before fan-out)

- Ground truth: `dataset/scene_annotations.py` (`SceneAnnotation`, `CheckAnnotation`, `load_split_scene_annotations`).
- Detector output: `predictions/detected_check.py` (`DetectedCheck` with `corners` (4,2) full-res pixels, clockwise, `score`, `orientation_known`; one predictions JSON per run).
- Paths: `config/paths.py`. Runs, weights and images go under `/Volumes/vega/datasets/check-transcriber/experiments/detection/`.

## Units

| # | Unit | Verified means |
|---|------|----------------|
| U1 | Metrics harness (`metrics/`) | pytest on hand-built cases (perfect, shifted, rotated order, missed, duplicate, out-of-frame); a GT-as-prediction run scores IoU≈1 / corner error≈0 on val |
| U2 | Classical OpenCV baseline (`classical/`) | tuned on val only, eval scored once; overlays read by eye; failure modes written down |
| U3 | YOLO-OBB nano via Ultralytics (`learned/`) | trains on v1 train at 1024, selected on val, scored once on eval |
| U4 | Corner refinement (`refinement/`) + upright/flipped classifier (`orientation/`) | val corner error drops vs unrefined; orientation accuracy on val |
| U5 | YOLO-pose 4-corner keypoint model (ordered corners directly) | same as U3; compared to OBB+refine on val |
| U6 | ONNX export + parity + latency + browser note (`export/`) | onnxruntime matches PyTorch on 20 eval images; size and CPU latency recorded |
| U7 | Comparison table, overlays, failure gallery (`visualization/`, `REPORT.md`) | JPEGs < 3 MB, each opened and read |
| U8 | Permissively licensed CenterNet-style detector (MobileNetV3 + ordered corner offsets, plain PyTorch, `learned/centernet/`) | scored on val/eval like the others |

## Log

- 2026-09-25: contracts written; ultralytics 8.4.163, onnx, onnxruntime, onnxslim, shapely installed in the vega venv.
- 2026-09-25: U1 metrics landed (35 tests; GT-as-prediction scores perfect; a perfect quad scores only 0.90-0.99 IoU against deformed outlines, so corner error is the sharper metric).
- 2026-09-25: YOLO data copies at 1280 long side (obb/ and pose/ trees; Ultralytics resolves symlinks, so each tree has its own image copy). YOLO26n-OBB at 1024 is ~8.5 min/epoch on MPS; cut to 20 epochs (~3 h) to stay within budget.
- 2026-09-25: upside-down crop dataset built from GT (train 17,287 / val 3,716 crops, jittered corners); classifier written, training queued behind the OBB run (one GPU job at a time).
