# experiments/detection: finding checks in a phone photo

Finds every check in a photo of checks lying on a household surface. Each check comes back as a quadrilateral whose corners are in the check's own top-left, top-right, bottom-right, bottom-left order, ready to be perspective-warped. Results and the recommendation are in [REPORT.md](REPORT.md); the browser plan is in [export/BROWSER_NOTE.md](export/BROWSER_NOTE.md).

## Pipelines

| Pipeline | Stages | Licence of what ships |
|---|---|---|
| Classical | `classical/` (OpenCV, no model) → `refinement/` → `orientation/` | ours |
| Learned | `learned/` YOLO26n-OBB → `refinement/` → `orientation/` | AGPL-3.0 (Ultralytics) |
| Learned, permissive (not yet trained) | `learned/centernet/` MobileNetV3 + ordered corner offsets | ours + BSD-3 |

## Commands (run from the repo root with the vega venv)

Detect, post-process, then score:

```
python -m experiments.detection.classical.run_classical_detector \
    --split val \
    --output <preds.json>
python -m experiments.detection.learned.run_yolo_detector \
    --weights <best.pt|.onnx> \
    --split val \
    --output <preds.json>
python -m experiments.detection.pipeline.postprocess_predictions \
    --predictions <preds.json> \
    --output <out.json> \
    --refine \
    --orient
python -m experiments.detection.metrics.score_predictions \
    --predictions <out.json> \
    --split val \
    --output-dir <dir>
```

Train and export:

```
python -m experiments.detection.learned.prepare_yolo_datasets
python -m experiments.detection.learned.train_yolo \
    --variant obb \
    --model yolo26n-obb.pt \
    --run-name <name> \
    --batch-size 4
python -m experiments.detection.orientation.build_orientation_crops
python -m experiments.detection.orientation.train_upside_down_classifier
python -m experiments.detection.export.export_yolo_onnx \
    --weights <best.pt>
```

Pictures for people:

```
python -m experiments.detection.visualization.render_detection_showcase \
    --split eval \
    --detector name=<preds.json>:<metrics.json> \
    --output-dir <dir>
```

Tests: `python -m pytest experiments/detection/tests`.

Data in, runs out: everything lives on `/Volumes/vega/datasets/check-transcriber/` (see `config/paths.py`).
