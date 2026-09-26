# learned/centernet/: permissively licensed CenterNet-style check detector (U8)

Plain PyTorch, no Ultralytics (AGPL). torchvision MobileNetV3 (BSD-3) backbone with ImageNet weights, FPN-style neck to stride 4, a 1-channel check-center heatmap and an 8-channel ordered corner-offset head (TL,TR,BR,BL x/y from the peak cell). NMS-free: a check is a 3x3 heatmap local maximum. Corners come out in the check's own order, so `orientation_known=True`. Exports to ONNX opset 17 (static 1x3xSxS) for onnxruntime-web WASM.

## Entry points

- Train: `python -m experiments.detection.learned.centernet.train_centernet --run-name <name> [--epochs N --batch-size B --input-size-pixels S --device mps]` (every `CenterNetTrainingConfig` field is a flag; outputs in `<vega>/experiments/detection/runs/centernet/<name>/`).
- Predict a split: `python -m experiments.detection.learned.run_centernet_detector --weights <best.pt> --split val --output <predictions.json> [--limit N]`, then score with `metrics.score_predictions`.
- ONNX: `python -m experiments.detection.learned.centernet.onnx_export --weights <best.pt> [--parity-scenes 5]`.

## Modules (dependencies flow downward)

| Module | Responsibility |
|---|---|
| `train_centernet.py` | training loop, AdamW + warmup/cosine, checkpoints, JSONL log |
| `quick_validation.py` | val-subset scoring for checkpoint selection (F1 at IoU 0.90) |
| `onnx_export.py` | ONNX export + onnxruntime/PyTorch parity check |
| `centernet_inference.py` | checkpoint loading, full-res image -> `DetectedCheck`s |
| `check_scene_dataset.py` | torch Dataset over the 1280 copies; returns tensors + target maps |
| `scene_augmentation.py` | one-affine geometric warp (any-angle rotation, scale, shift; no flips) + photometric jitter |
| `check_center_targets.py` | GT corners -> heatmap, offsets, weights, normalizers |
| `centernet_decoding.py` | numpy peak pick + corner math; the reference for the JS port |
| `centernet_model.py` | backbone, neck, heads, ONNX wrapper (sigmoid baked in) |
| `centernet_losses.py` | penalty-reduced focal loss + diagonal-normalized L1 |
| `render_detection_overlay.py` | draw ordered quads (TL big dot, TR small dot) and heatmaps |
| `letterbox_geometry.py`, `centernet_config.py` | leaf: affine helpers; constants and the training config dataclass |

## Invariants

- Offsets are in output cells relative to the cell CENTER: `corner = (cell + 0.5 + offset) * OUTPUT_STRIDE`. Encoder and decoder must change together; `tests/test_centernet_targets_and_decoding.py` checks the exact round trip.
- Anchor = quad center if on canvas, else centroid of the on-canvas part; checks with < 15% of their area on canvas are not targets. Off-canvas corners are still regressed.
- Never add horizontal/vertical flips: they reverse the corner winding. Rotation keeps the check's own TL first.
- Training reads the 1280-long-side copies (`yolo_data/long1280/obb/images/<split>/`); GT is scaled by `copy_width / image_width`. Inference downscales full-res images to 1280 with INTER_AREA first, to match.
- Everything heavy (runs, pretrained weights under `pretrained/torchvision/`) lives on vega.
