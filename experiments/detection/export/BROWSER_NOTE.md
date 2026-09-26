# Running the learned check detector in the browser (milestone 2b)

Takeaway: YOLO26n-OBB runs in onnxruntime-web on WASM with no WebGPU. It is a 10.2 MB one-time download, and inference should take roughly 0.1-0.25 s per photo on a mid-range Windows laptop, well inside the spec's 3 s budget from paste to count step. Nothing about the UI changes: it produces the same quadrilaterals the classical detector does.

## What ships

| File | Size | Licence | Role |
|---|---|---|---|
| `yolo26n-obb__imgsz=1024.onnx` (fp32, opset 17, static 1x3x1024x1024) | 10.2 MB | AGPL-3.0 (Ultralytics-trained) | finds checks as rotated boxes |
| upside-down classifier (plain PyTorch, 469k params) | ~1.9 MB fp32 | ours | 0 vs 180 degrees on a rectified crop |
| `ort.wasm` + `ort.min.js` (onnxruntime-web) | ~10 MB | MIT | runtime; cached by the service worker like OpenCV.js |

Artifacts: `/Volumes/vega/datasets/check-transcriber/experiments/detection/export/`. A dynamic-range int8 quantization (`onnxruntime.quantization.quantize_dynamic`) should roughly halve the size, but it is not yet measured for accuracy. Re-score on val before shipping it.

## Measured (this Mac, M4, native onnxruntime 1.30 CPU)

- Forward pass at 1024: 58 ms on 1 thread, 38 ms on 4 threads (bare session, no Ultralytics).
- Parity: on 20 eval scenes, the same letterboxed tensor through PyTorch and onnxruntime differs by at most 0.025 px (box) and 2e-5 (score). Scored end to end on val, the ONNX model equals the PyTorch model: 100% of scenes perfect, median corner error 8.3 vs 8.25 px before refinement.
- Expected in onnxruntime-web WASM: about 2-3x native single-thread on a laptop CPU, so 120-250 ms. With `ort.env.wasm.numThreads > 1` it is faster, but threads need cross-origin isolation (COOP/COEP headers), which GitHub Pages cannot set (a service-worker header shim exists but adds moving parts). Plan on single-thread SIMD. WebGPU is an optional accelerator only.

## Pre- and post-processing that moves to JS

1. Downscale the decoded photo to a long side of 1280 px, matching the training copies (`resize_to_training_scale`).
2. Letterbox to a 1024x1024 square: scale the long side to 1024, pad the rest with value 114, RGB, float32 in [0, 1], NCHW. It must be a square: the static graph expects 1024x1024, and val was scored that way.
3. Run the model. Output `output0` has shape (1, 6, 21504): per anchor `cx, cy, w, h, score, angle` in letterboxed pixels, angle in radians.
4. Keep anchors with score >= 0.25. Convert each to 4 corners (`cos/sin` of the angle around the center), then run rotated-box NMS at IoU 0.5 (polygon intersection of convex quads, a few dozen boxes, about 60 lines of JS). Do NOT export the NMS-free one-to-one head (`nms=False`): on val it left duplicates or misses in 6-8% of scenes at any threshold, versus 0% for this head plus NMS.
5. Undo the letterbox and the 1280 downscale to get full-resolution corners.
6. Corner refinement (`refinement/`), ported to OpenCV.js: `cv.remap` sampling plus small-array math, about 15-25 ms per check in Python.
7. Orientation: roll the corners so the first side is a long side, `cv.warpPerspective` to a 224x96 grayscale crop, run the classifier, and roll by 2 if the logit is > 0.

Numbering in reading order, the count header, and the confidence cues in spec 4.2 sit on top of this unchanged.

## Licensing (settled 2026-09-26: the app repo is AGPL-3.0, so option (a) applies)

The ONNX file is derived from Ultralytics' AGPL-3.0 code and pretrained weights. Shipping it inside the public app means either (a) licensing the app repo AGPL-3.0-compatible: the repo is public and serves its own source, so the source-availability duty is already met in practice, (b) buying an Ultralytics Enterprise licence, or (c) swapping in the permissively licensed CenterNet detector in `learned/centernet/`, which is now the recommended 2b detector (next section).

## The recommended 2b detector: CenterNet (permissive)

- **File:** `centernet-mnv3l__imgsz=768.onnx`, 13.8 MB fp32, opset 17, static 1x3x768x768, sigmoid inside the graph. Native 1-thread forward is 82 ms; expect 150-300 ms single-thread in WASM. Parity with PyTorch: 0.0002 px on 20 scenes.
- **Pre-processing:** downscale to a long side of 1280 px, letterbox to 768 px square, RGB, ImageNet mean/std normalization. See `learned/centernet/scene_augmentation.py` (`letterbox_scene_for_evaluation`) and `check_scene_dataset.normalize_image_to_tensor`.
- **Outputs:** `center_heatmap` (1, 1, 192, 192) and `corner_offsets` (1, 8, 192, 192).
- **Decoding (port `centernet_decoding.py`):**
  1. Take 3x3 local maxima on the heatmap above the score threshold, top-K.
  2. For each peak, compute `corner = (cell + 0.5 + offset) * 4` for TL, TR, BR, BL.
  3. Undo the letterbox and the 1280 downscale.
- **Then, in order:**
  1. Duplicate suppression (quad IoU 0.5).
  2. The count step, which can render now.
  3. The classical fit inside each detection (`CENTERNET_HYBRID_CONFIG`).
  4. Corner refinement.
  5. The orientation classifier, only for replaced quads. CenterNet's own corners are already ordered.
