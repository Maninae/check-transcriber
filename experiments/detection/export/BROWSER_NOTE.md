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

## Licensing, before shipping

The ONNX file is derived from Ultralytics' AGPL-3.0 code and pretrained weights. Shipping it inside the public app means either (a) licensing the app repo AGPL-3.0-compatible: the repo is public and serves its own source, so the source-availability duty is already met in practice, (b) buying an Ultralytics Enterprise licence, or (c) swapping in the permissively licensed CenterNet detector in `learned/centernet/`: built and tested, not yet trained, about 2.5 GPU hours, 13.1 MB ONNX, with the same JS post-processing minus NMS.
