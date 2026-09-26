# Running field reading in the browser

Takeaway: localization + printed reading fit the app's budget easily (~30 MB, milliseconds per field). Handwriting via TrOCR-small is feasible but heavy (128 MB one-time download, ~0.1 s per crop natively, a few tenths of a second in WASM), so ship it as an opt-in "read handwriting" model, not in the default load.

## Models (all ONNX, verified against PyTorch on 20 crops)

| model | job | file size | CPU latency (onnxruntime, this Mac) | equivalence | licence |
|---|---|--:|--:|---|---|
| segnet_mobilenetv3l_768 | 7 field boxes per check | 12.8 MB fp32 | 41 ms per check (66 ms with post-process) | 20/20 identical boxes, max logit diff 1e-4 | ours; encoder init timm mobilenetv3_large_100 (Apache-2.0) |
| crnn_general | printed + synthetic-style text, all fields | 8.3 MB fp32 | 1.5 ms per field | 20/20 identical text | ours |
| crnn_amount | courtesy amount digits | 8.2 MB fp32 | 1.3 ms per field | 20/20 | ours |
| style classifier | handwritten vs printed crop | 0.75 MB | <1 ms | max diff 1e-6 | ours |
| TrOCR-small-handwritten (Xenova export) | real handwriting | 128 MB (fp32 encoder 87.5 + int8 merged decoder 40.5) | ~75-100 ms per crop, greedy decode | same reads as fp32 on SSBI and ORAND-CAR | code MIT (microsoft/unilm); weights fine-tuned on IAM (non-commercial research terms): needs a licence decision |

- Keep models fp32 except the TrOCR decoder. int8 CRNN moves confidences by up to 1.6 nats (breaks the gate); full-int8 TrOCR (64 MB) drops real legal-line accuracy 0.76 -> 0.40 because the encoder does not survive quantization.
- Pre-/post-processing is plain array math plus OpenCV.js calls already in the app (resize, grayscale, autocontrast, crop + 15% margin). TrOCR needs grayscale + autocontrast input: RGB input collapses it on real crops (0.82 -> 0.09 on amounts).

## Runtime
- onnxruntime-web (WASM, SIMD + threads) runs all of these; expect roughly 2-4x native latency on a mid-range laptop. Budget per batch of 6 checks: segnet 6 x ~0.15 s, CRNN 42 fields x ~5 ms, TrOCR only on crops the style classifier calls handwritten (~15 x 0.3 s) -> ~5-6 s, inside the spec's 10 s. Run in the existing Web Worker.
- transformers.js can load `Xenova/trocr-small-handwritten` directly (dtype per file: encoder fp32, decoder q8); it wraps onnxruntime-web, so there is no second runtime. Its fp16 files are for WebGPU only.
- Hosting: every file is under GitHub Pages' 100 MB limit, so models can live in the Pages repo (keeps the CSP to the app's own origin) or on the CDN already allowed for OpenCV/Tesseract; either way the service worker caches them after first use.

## Is Tesseract.js still the right v1 for printed fields?
- Engine parity: tesserocr 5.5 with the app's own eng 4.0.0_best_int data reproduces Tesseract.js 7 (15/20 identical reads on a mixed sample; all 5 differences were handwritten misreads at <55 confidence). So the Tesseract numbers here are the app's numbers.
- On synthetic eval (held-out templates and 9 held-out typefaces), end to end with segnet boxes: check number Tesseract 88.4% vs CRNN 98.3%, printed payer about equal (~86%), printed amount/date far better with the CRNN. Tesseract needs the field cleanup rules (edge artifacts, protector asterisks) to be usable at all.
- Recommendation: make the CRNN (8 MB) the printed reader in milestone 4 and keep Tesseract.js only for the orientation pass (stage 4), where it is already loaded. The CRNN has not been validated on real printed checks yet; the gold photo set is the gate.
