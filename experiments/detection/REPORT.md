# Check detection: results on the held-out eval split

Takeaway: the learned detector (YOLO26n-OBB, plus our corner refinement and upside-down classifier) finds every check in 98.5% of eval photos. The classical OpenCV pipeline manages 79%. Both put corners at a median of about 1.2 px once refined. Ship classical first as planned (milestone 2); its misses land in the count step's manual fix. Then move to the learned detector (milestone 2b) after settling its AGPL licence question.

All numbers: v1 synthetic eval split (750 photos, 3,753 checks, templates and backgrounds unseen in training). Every tuning and model choice was made on val, and eval was scored once per final pipeline. Corner error is the mean of the four corners' pixel distances on the full-resolution photo (3-6 MP), after allowing any starting corner. Orientation is scored separately.

## Headline

| Pipeline | Precision | Recall | Recall@IoU 0.9 | Photos all-correct | Corner median px | p90 | p95 | < 2 px | < 5 px | Orientation | s / photo* |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Classical (OpenCV) | 95.7 | 96.1 | 91.3 | 79.1 | 2.18 | 13.6 | 25.7 | 47 | 75 | – | 1.26 |
| Classical + refine + orient | 95.7 | 96.1 | 91.6 | 79.1 | **1.22** | 9.9 | 23.5 | 67 | 84 | 99.3 | 1.33 |
| YOLO26n-OBB | 99.4 | **100.0** | 97.9 | **98.5** | 7.83 | 23.8 | 35.5 | 1 | 27 | – | 0.17 |
| YOLO26n-OBB + refine + orient | 99.4 | **100.0** | **98.8** | **98.5** | 1.24 | **8.1** | **14.1** | 65 | 84 | **99.5** | 0.27 |

Precision and recall are at IoU 0.5 against the true (deformed) check outline. "Photos all-correct" means every check was found and nothing extra. *Python on a shared, memory-starved M4; the browser estimate is in `export/BROWSER_NOTE.md`.

- The box model alone is the best finder and the worst corner-placer: a rotated rectangle cannot follow perspective keystone (median 7.8 px). Refinement fixes that: median 7.8 → 1.24 px, and 1% → 65% of checks under 2 px.
- The same refinement helps classical too (2.18 → 1.22 px), and the classifier gives both pipelines orientation at 99.3-99.5%.

## Where each breaks (eval, refined pipelines)

Recall at IoU 0.5 / median corner px; precision only where false positives belong to the group.

| Slice | Checks | Classical recall | Classical corner px | YOLO recall | YOLO corner px | Precision classical / YOLO |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Overlapped (another check on top) | 364 | 81.3 | 18.5 | 100.0 | 7.0 | – |
| Loose fan layout | 88 | 86.4 | 42.0 | 98.9 | 14.2 | 92.7 / 100 |
| Loose overlap layout | 899 | 90.8 | 2.3 | 100.0 | 2.4 | 93.0 / 99.2 |
| Partly out of frame | 103 | 88.3 | 1.3 | 100.0 | 1.2 | – |
| Patterned fabric (tartan, gingham) | 401 | 93.8 | 1.1 | 100.0 | 1.2 | **82.3 / 95.0** |
| Bedding (white on white) | 572 | 95.1 | 1.4 | 100.0 | 1.5 | 97.0 / 100 |
| Hand + phone cast shadow | 363 | 94.5 | 1.1 | 100.0 | 1.3 | 97.7 / 99.2 |
| Curled checks | 1,345 | 95.9 | 2.0 | 99.9 | 2.0 | – |
| Flat checks | 904 | 96.3 | 0.8 | 100.0 | 0.8 | – |

Failure modes, from the galleries (`showcase/failures__*.jpg`):
- **Classical**
  - It misses the check underneath a stack, or returns only its visible part.
  - It reads a plain block of tartan between stripes as paper (81 false positives on eval).
  - A hard shadow edge splits one check into a fragment.
  - It sometimes snaps to a printed box inside a check (a quarter-size quad).
- **YOLO**
  - Almost every false positive is on one held-out red tartan blanket (pale stripe blocks, scores 0.28-0.68).
  - The one miss is two stacked checks merged into one box.
  - The corner-error tail is hidden corners: a corner physically under another check cannot be refined, only guessed.
- **Refinement**
  - It improves the median and p90, but on white bedding and curled ends it can move a nearly right corner by a few px. Flat checks end at 0.8 px median.

## Recommendation

1. **Milestone 2 (ship now): classical + refinement + orientation classifier.**
   - It is 96% recall with sub-1.5 px corners on typical scenes.
   - The spec's count step already exists to catch the 21% of photos with a miss or extra.
   - The worst cases are stacked or fanned checks and patterned blankets, which the "Add a check" / remove flow handles.
   - The classifier is our own 1.9 MB model (no licence issue). It replaces the OCR-based up/down vote in spec 5.4, or backs it up.
2. **Milestone 2b: YOLO26n-OBB + refinement + classifier.**
   - It gets 98.5% of photos entirely right, including overlaps and out-of-frame checks.
   - It is a 10.2 MB ONNX that needs onnxruntime-web (WASM, no WebGPU).
   - Blocked only on licensing (next section).
3. **Before trusting either on Angie's photos, shoot the spec's printed mock-check set (spec section 8).**
   - Everything here is synthetic.
   - A look at a handful of public real phone photos (close crops of single Algerian cheques, out of our setting) showed both detectors struggling when the check fills the frame with no surface around it. That case is not in v1.

## AGPL note

Ultralytics (the training code and the `yolo26n-obb.pt` pretrained weights) is AGPL-3.0, so the trained model and its ONNX export carry AGPL obligations. Training and evaluating here is fine. Shipping it in the public app needs one of three things:
- (a) the app repo licensed AGPL-compatible. It is already a public static site serving its own source, so the practical cost is the licence choice itself.
- (b) an Ultralytics Enterprise licence.
- (c) the permissively licensed CenterNet detector we built (`learned/centernet/`: MobileNetV3 with BSD-3 code, ordered corners, no NMS, 13.1 MB ONNX with verified parity).
  - It is tested and overfit-checked, but not yet trained: about 2.5 GPU hours.
  - Train it only if Owen rules out (a) and (b), or if hidden-corner accuracy in overlaps turns out to matter; its ordered-corner regression can predict a hidden corner, and a box cannot.

## Reproduce

Paths are under `/Volumes/vega/datasets/check-transcriber/experiments/detection/`: predictions in `predictions/eval__*.json`, scores in `scores/eval__*/metrics.{md,json}` (with every breakdown), classical outputs in `classical/final/`, OBB weights in `runs/obb/y26n_1024_e8/weights/best.pt`, ONNX in `export/`, images in `showcase/`. Commands are in `README.md`.
