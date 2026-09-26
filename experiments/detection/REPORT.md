# Check detection: results on the held-out eval split

Takeaway: the learned detector (YOLO26n-OBB, plus our corner refinement and upside-down classifier) finds every check in 98.5% of eval photos. The classical OpenCV pipeline manages 79%. Both put corners at a median of about 1.2 px once refined. Ship classical first as planned (milestone 2); its misses land in the count step's manual fix. Then move to the learned detector (milestone 2b) after settling its AGPL licence question. (Update 2026-09-26: the licence is settled, since the app is AGPL-3.0, and the close-up round below changes what ships; see Recommendation.)

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

## Close-up regime (v1.1 close-up eval set, the way the operator shoots)

Takeaway: with 4-6 checks filling the frame, YOLO still gets 98% of photos entirely right. With ONE check filling the frame under a steep tilt, it finds the check (98% recall) but its corners are poor (refined median 11 px, p90 153 px), and classical places corners better. This is the largest open weakness of the learned pipeline.

600 photos from the v1 eval pools (no overlap with training): 450 "close" (4-6 checks filling the frame, median check width 1,367 px) and 150 "single" (one check filling 70-100% of the frame, any rotation, sometimes a corner cut, steep tilt). Scored once, CPU.

| Regime | Pipeline | Precision | Recall | Photos all-correct | Corner median px | p90 | Orientation |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| close (2,120 checks) | Classical + refine + orient | 93.6 | 93.9 | 68.9 | 2.89 | 25.9 | 98.5 |
| close | YOLO26n-OBB + refine + orient | 99.3 | **100.0** | **98.0** | 4.14 | **21.4** | 99.0 |
| single (150 checks) | Classical + refine + orient | 78.5 | 85.3 | 82.0 | **2.64** | **14.5** | 97.6 |
| single | YOLO26n-OBB + refine + orient | **96.7** | **98.0** | **94.7** | 11.02 | 152.6 | 97.9 |
| single | YOLO26n-OBB raw box | 96.7 | 98.0 | 94.7 | 78.25 | 176.8 | – |

Why the single regime fails for YOLO (see `showcase/v1.1-closeup__single__worst4.jpg`):
- The check is a strong trapezoid, which a rotated rectangle cannot represent.
- v1 training never showed a check larger than about half the frame, so the box also comes out loose and mis-rotated.
- Refinement only searches a narrow band around each side, so it cannot bridge a 100-300 px gap.
- Classical, which fits edges directly, is tight on the same photos but misses or splits 15% of them (for example, a printed rule splits one check into panels).

All three fixes proposed at the time (close-up training, CenterNet, the hybrid) were built; results are in the next section.

## Close-up round: fine-tune, hybrid, and the permissive CenterNet (2026-09-26)

Takeaway: the **hybrid** stage closes the single-check corner gap, not the fine-tune. When a learned box covers at least 20% of the photo, the classical fitter runs inside it and its quad replaces the box if they agree (IoU >= 0.7). With either YOLO model, single-check corner p90 drops from about 153 px to about 27-31 px. The **close-up fine-tune** of YOLO is roughly a wash on counting, with a small corner gain in the close regime. The **permissive CenterNet** is the best counter of all (99.5-100% of photos entirely right, orientation 100%), but its corners are coarse even after refinement.

Everything was tuned or selected on v1 val plus close-up val (300 photos); each frozen pipeline was scored once on eval. Training added 1,500 close-up training photos from v1's training pools only.

**v1 eval** (750 photos):

| Pipeline | Precision | Recall | Recall@0.9 | Photos all-correct | Corner median px | p90 | < 5 px % | Orientation |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Classical + refine + orient | 95.7 | 96.1 | 91.6 | 79.1 | 1.22 | 9.9 | 84 | 99.3 |
| YOLO (v1) + refine + orient | 99.4 | 100.0 | 98.8 | 98.5 | 1.24 | 8.1 | 84 | 99.5 |
| YOLO (v1) + hybrid + refine + orient | 99.4 | 100.0 | 98.9 | 98.5 | 1.24 | 7.8 | 84 | 99.5 |
| YOLO fine-tune + hybrid + refine + orient | 99.2 | 100.0 | 98.9 | 97.9 | 1.24 | 7.4 | 85 | 99.6 |
| CenterNet + dedup + hybrid + refine | **99.9** | 99.9 | 91.4 | **99.5** | 3.23 | 21.5 | 58 | **100.0** |

**Close-up eval**, per framing regime:

| Regime | Pipeline | Precision | Recall | Photos all-correct | Corner median px | p90 | < 5 px % | Orientation |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| close (450 photos) | Classical + refine + orient | 93.6 | 93.9 | 68.9 | 2.89 | 25.9 | 62 | 98.5 |
| close | YOLO (v1) + refine + orient | 99.3 | 100.0 | 98.0 | 4.14 | 21.4 | 54 | 99.0 |
| close | YOLO (v1) + hybrid + refine + orient | 99.3 | 100.0 | 98.0 | 4.03 | 21.0 | 54 | 98.9 |
| close | YOLO fine-tune + hybrid + refine + orient | 99.1 | 100.0 | 97.3 | 3.56 | 18.5 | 57 | 99.2 |
| close | CenterNet + dedup + hybrid + refine | **100.0** | 100.0 | **99.8** | 12.12 | 51.7 | 33 | **100.0** |
| single (150 photos) | Classical + refine + orient | 78.5 | 85.3 | 82.0 | 2.64 | 14.5 | 68 | 97.6 |
| single | YOLO (v1) + refine + orient | 96.7 | 98.0 | 94.7 | 11.02 | 152.6 | 42 | 97.9 |
| single | YOLO (v1) + hybrid + refine + orient | 96.7 | 98.0 | 94.7 | **2.81** | **26.6** | **64** | 97.9 |
| single | YOLO fine-tune + hybrid + refine + orient | 93.8 | 100.0 | 96.0 | 2.89 | 31.2 | 64 | 95.3 |
| single | CenterNet + dedup + hybrid + refine | **100.0** | **100.0** | **100.0** | 2.90 | 57.1 | 62 | **98.0** |

Notes:
- **Fine-tune vs hybrid.** On single checks, the fine-tune alone takes recall to 100%, but its corners stay about 150 px off at p90: a rotated rectangle cannot be a trapezoid. The hybrid fixes the corners for any learned detector. The fine-tune was chosen on val, where it looked slightly better; on eval it is within noise of the v1 model (better close-regime corners, a few more single-check false positives).
- **Plain classical's own limits.** Relaxing its area and angle limits changed nothing on close-up val. Its single-check trouble is printed inner boxes read as checks, and misses. So its defaults stay, and it gains from close-up framing only through the hybrid.
- **CenterNet** (MobileNetV3-Large, ordered corner offsets, 768 px, 12 epochs on v1 plus close-up training):
  - It counts almost perfectly once duplicate peaks on large checks are removed (quad-IoU suppression at 0.5, chosen on val).
  - Its corners come from offsets regressed off a stride-4 center (one output cell is about 16 px of a 3 MP photo), so refinement has more to fix.
  - Next steps, if the permissive model should also own the corners: train longer at 1024 px, or add a per-corner heatmap head.
- **Browser artifacts** (native onnxruntime on 1 thread; measured while other jobs shared the CPU, so treat as upper bounds):

| Artifact | Size | 1 thread | 4 threads | Parity vs PyTorch |
| :--- | ---: | ---: | ---: | :--- |
| YOLO fine-tune ONNX at 1024 | 10.2 MB | 90 ms | 67 ms | 0.03 px |
| CenterNet ONNX at 768 | 13.1 MB | 81 ms | 50 ms | 0.0002 px; the same detections on 20 scenes |

- **Hybrid cost.** It adds about 1 s of classical fitting per frame-filling check. It only fires on checks covering at least 20% of the photo, so single-check photos pay about 1 s and wide shots pay nothing.

## Recommendation

Updated 2026-09-26, after the close-up round.

1. **Milestone 2 (now): classical + refine + orient.**
   - Unchanged: the count step's add/remove flow covers its misses.
   - Expect it to be weakest exactly where the operator shoots: one frame-filling check (82% of such photos entirely right), and 4-6 close checks (69%).
2. **Milestone 2b (the real detector): YOLO + hybrid + refine + orient.**
   - Ship the fine-tuned checkpoint (selected on val): 97-98% of photos entirely right in every regime, and single-check corners at a median of about 2.9 px.
   - The repo is AGPL-3.0, so shipping it is allowed.
   - The hybrid needs the classical detector in the browser anyway, and milestone 2 ships it.
3. **Permissive fallback: CenterNet + dedup + hybrid + refine.**
   - Built and trained. It is the best counter and fully ours, but its corners need more training before it can replace YOLO.
   - A cheap, strong combination to test next: CenterNet decides how many checks there are and roughly where, and the hybrid/refine stages place the corners.
4. **Before trusting any of this: the printed mock-check photo set (spec section 8).** All numbers here are synthetic.

## AGPL note

Ultralytics (the training code and the `yolo26n-obb.pt` pretrained weights) is AGPL-3.0, so the trained model and its ONNX export carry AGPL obligations. Training and evaluating here is fine. Shipping it in the public app needs one of three things:
- (a) the app repo licensed AGPL-compatible. It is already a public static site serving its own source, so the practical cost is the licence choice itself.
- (b) an Ultralytics Enterprise licence.
- (c) the permissively licensed CenterNet detector we built (`learned/centernet/`: MobileNetV3 with BSD-3 code, ordered corners, no NMS, 13.1 MB ONNX with verified parity).
  - Trained 2026-09-26; results in the close-up round section.
  - Its ordered-corner regression can also predict a corner hidden under another check, which a box cannot.

Decision (2026-09-26): the app repo is now AGPL-3.0, so option (a) applies and the YOLO model may ship. The permissive CenterNet will also be trained, so the project owns a licence-free detector; its eval row will be added here.

## Reproduce

Paths are under `/Volumes/vega/datasets/check-transcriber/experiments/detection/`: predictions in `predictions/eval__*.json`, scores in `scores/eval__*/metrics.{md,json}` (with every breakdown), classical outputs in `classical/final/`, OBB weights in `runs/obb/y26n_1024_e8/weights/best.pt`, ONNX in `export/`, images in `showcase/`. Commands are in `README.md`.
