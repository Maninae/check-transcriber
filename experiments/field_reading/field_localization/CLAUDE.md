# field_localization (unit U2)

Given a rectified 1600 px check crop, output one box (or none) per target field. Two method families, one shared scoring path. Contract rows and paths are defined in `../PLAN.md`; everything writes to vega (`localization_config.py`).

## Modules
| Module | Responsibility |
|---|---|
| `localization_config.py` | paths, method ids, scored / box-target statuses |
| `box_geometry.py` | IoU, coverage, normalize / clip (pure) |
| `localization_targets.py` | field rows -> one GT record per check (`fields` dict; missing key = absent field) |
| `prediction_io.py` | contract JSONL rows |
| `localization_metrics.py` | per-row scoring (IoU, hit, coverage, area ratio, false box) + grouped summaries |
| `localization_report.py` | CLI: metrics JSON + markdown tables for a split |
| `layout_prior.py` | per (group, field) normalized priors from train (median box, quantile search region, height, presence) |
| `ink_refinement.py` | classical: adaptive threshold, ruled-line removal, size-filtered components -> words -> text line nearest the prior |
| `layout_prior_localizer.py` | CLI: runs `layout_prior_family_oracle` and `layout_prior_size_kind` on val / eval (CPU, 3 procs) |
| `segnet_config.py` | canvas 768x352, stride 2, encoder name, method id |
| `segnet_model.py` | MobileNetV3-L (timm, Apache-2.0) + FPN-lite decoder -> 7 field logit maps; ONNX-safe ops only |
| `segnet_augmentation.py` | corner-jitter homography + photometric noise |
| `segnet_dataset.py` | crop -> canvas geometry (single source of truth), soft fractional box targets |
| `segnet_postprocess.py` | logits -> sigmoid -> upsample -> largest-mass component -> box + confidence |
| `segnet_inference.py` | batched predict (post-processes per batch; never hold a split of logits) + confidence gate |
| `segnet_train.py` / `segnet_predict.py` | CLIs: train (val-IoU checkpoint selection), predict val+eval and pick the gate on val |
| `onnx_export.py` | ONNX export, 20-crop equivalence, size, onnxruntime CPU latency |
| `overlay_visual.py` | 6 eval crops with GT + every method's boxes |
| `mps_lock.py` | the area's one-MPS-job lock (`logs/MPS_LOCK`) |

## Run order
1. `python -m experiments.field_reading.field_localization.layout_prior_localizer`
2. `python -m experiments.field_reading.field_localization.segnet_train` (takes the MPS lock), then `segnet_predict`, then `onnx_export`
3. `localization_report --split eval` and `overlay_visual`

## Invariants
- Boxes are pixel-edge [x0, y0, x1, y1] in the 1600 px crop; canvas scale is `768 / crop_width` in both axes.
- Browser port: divide RGB by 255 (normalization is inside the model), paste the resized crop top-left on a zero 768x352 canvas, run, then `segnet_postprocess` logic and the gate from `<method>__gate.json`.
- Select on val only; eval is scored once per final method.
