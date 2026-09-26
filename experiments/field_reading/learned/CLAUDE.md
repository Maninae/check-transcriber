# learned/: learned text recognizers (U4)

CRNN-CTC recognizers trained from scratch, and TrOCR-small (zero-shot + fine-tuned), all reading one field crop into `(text, confidence)`. Output files follow the PLAN.md prediction contract.

## Modules
| Module | Responsibility |
|---|---|
| `text_charset.py` | charsets (general, amount), text <-> CTC labels (blank = 0) |
| `ctc_decoding.py` | greedy CTC decode, confidences (mean_char / min_char / path), constrained decoding |
| `line_image_preprocessing.py` | the CRNN input recipe (grey, height 32, width clamp, [-1,1]); the browser must mirror it |
| `crop_augmentation.py` | box jitter (+-10% of height, imperfect localization) + photometric/geometric augmentation |
| `context_crop_export.py` | one-time CLI: wide-margin train crops so jitter can pull in real neighbouring clutter |
| `line_crop_dataset.py` | CRNN datasets, collate, width-bucketed sampler |
| `crnn_model.py` | CRNN architecture + variant registry |
| `train_crnn.py` | CRNN training CLI (MPS lock, per-epoch val subset selection) |
| `crnn_reader.py` | CRNN checkpoint -> batch reader |
| `trocr_tokenizer.py` | TrOCR-small text codec on sentencepiece (transformers 5.x cannot load it) |
| `trocr_reader.py` | TrOCR reader (preprocessing, greedy decode, sequence confidence), model registry |
| `trocr_finetune_data.py`, `finetune_trocr.py` | TrOCR fine-tune data + CLI (time-budgeted; logs SSBI per checkpoint). Not used for shipped results (synth handwriting = font recall) |
| `trocr_onnx_decoding.py` | onnxruntime greedy decode: our no-cache graphs and the transformers.js merged past-KV graphs |
| `trocr_onnx_benchmark.py` | TrOCR ONNX equivalence/size/latency (ours vs Xenova/transformers.js) |
| `prediction_combination.py` | routers (oracle-hw, field, max-confidence, cascade) composed offline from base prediction files |
| `confidence_kind_selection.py` | which CTC confidence statistic gates best per field |
| `real_car_scoring.py` | ORAND-CAR-2014 real courtesy amounts (CC BY-NC-ND; eval only) |
| `run_prediction_queue.sh`, `run_training_queue.sh` | the MPS job sequences actually run |

## Shipped methods (prediction ids)
- `crnn_general`, `crnn_amount_route` (amount CRNN on amount_numeric): printed reader; synth handwriting = font recall.
- `trocr_small_handwritten_zs_grey`: zero-shot TrOCR-small-handwritten with grey+autocontrast input (RGB input fails on real crops).
- `u4_router` (non-oracle): amount -> max-confidence(TrOCR-hw-grey, amount CRNN); other fields -> style route.
- `trocr_hwgrey_crnn_route_oraclehw`: ceiling using the GT handwritten flag (not shippable).
| `reading_methods.py` | method registry: which reader reads which row (`_oraclehw` = uses GT handwritten flag) |
| `field_crop_sources.py` | oracle crops vs localizer boxes (`crop_field_from_check`) |
| `prediction_rows.py` | contract row dataclass + JSONL writer |
| `predict.py` | CLI writing `predictions/<split>/<method>__loc=<loc>.jsonl` |
| `selection_scoring.py` | method x field x slice tables via the metrics harness (val selection) |
| `real_ssbi_scoring.py` | sanity scores on 78 real handwritten SSBI crops (CC BY-NC; never copy off vega) |
| `onnx_export.py`, `onnx_benchmark.py` | ONNX fp32/int8 export; equivalence, size, CPU latency |
| `mps_lock.py` | the area MPS lock as a context manager (writes `owner.txt`) |
| `recognizer_paths.py` | leaf module: checkpoint / ONNX locations (keeps imports acyclic) |
| `onnx_trocr_reader.py` | onnxruntime TrOCR reader (the browser path) for ONNX/int8 accuracy checks |
| `handwriting_style_classifier.py` | printed-vs-handwritten CNN (185k params): the non-oracle routing signal |
| `real_set_router_analysis.py` | routers scored on real crops (SSBI, ORAND-CAR): TrOCR share, top-50%-confidence accuracy |

## Gotchas
- Synthetic handwriting (24 fonts) measures font recall, not handwriting: the CRNN wins synth val but collapses on real SSBI crops, where zero-shot TrOCR-hw (grey input) reads well. Judge handwriting on real crops.
- MPS recompiles kernels for every new input shape: batches are padded to widths that are multiples of 128 (`MPS_WIDTH_MULTIPLE`). Without it training stalls in shader compilation.
- Xenova's merged past-KV decoder returns (0, 8, 1, 32) placeholder encoder KV on cache steps: keep the step-0 cross-attention KV (`trocr_onnx_decoding`).
- TrOCR int8 damage comes from the encoder: ship fp32 encoder + int8 decoder (128 MB), not full int8.
- MPS has no CTC kernel: the loss runs on CPU; the network on MPS.
- TrOCR decoding starts at `</s>` (id 2) and emits pieces + `</s>` (no `<s>`); fine-tune targets follow that layout.
- The machine is shared: <= 2 workers, check `vm_stat`, every MPS job goes through `hold_mps_lock`.
- Weights and caches live on vega (`/Volumes/vega/ai-models/field-reading/`), never the internal disk.
