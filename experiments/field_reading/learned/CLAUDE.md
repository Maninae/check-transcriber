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
| `trocr_finetune_data.py`, `finetune_trocr.py` | TrOCR fine-tune data + CLI (time-budgeted) |
| `reading_methods.py` | method registry: which reader reads which row (`_oraclehw` = uses GT handwritten flag) |
| `field_crop_sources.py` | oracle crops vs localizer boxes (`crop_field_from_check`) |
| `prediction_rows.py` | contract row dataclass + JSONL writer |
| `predict.py` | CLI writing `predictions/<split>/<method>__loc=<loc>.jsonl` |
| `selection_scoring.py` | method x field x slice tables via the metrics harness (val selection) |
| `real_ssbi_scoring.py` | sanity scores on 78 real handwritten SSBI crops (CC BY-NC; never copy off vega) |
| `onnx_export.py`, `onnx_benchmark.py` | ONNX fp32/int8 export; equivalence, size, CPU latency |
| `mps_lock.py` | the area MPS lock as a context manager |

## Gotchas
- MPS recompiles kernels for every new input shape: batches are padded to widths that are multiples of 128 (`MPS_WIDTH_MULTIPLE`). Without it training stalls in shader compilation.
- MPS has no CTC kernel: the loss runs on CPU; the network on MPS.
- TrOCR decoding starts at `</s>` (id 2) and emits pieces + `</s>` (no `<s>`); fine-tune targets follow that layout.
- The machine is shared: <= 2 workers, check `vm_stat`, every MPS job goes through `hold_mps_lock`.
- Weights and caches live on vega (`/Volumes/vega/ai-models/field-reading/`), never the internal disk.
