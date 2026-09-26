# Model licences (U4 learned recognizers)

| Model | Source | Licence | Notes |
|---|---|---|---|
| `microsoft/trocr-small-handwritten` | https://huggingface.co/microsoft/trocr-small-handwritten | MIT (source repo microsoft/unilm, `LICENSE`); the HF model card declares no licence field | Fine-tuned by Microsoft on IAM. IAM's database terms are non-commercial research only, so shipping these weights (or our fine-tune of them) in a commercial product needs a legal look. |
| `microsoft/trocr-small-printed` | https://huggingface.co/microsoft/trocr-small-printed | MIT (microsoft/unilm); no licence field on the HF card | Fine-tuned on SROIE receipts (outputs upper case). |
| `Xenova/trocr-small-handwritten` | https://huggingface.co/Xenova/trocr-small-handwritten | none declared (ONNX conversion of the Microsoft MIT weights; inherits them and the IAM caveat) | transformers.js-ready ONNX (encoder + merged past-KV decoder, fp32/fp16/int8); benchmarked, not modified. |
| Our TrOCR ONNX exports | `recognizers/onnx/trocr_small_handwritten.{encoder,decoder}[.int8].onnx` | inherits trocr-small-handwritten | no-KV-cache decoder. |
| `crnn_general_h32`, `crnn_amount_h32` (ours) | trained from scratch on synth v1 train crops | ours, no third-party weights | No pretrained backbone. |
| PARSeq | not used | (Apache-2.0 repo) | Skipped: needs `torch.hub` executing the repo's code plus a pytorch-lightning dependency; not worth it within the budget. |

Checked 2026-09-25 via the HF API (`cardData.license` absent for both small checkpoints; `microsoft/trocr-base-handwritten` is tagged `license:mit`) and the unilm repo LICENSE (MIT).

## Evaluation-only data (never trained on, never copied off vega)
| Set | Licence | Use |
|---|---|---|
| SSBI real handwriting crops (78 labelled) | CC BY-NC 4.0 | `real_ssbi_scoring.py` sanity check |
| ORAND-CAR-2014 test crops (1,000 sampled) | CC BY-NC-ND 4.0 | `real_car_scoring.py` courtesy-amount check |
