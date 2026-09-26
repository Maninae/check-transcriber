# Model licences (U4 learned recognizers)

| Model | Source | Licence | Notes |
|---|---|---|---|
| `microsoft/trocr-small-handwritten` | https://huggingface.co/microsoft/trocr-small-handwritten | MIT (source repo microsoft/unilm, `LICENSE`); the HF model card declares no licence field | Fine-tuned by Microsoft on IAM. IAM's database terms are non-commercial research only, so shipping these weights (or our fine-tune of them) in a commercial product needs a legal look. |
| `microsoft/trocr-small-printed` | https://huggingface.co/microsoft/trocr-small-printed | MIT (microsoft/unilm); no licence field on the HF card | Fine-tuned on SROIE receipts (outputs upper case). |
| `trocr_small_handwritten_ft` (ours) | fine-tune of trocr-small-handwritten on synth v1 train crops | inherits the above | Weights under `/Volumes/vega/ai-models/field-reading/recognizers/trocr_small_handwritten_ft/`. |
| `crnn_general_h32`, `crnn_amount_h32` (ours) | trained from scratch on synth v1 train crops | ours, no third-party weights | No pretrained backbone. |
| PARSeq | not used | (Apache-2.0 repo) | Skipped: needs `torch.hub` executing the repo's code plus a pytorch-lightning dependency; not worth it within the budget. |

Checked 2026-09-25 via the HF API (`cardData.license` absent for both small checkpoints; `microsoft/trocr-base-handwritten` is tagged `license:mit`) and the unilm repo LICENSE (MIT).
