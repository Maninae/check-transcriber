# Field reading: results and recommendation (Sep 26 2026)

**Localization is solved; printed reading is good; real handwriting is only partly readable, and synthetic v1 handwriting cannot tell us which reader handles it.** The last point decides the design: the models that ace v1's font-rendered handwriting collapse on real handwriting, so every handwriting call below rests on real crops (SSBI, ORAND-CAR), and the gating leans on cross-checks rather than confidence thresholds.

## 1. Field localization (eval, held-out templates, status ok)

| method | mean IoU | hit rate (IoU >= 0.5) | false box on absent memo | end-to-end Tesseract accuracy |
|---|--:|--:|--:|--:|
| ground-truth crop (ceiling) | 1 | 100% | - | 41.6% |
| **segnet (learned, 12.8 MB, 41 ms)** | **0.887** | **99.99%** | **0.0%** | **40.1%** |
| layout prior, family known (oracle) | 0.757 | 92.9% | 22.6% | - |
| layout prior, family inferred from aspect | 0.701 | 86.1% | 18.0% | 35.4% |

Priors grab printed labels ("DATE", "PAY TO THE ORDER OF"), the payee address and the routing fraction; segnet is flat across all 6 families (IoU 0.875-0.892), handwritten or printed, and on too-small text.

## 2. Reading (eval, end to end with segnet boxes; accuracy = value-correct / all rows)

| method | payer | payee (raw text) | courtesy amount | legal line | date | memo | check # | all |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Tesseract (tuned + cleanup) | 71.0 | 16.0 | 41.9 | 19.3 | 32.2 | 21.7 | 88.4 | 40.1 |
| TrOCR-small-handwritten, zero-shot | 51.7 | 15.9 | 5.2 | 9.0 | 12.7 | 9.7 | 71.1 | 24.1 |
| CRNN (+ amount CRNN) | 71.2 | 18.4 | 80.3 | 55.9 | 76.9 | 74.2 | 98.3 | 65.9 |
| CRNN + amount/words cross-check | 71.2 | 18.4 | 82.4 | 55.9 | 76.9 | 74.2 | 98.3 | 66.2 |
| style router (CRNN printed, TrOCR handwritten) | 74.3 | 22.8 | 79.7 | 32.0 | 47.2 | 35.5 | 98.3 | 54.0 |

- Hard set (degraded handwriting or small/low-contrast/money-order print, 50% of rows): CRNN 68.9%, Tesseract 42.8%. Too-small text (under 14 px in the photo, reported separately): CRNN 56.1%, Tesseract 21.3%.
- Handwritten rows only, on synthetic handwriting: CRNN 42.8%, Tesseract 6.1%, TrOCR 10.0%. **These are font-recall numbers** (see section 3).

## 3. Real handwriting: the deciding evidence

| reader | SSBI amount (n=22) | SSBI legal line (25) | SSBI date (18) | SSBI payee (13) | ORAND-CAR amounts A / B (500 each) |
|---|--:|--:|--:|--:|--:|
| Tesseract | 0.00 | 0.00 | 0.00 | 0.00 | 0.07 / 0.03 |
| CRNN (trained on v1 fonts) | 0.18 | 0.00 | 0.06 | 0.00 | 0.47 / 0.54 (amount CRNN) |
| TrOCR-small-handwritten, grey input | **0.82** | **0.76** | 0.00 (digits right in any order: 0.22) | **0.62** | 0.50 / 0.29 |
| higher-confidence of TrOCR and amount CRNN | 0.59 | 0.72 | - | 0.46 | **0.57 / 0.56** |

- SSBI: 78 crops hand-transcribed by us, few writers, repeated strings (a sanity set). ORAND-CAR: real bank-check courtesy amounts, digits-only scoring. Both non-commercial licences, eval only.
- The CRNN learned the 24 train fonts: great on synthetic handwriting, nonsense on real cursive words, respectable on real digits. TrOCR is the reverse. Real dates defeat everything we have.

## 4. Gating: what can be filled safely

- **Courtesy amount, agreement rule**: fill only when the courtesy box and the legal line parse to the same value. Eval: 54.5% of amounts filled at 100.00% accuracy (printed 73.5%, handwritten 33.7%). No tuned threshold, so it survives calibration shift. On real checks accuracy stays high by construction (two independent reads must fail identically); coverage is unmeasured (no real set pairs both lines).
- **Payee, snap to the known list** (the nonprofit's co-ops in the app; a 4-name pool here, chance 25%): style router identifies 71.0% of payees at 95.4% precision (handwritten 62.5% / 95.7%); Tesseract 60.9% / 96.2%.
- **Confidence thresholds do not transfer**: a threshold set for 95% accuracy on val gives 88-94% on eval (new templates and fonts only). Thresholds for the app must be set on real photos (the gold set), not synth.
- Check number: CRNN fills 99.9% at 98.5% accuracy with a val-frozen threshold.

## 5. Recommendation for milestone 4

| field | read with | fill rule | default state |
|---|---|---|---|
| check number | CRNN | confidence gate (val 98%) | filled |
| payer (printed) | CRNN | confidence gate + payer autocomplete snap (spec 4.3) | filled / unsure |
| courtesy amount | amount CRNN (+ TrOCR when the handwriting model is enabled; take the higher confidence) | filled only when it agrees with the legal line; otherwise shown unsure | unsure unless agreement |
| legal line | CRNN for printed, TrOCR for handwritten (style classifier routes) | never shown; feeds the amount cross-check | - |
| payee | style-routed reader, then snap to the configured co-op list | snapped name when WRatio >= 60, else the raw read | unsure (blank only when handwritten and the handwriting reader is off) |
| date | CRNN when printed | printed: confidence gate + plausible-date window | printed: filled; **handwritten: blank** (offer the email date) |
| memo | CRNN when printed | confidence gate | printed: unsure; **handwritten: blank** |
| payer (handwritten, e.g. money-order FROM line) | - | - | **blank** |

- **Opt-in handwriting reader (app decision, Sep 26 2026):** when the operator turns it on, TrOCR reads every field the style classifier calls handwritten (payer, date, memo, payee, courtesy amount, legal line, check number), and every value resting on a TrOCR read is capped at unsure, including an amount whose two reads agree. With the reader off, a handwritten legal line counts as unread: it neither confirms nor contradicts the courtesy amount.
- Default download: segnet + CRNN + amount CRNN + style classifier, about 30 MB. TrOCR (128 MB) as an opt-in "read handwriting" download (spec's optional later model), pending the IAM licence decision.
- Browser details: `export/BROWSER_NOTE.md`.

## 6. Gaps
- No real handwriting on our own layouts; SSBI is tiny and repetitive, dates there are DD/MM. The planned hand-filled mock checks are the needed eval set, and the place to set every threshold.
- The CRNN is unvalidated on real printed checks.
- Synth v2 handwriting needs stroke-level variation and many more writers; fine-tuning on v1 handwriting teaches fonts.
- TrOCR weights are fine-tuned on IAM (non-commercial research terms): needs a licence decision before shipping.
- Real handwritten dates and memos have no working reader yet.

## Where things are
- Reports: `/Volumes/vega/datasets/check-transcriber/field-reading/reports/` (`final/final_comparison.md`, `localization/`, `learned/`, `eval/`, `gallery/`).
- Predictions: `.../field-reading/predictions/{val,eval}/<method>__loc=<localization>.jsonl`.
- Models: `/Volumes/vega/ai-models/field-reading/` (`localization/`, `recognizers/onnx/`, `tessdata/`).
- Module map: `CLAUDE.md`; plan and log: `PLAN.md`.
