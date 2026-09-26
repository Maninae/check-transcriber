# Field reading: plan and log

Goal: from a rectified 1600 px check crop, locate and read payer name, payee, courtesy amount, legal (words) amount, date, memo and check number; fill confident fields, blank unsure ones (spec 4.3, 5 stages 5-7). MICR is never read. Handwriting is the hard part and gets its own report.

## Data
- val/eval: synth v1 `ocr/` export on vega (rectified crops from ground-truth corners, field crops, one row per field). Scored rows: `status == ok` (primary) and `too_small` (text under 14 px tall in the photo, reported separately). Occluded / out-of-frame rows are excluded and counted.
- train: re-cut by `data_access/train_crop_export.py` into the identical schema under `/Volumes/vega/datasets/check-transcriber/field-reading/derived/` (verified: 138/138 boxes and statuses identical to synth's own val rows, pixel diff under 1.3/255 = JPEG noise).
- Canonical ground truth (`amount_cents`, `date_iso`, `check_number`) is joined from annotations by `data_access/field_manifest.py`.
- Held out by construction: eval templates, backgrounds, payees, banks and all 5 eval handwriting fonts never occur in train. Model selection uses val only; eval is scored once per final method.

## Contracts (fixed before fan-out)
- Row key: `<scene_id>__check=<i>__field=<name>` (`field_manifest.make_row_key`).
- Reading predictions: `field-reading/predictions/<split>/<method>__loc=<localization>.jsonl`, one row per scored field row: `{row_key, scene_id, check_index, field_name, method, localization, pred_text, confidence, pred_box, latency_ms}`. `pred_text == ""` means blank. `confidence` in [0, 1] (null if the method has none). `localization` is `oracle` (synth's field crop) or a localizer id.
- Localization predictions: `field-reading/predictions/localization/<split>/<method>.jsonl`, one row per (check, target field), present or not: `{row_key, scene_id, check_index, field_name, pred_box | null, confidence}` in 1600 px check-crop pixel-edge coordinates.
- End-to-end readers crop with `data_access/field_crop.crop_field_from_check` (same margin rule as synth's crops).
- MPS: one heavy job at a time for this whole area. Take `mkdir /Volumes/vega/datasets/check-transcriber/field-reading/logs/MPS_LOCK` before an MPS training or bulk inference run, remove it (`rmdir`) after; wait if it exists. Check `vm_stat` first; the detector owner trains concurrently.

## Units
| # | Unit | Owner | Verified means |
|---|------|-------|----------------|
| U0 | Data layer + train crop export | lead | export parity test vs synth val rows; train manifest row counts sane |
| U1 | Metrics harness (`metrics/`) | Opus builder | pytest green; parsers recover canonical value from >= 99.5% of GT texts; a toy prediction file produces JSON + md with every breakdown and the gating table |
| U2 | Field localization (`field_localization/`): layout prior (+ ink refinement), learned localizer | Opus builder | per-field IoU / hit rate on eval for each method, boxes drawn on 6 crops and read by eye |
| U3 | Tesseract baselines (`ocr_baselines/`) | lead | config search on val subset logged; eval predictions scored |
| U4 | Modern recognizers (`learned/`): TrOCR zero-shot and fine-tuned, printed recognizer, digit path for courtesy amount | Opus builder | val-selected, eval scored once, licences recorded |
| U5 | Hard set + comparison + gating curves (`metrics/`) | lead | tables on eval and hard set, per field x method |
| U6 | ONNX export, equivalence on 20 crops, size, CPU latency, browser note (`export/`) | lead or builder | max abs logit diff and identical decoded text on 20 crops |
| U7 | Contact sheets + failure gallery (`gallery/`) | lead | read by eye, each sheet < 3 MB |
| U8 | Recommendation + report to main | lead | |

## Log
- 2026-09-25 21:05 U0: data layer written; export parity verified; full train export launched (4 workers).
- 21:25 U1 metrics harness landed (81 tests; GT parsers reproduce canonical values on 100% of rows, all splits). Hard set narrowed at 21:45 to handwritten_degraded OR printed_degraded (50% of ok rows); handwritten slice and its gating table reported separately.
- 21:30 U3 Tesseract: tesserocr 5.5.1 with the app's eng 4.0.0_best_int. Field text cleanup (edge artifacts, money-token extraction) lifted printed amount 0.41 -> 0.74 and printed date 0.47 -> 0.75 on val. Config search re-run with the harness's correctness rule on 240 rows/field.
- 21:50 Tesseract.js 7 parity vs tesserocr on 20 val crops: 15/20 identical text; all 5 differences are low-confidence handwritten misreads; confidences within a few points.
- 22:05 Tesseract eval (oracle crops): tuned 41.6% all fields; payer 74.0, check number 89.5; handwritten coverage at 95% accuracy ~0 on every field. Real SSBI handwriting: 0/78 exact.
- 22:05 Machine thrashing (swap ~17 GB): all our jobs capped at 2 workers; GPU yielded to the detection owner until ~23:30 on the coordinator's request; segnet checkpointed and paused, CRNN queued.
- Real handwriting mini-set: 78 SSBI crops transcribed by eye (`field-reading/real_ssbi/`, LABELS.md), 5 excluded as ambiguous/illegible.
- 00:25 U2 done: segnet eval IoU 0.887 / hit 99.99% / 0% false memo boxes vs layout priors 0.757 (family oracle) and 0.701 (size kind). ONNX 12.8 MB, 41 ms CPU, 20/20 identical boxes. End-to-end Tesseract: oracle 41.6%, segnet boxes 40.1%, size-kind prior 35.4%.
- 00:45 General CRNN (2.1M params, 26 min) val exact 0.71, but on real SSBI handwriting it collapses (amount 0.18, words 0.00) while zero-shot TrOCR-small-handwritten reads real handwriting well (amount 0.82, words 0.76). Synth handwriting = font recall. Coordinator ruling: judge handwriting on real crops; no TrOCR fine-tune on v1 handwriting (run stopped).
- 00:55 Added ORAND-CAR-2014 (real check courtesy amounts, CC BY-NC-ND, eval only; 500 per subset sample). Tesseract tuned: 7.0% CAR-A, 3.0% CAR-B digits exact.
- Stage-7 gating added (field_gating/): amount numeric/words cross-check (Tesseract: safe-fill at 98% goes 4.4% -> 20.1% of amounts), payee snap to known list (Tesseract identifies 54% of handwritten payees at 96% precision).
- 03:00 U4 done (CRNN general/amount, TrOCR-hw zero-shot grey, style router, ONNX). ORAND-CAR real amounts: max-conf(TrOCR, amount CRNN) 0.57/0.56. TrOCR fp32 enc + int8 dec 128 MB = fp32 accuracy; full int8 64 MB loses real words 0.76 -> 0.40.
- 03:05 U5-U8: final comparison (reports/final/final_comparison.md), agreement rule 54.5% amounts at 100% (segnet e2e), payee snap, galleries (synth showcase, confident failures, real handwriting), browser note, README recommendation.
