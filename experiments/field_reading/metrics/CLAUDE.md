# metrics/: field-reading scoring harness

Scores reading predictions (PLAN.md contract) against ground truth and writes `reports/<split>/<run>.json` + `.md`.

## Modules (dependency order: parsing -> scoring -> aggregation -> report -> CLI)
| Module | Responsibility |
| --- | --- |
| `money_amount_parsing.py` | courtesy-box text and legal-line words -> integer cents |
| `date_parsing.py` | date text -> ISO (US month-first, 2-digit years = 20YY) |
| `field_value_parsing.py` | `normalize_field_value(field, text)`: THE definition of a correct read; the app mirrors it in JS |
| `text_error_rates.py` | CER / WER via rapidfuzz |
| `row_scoring.py` | per-row covered / field_correct / exact_raw / text_correct / cer / wer |
| `confidence_gating.py` | coverage-vs-accuracy curve, acc at 80/90/95/100% coverage, max coverage at 95/98/99% accuracy |
| `breakdowns.py` | `summarize_scored_rows`, the single aggregation behind every table |
| `hard_set.py` | hard-set flags + low-contrast statistic, cached at `reports/hard_set_flags__split=<s>.jsonl`; also a CLI |
| `report_builder.py` | assembles every report section (dict + markdown) |
| `report_tables.py` | markdown rendering |
| `score_predictions.py` | CLI + `build_methods_by_fields_table` for cross-method comparisons |

## Invariants
- Blank prediction (`pred_text == ""`) = not covered, never correct, CER 1.0. Missing prediction = blank.
- Unparseable prediction = None = wrong. GT value = parsed GT text, falling back to the canonical column.
- Headline = status ok rows only; too_small is always its own section. Occluded / out-of-frame rows are excluded and counted.
- Parsed GT text must match canonical values on >= 99.5% of rows (`tests/test_metrics_gt_self_consistency.py`; currently 100% on train/val/eval).
- Slices over ok rows: `handwritten`; `handwritten_degraded` = handwritten AND (small_text OR low_contrast); `printed_degraded` = printed AND (small_text OR low_contrast OR money_order); `hard` = either degraded slice; easy = clean printed. All are `--subset` values alongside `all` and `too_small`.
- Every md report carries a gating table on handwritten ok rows (the app's real question: how much handwriting can be filled at 95/98% accuracy).
- Metrics score raw predictions: never import reader-side cleanup (e.g. `ocr_baselines/field_text_cleanup.py`) here.
- Low-contrast quartile is per field, per split. Rebuild a stale flag cache with `python -m experiments.field_reading.metrics.hard_set --split <s>`.
