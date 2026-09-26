# js/fields/ — the field gate (spec section 5, stage 7)

Turns the worker's raw reads for one check into what the review grid shows: per field `confident` (plain), `unsure` (amber, value shown) or `blank` (amber, empty). Pure JS, main thread and Node; no DOM, no storage, no OpenCV.

| Module | Owns |
|---|---|
| `field_gating.js` | `gateCheckFields(rawReads, { knownPayerNames, knownPayeeNames, todayIso })`: THE contract between the worker (`js/pipeline/fields/`) and the grid (`js/review/`). Read its docstring for both shapes. |
| `field_gating_config.js` | Every threshold, in one place, marked PROVISIONAL (synthetic val tuning; the gold photo set must re-tune them). |
| `money_parsing.js` | Courtesy box and legal line to cents; two-decimal formatting. Port of `experiments/field_reading/metrics/money_amount_parsing.py`. |
| `date_parsing.js` | Check dates to ISO, the plausible window, display formats. Port of `metrics/date_parsing.py`. |
| `fuzzy_matching.js` | rapidfuzz `WRatio` port (payee snap), Levenshtein (payer autocomplete snap), `normalize_free_text`. |

## Rules (experiments/field_reading/README.md section 5, followed exactly)
- Amount is confident ONLY when the courtesy box and the legal line parse to the same cents; otherwise the courtesy value is unsure, with "the written amount reads differently" when the legal line parsed to something else. The legal line is never shown.
- Handwritten fields (style classifier p > 0.5) are blank unless the opt-in handwriting reader read them, and then at most unsure.
- Payee is snapped to the operator's co-op list (WRatio >= 60) and always unsure; blank without a list. Payer snaps to confirmed names within a small edit distance and turns unsure when the snap changed the text.
- Printed memo is never confident; printed date needs the plausible window (a year either side of today).

## Invariants
- The parsers are line-for-line ports. Change the Python in `experiments/field_reading/` first, then mirror it here, then run `app/tests/parity/dump_gating_reference.py` + `run_gating_parity.mjs` (every eval string must match).
- Thresholds live only in `field_gating_config.js`; `app/tests/field_reading/python_field_gate.py` parses them out of that file, so the Python mirror can never disagree on a number.
- Unit tests: `node app/tests/unit/test_field_gating.mjs`.
