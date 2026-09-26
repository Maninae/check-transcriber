# metrics/: scoring detector predictions against the synthetic GT

Scores a predictions file (`predictions/detected_check.py` format) against a split's `SceneAnnotation`s and writes `metrics.json` (everything, including per-scene and per-check records for failure mining) and `metrics.md` (headline tables plus one table per breakdown).

## Entry points

- CLI: `python -m experiments.detection.metrics.score_predictions --predictions <file.json> --split val|eval [--limit N] [--score-threshold 0.0] --output-dir <dir>`
- In-process (tuning loops): `score_predictions_against_split(predictions_by_scene_id, scene_annotations, score_threshold=..., detector_config=..., include_records=False)` then `headline_summary_line(metrics)`.
- Sanity check: `python -m experiments.detection.metrics.ground_truth_as_predictions --split val --output <file>` writes GT corners as predictions. Scoring it must give IoU vs corner quad 1.0, corner error 0, orientation 100%, and IoU vs outline ~0.992 on val (deformed outlines are not quads; the worst check is ~0.904, so recall@0.9 has almost no headroom for heavily deformed checks).

## Modules (dependencies flow downward)

| Module | Responsibility |
|---|---|
| `score_predictions.py` | public API + CLI; wires the modules below |
| `metrics_report_writing.py` | JSON/markdown writing, `headline_summary_line`, per-column rounding |
| `attribute_breakdowns.py` | recall/precision/IoU/corner error grouped by scene and check attributes |
| `split_aggregation.py` | pooled P/R/F1, IoU and corner-error statistics, scene rates |
| `scene_scoring.py` | one scene: threshold, clip, match, fill records |
| `detection_matching.py` | IoU matrix and greedy one-to-one matching |
| `quadrilateral_geometry.py` | clipped/repaired polygons, IoU, cyclic corner alignment |
| `metric_records.py` | record dataclasses and shared thresholds (leaf) |

## Definitions (invariants)

- Every polygon is clipped to `[0, W] x [0, H]` before IoU. Invalid predicted quads are repaired with `shapely.make_valid`; degenerate or non-finite quads score IoU 0.
- Matching is against the GT **outline**, greedy by descending IoU, one-to-one, at 0.5 / 0.75 / 0.9 (reported) and 0.1 (diagnostic, so near-misses still get a per-check record). Stricter matchings are subsets of looser ones.
- Localization stats (IoU, corner error, orientation) aggregate only checks matched at 0.5.
- Corner error: mean Euclidean px distance at the best of the 4 cyclic shifts of the predicted order, over GT corners inside the image only (`corners_used_for_error`). Shift 0 = orientation correct (scored only when `orientation_known`); shift 0 or 2 = axis correct (scored for all).
- The score threshold drops predictions before every metric, counts included.
- Breakdowns: scene attributes get precision (false positives belong to a scene); check attributes do not. Deformation groups overlap (a fold + curl check counts in both).
- A scene absent from the predictions file counts as zero predictions and is tallied, not skipped.

## Tests

`experiments/detection/tests/test_metrics_*.py` on hand-built scenes from `tests/metrics_test_scenes.py`; no dataset needed.
