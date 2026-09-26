"""Score reading prediction files against ground truth and write a JSON + markdown report.

Usage (from the worktree root):
    python -m experiments.field_reading.metrics.score_predictions \
        --split val \
        --predictions path/to/methodA__loc=oracle.jsonl path/to/methodB__loc=oracle.jsonl \
        [--subset all|hard|handwritten|handwritten_degraded|printed_degraded|too_small] [--run-name NAME] [--output-dir DIR] [--only-predicted-rows]

- Prediction rows follow PLAN.md's contract and join on `row_key`; each (method, localization)
  pair is one method label `<method>__loc=<localization>`.
- Scored ground truth = target-field rows with status ok or too_small; a missing prediction is blank.
  Occluded / out-of-frame rows are excluded and counted. `--only-predicted-rows` restricts scoring
  to row keys some prediction file covers (for quick subset runs).
- Writes `<output-dir>/<run-name>.json` and `.md` (default output dir: reports root / split).
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from experiments.field_reading.config import REPORTS_ROOT
from experiments.field_reading.data_access.field_manifest import load_field_rows
from experiments.field_reading.metrics.breakdowns import summarize_per_field_with_total
from experiments.field_reading.metrics.hard_set import HARD_SET_FLAG_NAMES, load_or_build_hard_set_flags
from experiments.field_reading.metrics.report_builder import (
    SUBSET_NAMES,
    build_methods_by_fields_frame,
    build_report,
    select_headline_rows,
)
from experiments.field_reading.metrics.row_scoring import score_joined_rows

logger = logging.getLogger(__name__)

REQUIRED_PREDICTION_KEYS = ["row_key", "field_name", "method", "localization", "pred_text", "confidence"]
SCORED_STATUSES = ["ok", "too_small"]
HARD_SET_JOIN_COLUMNS = ["row_key", "in_hard_set", "contrast_statistic", "handwritten_degraded", "printed_degraded",
                         *[name for name in HARD_SET_FLAG_NAMES if name != "handwritten"]]
HARD_SET_BOOLEAN_COLUMNS = ["in_hard_set", "handwritten_degraded", "printed_degraded", *HARD_SET_FLAG_NAMES]


def load_prediction_files(prediction_paths: list[Path]) -> pd.DataFrame:
    """All prediction rows with a `method_label`; fails loudly on missing keys or duplicate row keys per method."""
    frames = []
    for prediction_path in prediction_paths:
        predictions = pd.read_json(prediction_path, lines=True, dtype={"pred_text": str})
        missing_keys = [key for key in REQUIRED_PREDICTION_KEYS if key not in predictions.columns]
        if missing_keys:
            raise ValueError(f"{prediction_path} lacks prediction keys {missing_keys}")
        frames.append(predictions)
    all_predictions = pd.concat(frames, ignore_index=True)
    all_predictions["method_label"] = all_predictions.method.astype(str) + "__loc=" + all_predictions.localization.astype(str)
    all_predictions["pred_text"] = all_predictions.pred_text.fillna("").astype(str)
    duplicated = all_predictions.duplicated(["method_label", "row_key"])
    if duplicated.any():
        raise ValueError(f"{int(duplicated.sum())} duplicate (method, row_key) predictions, e.g. "
                         f"{all_predictions[duplicated].iloc[0][['method_label', 'row_key']].tolist()}")
    return all_predictions


def load_scored_ground_truth(split_name: str, hard_set_flags_path: Path | None) -> tuple[pd.DataFrame, dict[str, int]]:
    """GT rows with status ok/too_small joined with hard-set flags, and counts of the excluded statuses."""
    field_rows = load_field_rows(split_name)
    excluded_row_counts = {status: int(count) for status, count in field_rows[~field_rows.status.isin(SCORED_STATUSES)].status.value_counts().items()}
    scored_ground_truth = field_rows[field_rows.status.isin(SCORED_STATUSES)]
    hard_set_flags = load_or_build_hard_set_flags(split_name, hard_set_flags_path)
    missing_columns = [name for name in HARD_SET_JOIN_COLUMNS if name not in hard_set_flags.columns]
    if missing_columns:
        raise ValueError(f"hard-set flag cache lacks {missing_columns}; rebuild it with "
                         f"`python -m experiments.field_reading.metrics.hard_set --split {split_name}`")
    scored_ground_truth = scored_ground_truth.merge(hard_set_flags[HARD_SET_JOIN_COLUMNS], on="row_key", how="left", validate="one_to_one")
    unflagged_count = int(scored_ground_truth.in_hard_set.isna().sum())
    if unflagged_count:
        logger.warning("%d scored rows have no hard-set flags (stale cache?); treating them as not hard", unflagged_count)
    for column_name in HARD_SET_BOOLEAN_COLUMNS:
        scored_ground_truth[column_name] = scored_ground_truth[column_name].fillna(False).astype(bool)
    return scored_ground_truth.reset_index(drop=True), excluded_row_counts


def score_prediction_frame(ground_truth_rows: pd.DataFrame, predictions: pd.DataFrame, only_predicted_rows: bool) -> pd.DataFrame:
    """Scored rows for every method label: GT rows left-joined with that method's predictions (missing = blank)."""
    if only_predicted_rows:
        ground_truth_rows = ground_truth_rows[ground_truth_rows.row_key.isin(set(predictions.row_key))]
    unmatched_count = int((~predictions.row_key.isin(set(ground_truth_rows.row_key))).sum())
    if unmatched_count:
        logger.info("%d prediction rows have no scored GT row (excluded status or other split); ignored", unmatched_count)
    scored_frames = []
    for method_label, method_predictions in predictions.groupby("method_label", sort=True):
        joined = ground_truth_rows.merge(method_predictions[["row_key", "pred_text", "confidence"]], on="row_key", how="left")
        joined["method_label"] = method_label
        joined["pred_text"] = joined.pred_text.fillna("")
        joined["confidence"] = pd.to_numeric(joined.confidence, errors="coerce")
        scored_frames.append(score_joined_rows(joined))
        logger.info("scored %s: %d rows", method_label, len(joined))
    return pd.concat(scored_frames, ignore_index=True)


def score_prediction_files(prediction_paths: list[Path], split_name: str, only_predicted_rows: bool = False,
                           hard_set_flags_path: Path | None = None) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load, join and score prediction files; returns the scored rows and excluded-status counts."""
    ground_truth_rows, excluded_row_counts = load_scored_ground_truth(split_name, hard_set_flags_path)
    return score_prediction_frame(ground_truth_rows, load_prediction_files(prediction_paths), only_predicted_rows), excluded_row_counts


def build_methods_by_fields_table(prediction_paths: list[Path], split_name: str, subset_name: str = "all",
                                  metric_name: str = "accuracy", hard_set_flags_path: Path | None = None) -> pd.DataFrame:
    """Methods x fields table of one metric (accuracy, mean_cer, coverage, ...) across prediction files."""
    scored_rows, _ = score_prediction_files(prediction_paths, split_name, hard_set_flags_path=hard_set_flags_path)
    headline_summary = summarize_per_field_with_total(select_headline_rows(scored_rows, subset_name))
    return build_methods_by_fields_frame(headline_summary, metric_name)


def write_report(report: dict, markdown_text: str, output_directory: Path, run_name: str) -> tuple[Path, Path]:
    """Write `<run_name>.json` and `<run_name>.md`; returns both paths."""
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path, markdown_path = output_directory / f"{run_name}.json", output_directory / f"{run_name}.md"
    json_path.write_text(json.dumps(report, indent=1))
    markdown_path.write_text(markdown_text)
    return json_path, markdown_path


def main() -> None:
    """CLI entry point (see module docstring)."""
    parser = argparse.ArgumentParser(description="Score field-reading predictions against ground truth.")
    parser.add_argument("--split", required=True, choices=["train", "val", "eval"])
    parser.add_argument("--predictions", required=True, nargs="+", type=Path)
    parser.add_argument("--subset", default="all", choices=SUBSET_NAMES)
    parser.add_argument("--run-name", default=None, help="default: first prediction file stem (+ __subset=X)")
    parser.add_argument("--output-dir", type=Path, default=None, help="default: reports root / split")
    parser.add_argument("--only-predicted-rows", action="store_true")
    parser.add_argument("--hard-set-flags", type=Path, default=None, help="default: cached flags for the split")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_name = arguments.run_name or arguments.predictions[0].stem + ("" if arguments.subset == "all" else f"__subset={arguments.subset}")
    scored_rows, excluded_row_counts = score_prediction_files(arguments.predictions, arguments.split,
                                                              arguments.only_predicted_rows, arguments.hard_set_flags)
    report, markdown_text = build_report(scored_rows, arguments.split, arguments.subset, run_name,
                                         [str(path) for path in arguments.predictions], excluded_row_counts)
    json_path, markdown_path = write_report(report, markdown_text, arguments.output_dir or REPORTS_ROOT / arguments.split, run_name)
    logger.info("wrote %s and %s", json_path, markdown_path)


if __name__ == "__main__":
    main()
