"""Final methods x fields comparison on eval, plus val-frozen confidence thresholds (the app's real operating point).

Tables written to `reports/final/final_comparison.md` (+ JSON):
- accuracy (and CER) per method x field on eval subsets: all ok rows, hard, handwritten, too_small;
- oracle gating on eval: max coverage at >= 95 / 98% accuracy (threshold chosen on eval itself, optimistic);
- agreement rule (methods with `+xcheck`): fill the courtesy amount only when it agrees with the words line;
  no tuned threshold, so it is immune to val->eval calibration shift;
- frozen gating: threshold chosen on VAL (max coverage with val accuracy >= target), applied unchanged to
  eval; reports eval coverage and eval accuracy at that threshold. This is what the app would ship.

Methods are passed as prediction-file stems present in both `predictions/val/` and `predictions/eval/`.

Run: python -m experiments.field_reading.metrics.final_comparison --methods tesseract_tuned__loc=oracle ...
"""

import argparse
import json
import logging

import numpy as np
import pandas as pd

from experiments.field_reading.field_gating.amount_cross_check import AGREEMENT_CONFIDENCE_FLOOR
from experiments.field_reading.config import PREDICTIONS_ROOT, REPORTS_ROOT, TARGET_FIELD_NAMES
from experiments.field_reading.metrics.breakdowns import summarize_per_field_with_total
from experiments.field_reading.metrics.confidence_gating import build_gating_operating_points
from experiments.field_reading.metrics.report_builder import build_methods_by_fields_frame, select_headline_rows
from experiments.field_reading.metrics.report_tables import methods_by_fields_markdown
from experiments.field_reading.metrics.score_predictions import score_prediction_files

logger = logging.getLogger(__name__)

EVAL_SUBSETS = ("all", "hard", "handwritten", "too_small")
FROZEN_ACCURACY_TARGETS = (0.95, 0.98)
OUTPUT_DIRECTORY = REPORTS_ROOT / "final"


def val_threshold_for_target(val_rows: pd.DataFrame, accuracy_target: float) -> float | None:
    """Lowest confidence threshold whose val fill set keeps accuracy >= target (max coverage); None if unreachable."""
    confidences = val_rows.confidence.astype(float).fillna(-np.inf).to_numpy()
    points = build_gating_operating_points(confidences, val_rows.field_correct.to_numpy(bool), val_rows.covered.to_numpy(bool))
    passing = points[points.accuracy >= accuracy_target - 1e-12]
    if passing.empty:
        return None
    return float(passing.loc[passing.coverage.idxmax()].threshold)


def apply_threshold(eval_rows: pd.DataFrame, threshold: float | None) -> tuple[float, float | None]:
    """(coverage, accuracy on filled) on eval when filling only covered rows with confidence >= threshold."""
    if threshold is None or not len(eval_rows):
        return 0.0, None
    filled = eval_rows.covered.astype(bool) & (eval_rows.confidence.astype(float) >= threshold)
    coverage = float(filled.mean())
    accuracy = float(eval_rows.field_correct[filled].mean()) if filled.any() else None
    return coverage, accuracy


def frozen_gating_table(val_scored: pd.DataFrame, eval_scored: pd.DataFrame, handwritten_only: bool) -> pd.DataFrame:
    """Per method x field: val-chosen threshold and the eval coverage / accuracy it yields, per accuracy target."""
    records = []
    val_ok, eval_ok = select_headline_rows(val_scored, "all"), select_headline_rows(eval_scored, "all")
    if handwritten_only:
        val_ok, eval_ok = val_ok[val_ok.handwritten.astype(bool)], eval_ok[eval_ok.handwritten.astype(bool)]
    for (method_label, field_name), val_group in val_ok.groupby(["method_label", "field_name"], sort=True):
        eval_group = eval_ok[(eval_ok.method_label == method_label) & (eval_ok.field_name == field_name)]
        record = {"method_label": method_label, "field_name": field_name, "n_eval": len(eval_group)}
        for target in FROZEN_ACCURACY_TARGETS:
            threshold = val_threshold_for_target(val_group, target)
            coverage, accuracy = apply_threshold(eval_group, threshold)
            record[f"threshold@{target:.2f}"] = threshold
            record[f"eval_coverage@{target:.2f}"] = coverage
            record[f"eval_accuracy@{target:.2f}"] = accuracy
        records.append(record)
    return pd.DataFrame(records)


def frozen_gating_markdown(table: pd.DataFrame) -> str:
    """Cells: eval coverage% / eval accuracy% at the val-frozen threshold, per target."""
    lines = []
    for target in FROZEN_ACCURACY_TARGETS:
        cell = table.apply(lambda r: f"{100 * r[f'eval_coverage@{target:.2f}']:.1f} / "
                                     + (f"{100 * r[f'eval_accuracy@{target:.2f}']:.1f}" if pd.notna(r[f'eval_accuracy@{target:.2f}']) else "-"), axis=1)
        pivot = table.assign(cell=cell).pivot(index="method_label", columns="field_name", values="cell")
        pivot = pivot[[f for f in TARGET_FIELD_NAMES if f in pivot.columns]].fillna("-")
        lines.append(f"\n**Val-frozen threshold for >= {100 * target:.0f}% val accuracy** (eval coverage% / eval accuracy%)\n")
        lines.append("| method | " + " | ".join(pivot.columns) + " |")
        lines.append("| :-- |" + " --: |" * len(pivot.columns))
        lines.extend(f"| {index} | " + " | ".join(row) + " |" for index, row in pivot.iterrows())
    return "\n".join(lines)


def agreement_rule_markdown(eval_scored: pd.DataFrame) -> str:
    """Courtesy-amount coverage / accuracy on eval when filling only numeric-words agreements (+xcheck methods)."""
    ok_rows = select_headline_rows(eval_scored, "all")
    amount_rows = ok_rows[(ok_rows.field_name == "amount_numeric") & ok_rows.method_label.str.contains(r"\+xcheck", regex=True)]
    lines = ["| method | rows | coverage% | accuracy% |", "| :-- | :-- | --: | --: |"]
    for method_label, group in amount_rows.groupby("method_label", sort=True):
        for label, rows in (("all", group), ("handwritten", group[group.handwritten.astype(bool)]), ("printed", group[~group.handwritten.astype(bool)])):
            agreeing = rows.confidence.astype(float) >= AGREEMENT_CONFIDENCE_FLOOR
            accuracy = f"{100 * rows.field_correct[agreeing].mean():.2f}" if agreeing.any() else "-"
            lines.append(f"| {method_label} | {label} | {100 * agreeing.mean():.1f} | {accuracy} |")
    return "\n".join(lines)


def main() -> None:
    """Score all methods on val and eval once and write the final comparison."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--run-name", default="final_comparison")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    eval_scored, _ = score_prediction_files([PREDICTIONS_ROOT / "eval" / f"{m}.jsonl" for m in arguments.methods], "eval")
    val_scored, _ = score_prediction_files([PREDICTIONS_ROOT / "val" / f"{m}.jsonl" for m in arguments.methods], "val")
    sections, payload = [f"# Field reading: final comparison (eval, {len(arguments.methods)} methods)\n",
                         "acc% = correct / all rows in the slice (blank = wrong). Oracle localization unless the method id says otherwise.\n"], {}
    for subset_name in EVAL_SUBSETS:
        summary = summarize_per_field_with_total(select_headline_rows(eval_scored, subset_name))
        accuracy = build_methods_by_fields_frame(summary, "accuracy")
        cer = build_methods_by_fields_frame(summary, "mean_cer")
        n_rows = int(summary.groupby("method_label").n.max().max()) if "n" in summary else None
        sections.append(f"\n## Eval accuracy %, subset `{subset_name}`\n\n" + methods_by_fields_markdown(accuracy))
        sections.append(f"\n### Mean CER, subset `{subset_name}`\n\n" + methods_by_fields_markdown(cer, value_formatter=lambda v: f"{v:.3f}" if pd.notna(v) else "-"))
        payload[subset_name] = {"accuracy": accuracy.to_dict(), "mean_cer": cer.to_dict(), "n": n_rows}
    for label, handwritten_only in (("all ok rows", False), ("handwritten ok rows", True)):
        frozen = frozen_gating_table(val_scored, eval_scored, handwritten_only)
        sections.append(f"\n## Confidence gating with thresholds frozen on val, {label}\n" + frozen_gating_markdown(frozen))
        payload[f"frozen_gating__{'handwritten' if handwritten_only else 'all'}"] = frozen.to_dict("records")
    if any("+xcheck" in method for method in arguments.methods):
        sections.append("\n## Courtesy amount, agreement rule (fill only when numeric and words agree), eval ok rows\n\n" + agreement_rule_markdown(eval_scored))
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIRECTORY / f"{arguments.run_name}.md").write_text("\n".join(sections) + "\n")
    (OUTPUT_DIRECTORY / f"{arguments.run_name}.json").write_text(json.dumps(payload, indent=1, default=str))
    logger.info("wrote %s", OUTPUT_DIRECTORY / f"{arguments.run_name}.md")


if __name__ == "__main__":
    main()
