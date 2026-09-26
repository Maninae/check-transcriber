"""Measure corner refinement on a split: corner error before vs after, with breakdowns.

    python -m experiments.detection.refinement.evaluate_refinement --split val --limit 300 --perturbation obb
    python -m experiments.detection.refinement.evaluate_refinement --split val --perturbation jitter --amount 20
    python -m experiments.detection.refinement.evaluate_refinement --split val --predictions preds.json

Inputs are simulated from GT (`obb`, `jitter` N px, `scale` +-S) or, with `--predictions`,
a real detector's predictions file matched to GT by IoU. Results (summary.md,
summary.json, records.jsonl, debug/*.jpg) go under the vega experiments root. Tune on
val only; eval is for the final number.
"""

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from experiments.detection.config.paths import DETECTION_EXPERIMENTS_ROOT
from experiments.detection.dataset.scene_annotations import load_split_scene_annotations
from experiments.detection.predictions.detected_check import load_predictions_file
from experiments.detection.refinement.debug_crop_rendering import render_debug_panels_for_records
from experiments.detection.refinement.refinement_config import CornerRefinementConfig
from experiments.detection.refinement.refinement_evaluation_records import (
    evaluate_scene,
    group_records_for_breakdown,
    summarize_corner_errors,
)

logger = logging.getLogger(__name__)

REFINEMENT_RESULTS_ROOT = DETECTION_EXPERIMENTS_ROOT / "refinement"
DEFAULT_AMOUNT_BY_PERTURBATION = {"obb": 0.0, "jitter": 8.0, "scale": 0.03}
MAXIMUM_WORKER_PROCESSES = 2  # shared 16 GB Mac with a model training


def evaluate_scene_with_arguments(scene_and_detections, perturbation_kind, perturbation_amount, config):
    """Pool-friendly wrapper around `evaluate_scene`."""
    scene, scene_detections = scene_and_detections
    return evaluate_scene(scene, perturbation_kind, perturbation_amount, config, scene_detections)


def format_summary_table(groups: dict[str, list[dict]]) -> str:
    """Markdown table: one row per group, before -> after for each statistic."""
    header = "| group | checks | mean px | median px | p90 px | p95 px | check-mean median | check-mean p90 | check-mean p95 | <2 px % | <5 px % |"
    lines = [header, "|" + "---|" * 11]
    for group_name, records in groups.items():
        before, after = summarize_corner_errors(records, "errors_before"), summarize_corner_errors(records, "errors_after")
        cells = [
            f"{before[key]:.2f} → {after[key]:.2f}" if "px_percent" not in key else f"{before[key]:.0f} → {after[key]:.0f}"
            for key in ("mean", "median", "p90", "p95", "check_mean_median", "check_mean_p90", "check_mean_p95", "checks_below_2px_percent", "checks_below_5px_percent")
        ]
        lines.append(f"| {group_name} | {len(records)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def choose_debug_records(records: list[dict], count: int, random_generator) -> list[dict]:
    """Half worst-after checks, half random ones, so both failures and typical cases are seen."""
    worst = sorted(records, key=lambda record: -max(record["errors_after"]))[: count // 2]
    worst_ids = {id(record) for record in worst}
    remaining = [record for record in records if id(record) not in worst_ids]
    random_picks = [remaining[index] for index in random_generator.choice(len(remaining), size=min(count - len(worst), len(remaining)), replace=False)]
    return worst + random_picks


def parse_arguments() -> argparse.Namespace:
    """CLI flags."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=300, help="first N scenes of the split")
    parser.add_argument("--perturbation", choices=("obb", "jitter", "scale"), default="obb")
    parser.add_argument("--amount", type=float, default=None, help="jitter max px, or scale max fraction")
    parser.add_argument("--predictions", type=Path, default=None, help="real predictions JSON (overrides --perturbation)")
    parser.add_argument("--config-json", default="{}", help="JSON overrides for CornerRefinementConfig fields")
    parser.add_argument("--workers", type=int, default=MAXIMUM_WORKER_PROCESSES)
    parser.add_argument("--debug-count", type=int, default=8)
    parser.add_argument("--tag", default="default")
    return parser.parse_args()


def main() -> None:
    """Run the evaluation and write the results directory."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    arguments = parse_arguments()
    if arguments.split == "eval":
        logger.warning("scoring EVAL: do this once per final configuration, never to tune")
    overrides = json.loads(arguments.config_json)
    config = CornerRefinementConfig(**{key: tuple(value) if isinstance(value, list) else value for key, value in overrides.items()})
    scenes = load_split_scene_annotations(arguments.split, limit=arguments.limit)

    if arguments.predictions is not None:
        _, predictions_by_scene_id = load_predictions_file(arguments.predictions)
        input_name, amount = f"predictions={arguments.predictions.stem}", 0.0
        work_items = [(scene, predictions_by_scene_id.get(scene.scene_id, [])) for scene in scenes]
    else:
        amount = arguments.amount if arguments.amount is not None else DEFAULT_AMOUNT_BY_PERTURBATION[arguments.perturbation]
        input_name = arguments.perturbation + (f"=amount={amount:g}" if arguments.perturbation != "obb" else "")
        work_items = [(scene, None) for scene in scenes]

    worker = partial(evaluate_scene_with_arguments, perturbation_kind=arguments.perturbation, perturbation_amount=amount, config=config)
    # ProcessPoolExecutor raises BrokenProcessPool if a worker is killed (memory pressure)
    # instead of hanging forever like multiprocessing.Pool.
    with ProcessPoolExecutor(max_workers=min(arguments.workers, MAXIMUM_WORKER_PROCESSES)) as executor:
        records = [record for scene_records in executor.map(worker, work_items, chunksize=4) for record in scene_records]

    output_directory = REFINEMENT_RESULTS_ROOT / f"{arguments.split}__{input_name}__n={len(scenes)}__{arguments.tag}"
    output_directory.mkdir(parents=True, exist_ok=True)
    groups = group_records_for_breakdown(records)
    milliseconds = np.array([record["milliseconds"] for record in records])
    table = format_summary_table(groups)
    timing_line = (
        f"refinement time per check (ms, {min(arguments.workers, MAXIMUM_WORKER_PROCESSES)} workers in parallel): "
        f"median {np.median(milliseconds):.1f}, p95 {np.percentile(milliseconds, 95):.1f}; "
        f"quads reverted by guard rails: {sum(record['quad_reverted'] for record in records)}"
    )
    (output_directory / "summary.md").write_text(f"# {output_directory.name}\n\n{timing_line}\n\n{table}\n")
    summary_json = {
        "config": config.to_json_dict(),
        "arguments": {key: str(value) for key, value in vars(arguments).items()},
        "groups": {
            name: {"before": summarize_corner_errors(subset, "errors_before"), "after": summarize_corner_errors(subset, "errors_after")}
            for name, subset in groups.items()
        },
    }
    (output_directory / "summary.json").write_text(json.dumps(summary_json, indent=1))
    with open(output_directory / "records.jsonl", "w") as records_file:
        for record in records:
            records_file.write(json.dumps(record) + "\n")
    if arguments.debug_count > 0:
        debug_records = choose_debug_records(records, arguments.debug_count, np.random.default_rng(0))
        render_debug_panels_for_records(debug_records, {scene.scene_id: scene.image_path for scene in scenes}, output_directory / "debug")
    print(timing_line)
    print(table)
    print(f"results: {output_directory}")


if __name__ == "__main__":
    main()
