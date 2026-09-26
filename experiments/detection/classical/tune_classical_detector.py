"""Coordinate-descent sweep of the classical detector's thresholds on a val subset.

    python -m experiments.detection.classical.tune_classical_detector \\
        --split val --offset 120 --limit 150 --workers 2 --output-dir <dir>

For each parameter in `PARAMETER_GRID: dict[str, list] = {
    "maximum_interior_seam_strength": [0.55, 0.85, 1.01],
    "minimum_verification_score": [0.5, 0.65, 0.8],
    "minimum_line_hypothesis_score": [0.6, 0.9],
    "minimum_edge_support": [0.3, 0.6],
    "minimum_interior_print_fraction": [0.03, 0.07],
    "maximum_interior_texture": [3.5, 6.0],
    "maximum_covered_fraction": [0.45, 0.75],
    "relative_area_floor": [0.2, 0.4],
    "refinement_search_fraction": [0.003, 0.008],
    "working_long_side_pixels": [1400, 2000],
}


def objective_from_metrics(metrics: dict) -> float:
    """Mean of F1@0.5 and F1@0.9."""
    return 0.5 * (metrics["detection"]["0.50"]["f1"] + metrics["detection"]["0.90"]["f1"])


def evaluate_overrides(config_overrides: dict, scenes: list, worker_count: int) -> dict:
    """Run and score one config; returns a flat result record."""
    config = build_config_from_overrides(config_overrides)
    predictions_by_scene_id, seconds_per_image = run_detector_on_scenes(scenes, config, worker_count)
    metrics = score_predictions_against_split(predictions_by_scene_id, scenes, include_records=False)
    detection = metrics["detection"]
    return {
        "overrides": config_overrides,
        "objective": objective_from_metrics(metrics),
        "recall_50": detection["0.50"]["recall"],
        "precision_50": detection["0.50"]["precision"],
        "f1_50": detection["0.50"]["f1"],
        "f1_90": detection["0.90"]["f1"],
        "seconds_per_image": seconds_per_image,
    }


def main() -> None:
    """Sweep, log every result, save the best overrides."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Coordinate sweep of classical detector thresholds.")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--offset", type=int, default=0, help="skip the first N scenes (sorted by id)")
    parser.add_argument("--limit", type=int, default=150)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--start-config-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    known_fields = {field.name for field in dataclasses.fields(ClassicalDetectorConfig)}

    scenes = load_split_scene_annotations(arguments.split)[arguments.offset : arguments.offset + arguments.limit]
    best_overrides = json.loads(arguments.start_config_json.read_text()) if arguments.start_config_json else {}
    results_path = arguments.output_dir / "sweep_results.jsonl"
    best_result = evaluate_overrides(best_overrides, scenes, arguments.workers)
    with open(results_path, "a") as results_file:
        results_file.write(json.dumps(best_result) + "\n")
    logger.info("start objective %.4f", best_result["objective"])

    for _ in range(arguments.passes):
        for parameter_name, candidate_values in PARAMETER_GRID.items():
            if parameter_name not in known_fields:
                raise KeyError(f"unknown config field {parameter_name}")
            for candidate_value in candidate_values:
                if best_overrides.get(parameter_name, getattr(ClassicalDetectorConfig(), parameter_name)) == candidate_value:
                    continue
                trial_overrides = best_overrides | {parameter_name: candidate_value}
                trial_result = evaluate_overrides(trial_overrides, scenes, arguments.workers)
                with open(results_path, "a") as results_file:
                    results_file.write(json.dumps(trial_result) + "\n")
                logger.info("%s=%s objective %.4f (best %.4f)", parameter_name, candidate_value,
                            trial_result["objective"], best_result["objective"])
                if trial_result["objective"] > best_result["objective"]:
                    best_overrides, best_result = trial_overrides, trial_result
    (arguments.output_dir / "best_config.json").write_text(json.dumps(best_overrides, indent=2))
    logger.info("best objective %.4f with %s", best_result["objective"], best_overrides)


if __name__ == "__main__":
    main()
