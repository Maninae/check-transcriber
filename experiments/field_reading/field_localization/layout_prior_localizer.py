"""Run the layout-prior + ink-refinement localizers (a1 family oracle, a2 size kind) and write predictions.

Priors are estimated on train once and cached at LAYOUT_PRIOR_JSON_PATH. CPU only, 3 worker processes.

Run: python -m experiments.field_reading.field_localization.layout_prior_localizer --splits val eval
"""

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import cv2

from experiments.field_reading.config import TARGET_FIELD_NAMES
from experiments.field_reading.field_localization.ink_refinement import (InkRefinementParams, compute_clean_ink_mask,
                                                                         refine_prior_to_ink_box)
from experiments.field_reading.field_localization.layout_prior import (estimate_layout_priors, load_layout_priors,
                                                                       save_layout_priors)
from experiments.field_reading.field_localization.localization_config import (LAYOUT_PRIOR_GROUP_COLUMN_BY_METHOD,
                                                                              LAYOUT_PRIOR_JSON_PATH,
                                                                              localization_prediction_path)
from experiments.field_reading.field_localization.localization_targets import load_localization_checks
from experiments.field_reading.field_localization.prediction_io import make_prediction_row, write_prediction_rows

logger = logging.getLogger(__name__)

WORKER_PROCESS_COUNT = 3


def load_or_estimate_layout_priors(force_reestimate: bool = False) -> dict[str, dict]:
    """{group_column: {group: {field: prior}}}, estimated from train on first use."""
    if LAYOUT_PRIOR_JSON_PATH.exists() and not force_reestimate:
        return load_layout_priors(LAYOUT_PRIOR_JSON_PATH)
    train_checks = load_localization_checks("train")
    priors_by_group_column = {column: estimate_layout_priors(train_checks, column)
                              for column in sorted(set(LAYOUT_PRIOR_GROUP_COLUMN_BY_METHOD.values()))}
    save_layout_priors(priors_by_group_column, LAYOUT_PRIOR_JSON_PATH)
    logger.info("wrote layout priors to %s", LAYOUT_PRIOR_JSON_PATH)
    return priors_by_group_column


def localize_check_with_priors(check_record: dict, priors_by_group_column: dict, method_ids: list[str],
                               params: InkRefinementParams) -> dict[str, list[dict]]:
    """method id -> prediction rows for every target field of one check."""
    grayscale_check = cv2.imread(check_record["check_crop_path"], cv2.IMREAD_GRAYSCALE)
    clean_ink_mask = compute_clean_ink_mask(grayscale_check, params)
    rows_by_method: dict[str, list[dict]] = {}
    for method_id in method_ids:
        group_column = LAYOUT_PRIOR_GROUP_COLUMN_BY_METHOD[method_id]
        group_priors = priors_by_group_column[group_column][check_record[group_column]]
        method_rows = []
        for field_name in TARGET_FIELD_NAMES:
            refined_box, confidence = (refine_prior_to_ink_box(clean_ink_mask, group_priors[field_name], params)
                                       if field_name in group_priors else (None, 0.0))
            method_rows.append(make_prediction_row(check_record["scene_id"], check_record["check_index"],
                                                   field_name, refined_box, confidence))
        rows_by_method[method_id] = method_rows
    return rows_by_method


def run_layout_prior_localizers(check_records: list[dict], method_ids: list[str],
                                params: InkRefinementParams) -> dict[str, list[dict]]:
    """Localize every check in parallel; method id -> all prediction rows (check order preserved)."""
    priors_by_group_column = load_or_estimate_layout_priors()
    worker = partial(localize_check_with_priors, priors_by_group_column=priors_by_group_column,
                     method_ids=method_ids, params=params)
    rows_by_method: dict[str, list[dict]] = {method_id: [] for method_id in method_ids}
    with ProcessPoolExecutor(WORKER_PROCESS_COUNT) as executor:
        for check_rows_by_method in executor.map(worker, check_records, chunksize=16):
            for method_id, method_rows in check_rows_by_method.items():
                rows_by_method[method_id].extend(method_rows)
    return rows_by_method


def main() -> None:
    """CLI entry point: write both classical methods' predictions for the requested splits."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--splits", nargs="+", default=["val", "eval"])
    parser.add_argument("--reestimate-priors", action="store_true")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_or_estimate_layout_priors(force_reestimate=arguments.reestimate_priors)
    method_ids = list(LAYOUT_PRIOR_GROUP_COLUMN_BY_METHOD)
    for split_name in arguments.splits:
        rows_by_method = run_layout_prior_localizers(load_localization_checks(split_name), method_ids,
                                                     InkRefinementParams())
        for method_id, method_rows in rows_by_method.items():
            output_path = localization_prediction_path(split_name, method_id)
            write_prediction_rows(method_rows, output_path)
            logger.info("wrote %d rows to %s", len(method_rows), output_path)


if __name__ == "__main__":
    main()
