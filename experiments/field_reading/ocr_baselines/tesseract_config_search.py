"""Grid-search Tesseract read configs per field on a stratified val sample; pick the best per field.

Selection score: the metrics harness's own correctness rule (`normalize_field_value` equality:
money as cents, dates as ISO, digits, normalized text), tie-broken by CER on the loose key
(casefold, keep only [0-9a-z/]). Final numbers always come from the shared metrics harness.

Output: `field-reading/reports/tesseract_config_search__split=val.json` (every config x field)
and the chosen per-field configs `tesseract_best_configs.json` next to it.

Run: python -m experiments.field_reading.ocr_baselines.tesseract_config_search --rows-per-field 240 --workers 4
"""

import argparse
import itertools
import json
import logging
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict

import cv2
import pandas as pd
from rapidfuzz.distance import Levenshtein

from experiments.field_reading.config import REPORTS_ROOT, TARGET_FIELD_NAMES
from experiments.field_reading.data_access.field_manifest import load_field_rows
from experiments.field_reading.metrics.field_value_parsing import normalize_field_value
from experiments.field_reading.ocr_baselines.field_crop_preprocessing import TesseractPreprocessingConfig
from experiments.field_reading.ocr_baselines.tesseract_reader import (FIELD_CHARACTER_WHITELISTS, TesseractReadConfig,
                                                                        read_field_crop_with_tesseract)

logger = logging.getLogger(__name__)

PAGE_SEGMENTATION_MODES = (7, 13, 6)
TARGET_CROP_HEIGHTS = (None, 40, 64)
BINARIZATIONS = ("none", "otsu", "adaptive")
SAMPLE_SEED = 7
SEARCH_RESULTS_PATH = REPORTS_ROOT / "tesseract_config_search__split=val.json"
BEST_CONFIGS_PATH = REPORTS_ROOT / "tesseract_best_configs.json"


def loose_match_key(text: str) -> str:
    """Casefolded text reduced to digits, letters and '/', for config selection only."""
    return re.sub(r"[^0-9a-z/]", "", text.casefold())


def all_read_configs(field_name: str) -> list[TesseractReadConfig]:
    """The search grid for one field (whitelist variants only where a whitelist exists)."""
    whitelist_options = (False, True) if field_name in FIELD_CHARACTER_WHITELISTS else (False,)
    return [TesseractReadConfig(psm, whitelist, TesseractPreprocessingConfig(height, binarization, rules))
            for psm, height, binarization, rules, whitelist in itertools.product(
                PAGE_SEGMENTATION_MODES, TARGET_CROP_HEIGHTS, BINARIZATIONS, (False, True), whitelist_options)]


def stratified_sample(rows: pd.DataFrame, rows_per_field: int) -> pd.DataFrame:
    """Up to `rows_per_field` usable rows per field, half handwritten where the field has handwriting."""
    samples = []
    for field_name, field_rows in rows[rows.usable].groupby("field_name"):
        for handwritten, group in field_rows.groupby("handwritten"):
            share = rows_per_field // field_rows.handwritten.nunique()
            samples.append(group.sample(min(share, len(group)), random_state=SAMPLE_SEED))
    return pd.concat(samples, ignore_index=True)


def score_config_on_rows(task: tuple[dict, str, list[tuple[str, str]]]) -> dict:
    """Worker: read every (crop path, gt) with one config; return harness accuracy and loose-key CER."""
    config_dict, field_name, crop_paths_and_truths = task
    preprocessing = TesseractPreprocessingConfig(**config_dict.pop("preprocessing"))
    config = TesseractReadConfig(preprocessing=preprocessing, **config_dict)
    correct_count, character_error_total = 0, 0.0
    for crop_path, truth in crop_paths_and_truths:
        crop_rgb = cv2.cvtColor(cv2.imread(crop_path), cv2.COLOR_BGR2RGB)
        predicted_text, _ = read_field_crop_with_tesseract(crop_rgb, field_name, config)
        truth_key, predicted_key = loose_match_key(truth), loose_match_key(predicted_text)
        predicted_value = normalize_field_value(field_name, predicted_text)
        correct_count += predicted_value is not None and predicted_value == normalize_field_value(field_name, truth)
        character_error_total += Levenshtein.distance(truth_key, predicted_key) / max(1, len(truth_key))
    count = len(crop_paths_and_truths)
    return {"field_name": field_name, "config_id": config.config_id(), "config": asdict(config),
            "accuracy": correct_count / count, "mean_cer": character_error_total / count, "n": count}


def main() -> None:
    """Run the grid, write every result and the per-field winners."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows-per-field", type=int, default=120)
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sample = stratified_sample(load_field_rows("val"), arguments.rows_per_field)
    tasks = []
    for field_name in TARGET_FIELD_NAMES:
        field_sample = sample[sample.field_name == field_name]
        pairs = list(zip(field_sample.field_crop_path, field_sample.text))
        tasks.extend((asdict(config), field_name, pairs) for config in all_read_configs(field_name))
    logger.info("%d (config, field) tasks over %d sampled rows", len(tasks), len(sample))
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        results = list(pool.map(score_config_on_rows, tasks, chunksize=2))
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    SEARCH_RESULTS_PATH.write_text(json.dumps(results, indent=1))
    best_by_field = {}
    for field_name in TARGET_FIELD_NAMES:
        field_results = sorted((r for r in results if r["field_name"] == field_name), key=lambda r: (-r["accuracy"], r["mean_cer"]))
        best_by_field[field_name] = field_results[0]
        default = next(r for r in field_results if r["config_id"] == TesseractReadConfig().config_id())
        logger.info("%-15s best %.3f (cer %.3f) %s | default %.3f", field_name, field_results[0]["accuracy"],
                    field_results[0]["mean_cer"], field_results[0]["config_id"], default["accuracy"])
    BEST_CONFIGS_PATH.write_text(json.dumps(best_by_field, indent=1))


if __name__ == "__main__":
    main()
