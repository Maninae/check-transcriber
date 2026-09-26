"""Write Tesseract field predictions for a split in the shared prediction contract (see PLAN.md).

Methods:
- `tesseract_default`: psm 7, native scale, no preprocessing, no whitelist (the naive app baseline).
- `tesseract_tuned`: the per-field winner of `tesseract_config_search` on val.

Localization:
- `oracle`: synth's ground-truth field crop.
- any localizer id: boxes from `predictions/localization/<split>/<id>.jsonl`, cropped from the
  1600 px check with `crop_field_from_check`; a null box is a blank prediction.

Run: python -m experiments.field_reading.ocr_baselines.tesseract_predict --split eval --method tesseract_tuned --localization oracle
"""

import argparse
import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict

import cv2
import pandas as pd

from experiments.field_reading.config import PREDICTIONS_ROOT, TARGET_FIELD_NAMES
from experiments.field_reading.data_access.field_crop import crop_field_from_check
from experiments.field_reading.data_access.field_manifest import load_field_rows
from experiments.field_reading.ocr_baselines.field_crop_preprocessing import TesseractPreprocessingConfig
from experiments.field_reading.ocr_baselines.tesseract_config_search import BEST_CONFIGS_PATH
from experiments.field_reading.ocr_baselines.tesseract_reader import TesseractReadConfig, read_field_crop_with_tesseract

logger = logging.getLogger(__name__)

SCORED_STATUSES = ("ok", "too_small")
ORACLE_LOCALIZATION = "oracle"


def read_configs_for_method(method: str) -> dict[str, TesseractReadConfig]:
    """Per-field read config for a Tesseract method id."""
    if method == "tesseract_default":
        return {field_name: TesseractReadConfig() for field_name in TARGET_FIELD_NAMES}
    if method == "tesseract_tuned":
        best = json.loads(BEST_CONFIGS_PATH.read_text())
        configs = {}
        for field_name in TARGET_FIELD_NAMES:
            config_dict = dict(best[field_name]["config"])
            preprocessing = TesseractPreprocessingConfig(**config_dict.pop("preprocessing"))
            configs[field_name] = TesseractReadConfig(preprocessing=preprocessing, **config_dict)
        return configs
    raise ValueError(f"unknown tesseract method {method!r}")


def load_localized_boxes(split_name: str, localizer_id: str) -> dict[str, dict]:
    """row_key -> {pred_box, confidence} from a localizer's prediction file."""
    path = PREDICTIONS_ROOT / "localization" / split_name / f"{localizer_id}.jsonl"
    return {row["row_key"]: row for row in map(json.loads, path.read_text().splitlines())}


def predict_rows_for_check(task: tuple[str, str, list[dict], dict[str, dict]]) -> list[dict]:
    """Worker: read every scored field of one check (one image decode per check)."""
    method, localization, field_rows, config_dicts = task
    configs = {name: TesseractReadConfig(preprocessing=TesseractPreprocessingConfig(**c.pop("preprocessing")), **c)
               for name, c in ((name, dict(c)) for name, c in config_dicts.items())}
    check_rgb = None
    if localization != ORACLE_LOCALIZATION:
        check_rgb = cv2.cvtColor(cv2.imread(field_rows[0]["check_crop_path"]), cv2.COLOR_BGR2RGB)
    predictions = []
    for row in field_rows:
        started = time.perf_counter()
        if localization == ORACLE_LOCALIZATION:
            crop_rgb, box = cv2.cvtColor(cv2.imread(row["field_crop_path"]), cv2.COLOR_BGR2RGB), row["box_in_check_crop"]
        else:
            box = row["localized_box"]
            crop_rgb = crop_field_from_check(check_rgb, box) if box is not None else None
        text, confidence = ("", 0.0)
        if crop_rgb is not None and crop_rgb.size:
            text, confidence = read_field_crop_with_tesseract(crop_rgb, row["field_name"], configs[row["field_name"]])
        predictions.append({"row_key": row["row_key"], "scene_id": row["scene_id"], "check_index": row["check_index"],
                            "field_name": row["field_name"], "method": method, "localization": localization,
                            "pred_text": text, "confidence": round(confidence, 4), "pred_box": box,
                            "latency_ms": round(1000 * (time.perf_counter() - started), 2)})
    return predictions


def main() -> None:
    """Predict every scored row of a split and write the JSONL."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", required=True)
    parser.add_argument("--method", required=True, choices=["tesseract_default", "tesseract_tuned"])
    parser.add_argument("--localization", default=ORACLE_LOCALIZATION)
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rows = load_field_rows(arguments.split)
    rows = rows[rows.status.isin(SCORED_STATUSES)].copy()
    if arguments.localization != ORACLE_LOCALIZATION:
        boxes = load_localized_boxes(arguments.split, arguments.localization)
        rows["localized_box"] = rows.row_key.map(lambda key: boxes.get(key, {}).get("pred_box"))
    config_dicts = {name: asdict(config) for name, config in read_configs_for_method(arguments.method).items()}
    columns = [c for c in ("row_key", "scene_id", "check_index", "field_name", "field_crop_path", "check_crop_path",
                           "box_in_check_crop", "localized_box") if c in rows.columns]
    tasks = [(arguments.method, arguments.localization, group[columns].to_dict("records"), config_dicts)
             for _, group in rows.groupby(["scene_id", "check_index"], sort=True)]
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        predictions = [p for check_predictions in pool.map(predict_rows_for_check, tasks, chunksize=8) for p in check_predictions]
    output_path = PREDICTIONS_ROOT / arguments.split / f"{arguments.method}__loc={arguments.localization}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(p) + "\n" for p in predictions))
    logger.info("wrote %d predictions to %s", len(predictions), output_path)


if __name__ == "__main__":
    main()
