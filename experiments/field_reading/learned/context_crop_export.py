"""Cut wide-margin "context crops" of the train field rows for recognizer training.

The standard field crop only has a 15% margin, so augmentation could only shrink the box. To mimic
an imperfect localizer (box edges off by +-10% of the box height, pulling in neighbouring print,
rules and other fields), training samples are cut from a wider window of the rectified check and
the jittered box is re-cropped with the standard margin rule at load time.

Output under TRAIN_CROPS_ROOT / "learned_context":
    crops/<row_key>.jpg                    box + CONTEXT_MARGIN_FRACTION * box height per side
    context_crops__split=train.jsonl       row_key, field_name, text, handwritten, status,
                                           context_crop_path, box_in_context, scene/font ids

Run: python -m experiments.field_reading.learned.context_crop_export --workers 3
"""

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd

from experiments.field_reading.config import TARGET_FIELD_NAMES, TRAIN_CROPS_ROOT
from experiments.field_reading.data_access.field_manifest import load_field_rows

logger = logging.getLogger(__name__)

CONTEXT_ROOT = TRAIN_CROPS_ROOT / "learned_context"
CONTEXT_MANIFEST_PATH = CONTEXT_ROOT / "context_crops__split=train.jsonl"
RECOGNIZER_TRAIN_FIELD_NAMES = TARGET_FIELD_NAMES + ["bank_name"]
SCORED_STATUSES = ("ok", "too_small")
CONTEXT_MARGIN_FRACTION = 0.6
CONTEXT_MIN_MARGIN_PX = 16
CONTEXT_JPEG_QUALITY = 95


def export_check_context_crops(check_crop_path: str, field_records: list[dict]) -> list[dict]:
    """Cut every field of one check; returns manifest records (skips crops already on disk)."""
    check_image = None
    manifest_records = []
    for record in field_records:
        output_path = CONTEXT_ROOT / "crops" / f"{record['row_key']}.jpg"
        x0, y0, x1, y1 = record["box_in_check_crop"]
        margin = max(CONTEXT_MIN_MARGIN_PX, CONTEXT_MARGIN_FRACTION * (y1 - y0))
        width, height = record["check_crop_size"]
        window = [int(max(0, np.floor(x0 - margin))), int(max(0, np.floor(y0 - margin))),
                  int(min(width, np.ceil(x1 + margin))), int(min(height, np.ceil(y1 + margin)))]
        if not output_path.exists():
            if check_image is None:
                check_image = cv2.imread(check_crop_path, cv2.IMREAD_COLOR)
            crop = check_image[window[1]:window[3], window[0]:window[2]]
            cv2.imwrite(str(output_path), crop, [cv2.IMWRITE_JPEG_QUALITY, CONTEXT_JPEG_QUALITY])
        manifest_records.append({
            "row_key": record["row_key"], "scene_id": record["scene_id"], "check_index": record["check_index"],
            "field_name": record["field_name"], "text": record["text"], "handwritten": record["handwritten"],
            "status": record["status"], "handwriting_font_id": record["handwriting_font_id"],
            "context_crop_path": str(output_path),
            "box_in_context": [x0 - window[0], y0 - window[1], x1 - window[0], y1 - window[1]],
        })
    return manifest_records


def main() -> None:
    """Export context crops for every scored train row of the recognizer fields."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=3)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rows = load_field_rows("train", target_fields_only=False)
    rows = rows[rows.field_name.isin(RECOGNIZER_TRAIN_FIELD_NAMES) & rows.status.isin(SCORED_STATUSES)]
    (CONTEXT_ROOT / "crops").mkdir(parents=True, exist_ok=True)
    columns = ["row_key", "scene_id", "check_index", "field_name", "text", "handwritten", "status",
               "handwriting_font_id", "box_in_check_crop", "check_crop_size"]
    jobs = [(path, group[columns].to_dict("records")) for path, group in rows.groupby("check_crop_path")]
    logger.info("exporting %d rows from %d checks", len(rows), len(jobs))
    manifest_records: list[dict] = []
    with ProcessPoolExecutor(max_workers=arguments.workers) as executor:
        for job_index, records in enumerate(executor.map(export_check_context_crops, *zip(*jobs), chunksize=64)):
            manifest_records.extend(records)
            if job_index % 2000 == 0:
                logger.info("%d / %d checks", job_index, len(jobs))
    pd.DataFrame.from_records(manifest_records).to_json(CONTEXT_MANIFEST_PATH, orient="records", lines=True)
    logger.info("wrote %d rows to %s", len(manifest_records), CONTEXT_MANIFEST_PATH)


if __name__ == "__main__":
    main()
