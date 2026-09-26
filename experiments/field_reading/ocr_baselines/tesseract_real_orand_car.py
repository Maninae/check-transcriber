"""Score Tesseract on real handwritten courtesy amounts (ORAND-CAR-2014 test sets, CC BY-NC-ND, eval only).

Metric: exact match on digits only (all non-digits stripped from the read; the ground truth is a
digit string, and the checks use '.' as the thousands separator), plus CER on the digit strings.
Sample: 500 per subset (CAR-A Uruguay, CAR-B Chile), seed 0, the same sample the learned readers use.

Run: python -m experiments.field_reading.ocr_baselines.tesseract_real_orand_car
"""

import logging
import random
import re

import cv2
import pandas as pd
from rapidfuzz.distance import Levenshtein

from experiments.field_reading.config import REAL_SAMPLES_ROOT, REPORTS_ROOT
from experiments.field_reading.ocr_baselines.tesseract_predict import read_configs_for_method
from experiments.field_reading.ocr_baselines.tesseract_reader import read_field_crop_with_tesseract

logger = logging.getLogger(__name__)

ORAND_ROOT = REAL_SAMPLES_ROOT / "orand-car-2014" / "ORAND-CAR-2014"
SUBSETS = {"CAR-A": ("a_test_gt.txt", "a_test_images"), "CAR-B": ("b_test_gt.txt", "b_test_images")}
SAMPLE_PER_SUBSET = 500
SAMPLE_SEED = 0


def load_orand_sample() -> list[dict]:
    """Fixed random sample of test crops per subset: {subset, path, digits}."""
    rows = []
    for subset, (gt_name, image_directory) in SUBSETS.items():
        lines = [line.split() for line in (ORAND_ROOT / subset / gt_name).read_text().splitlines() if line.strip()]
        for file_name, digits in random.Random(SAMPLE_SEED).sample(lines, SAMPLE_PER_SUBSET):
            rows.append({"subset": subset, "path": str(ORAND_ROOT / subset / image_directory / file_name), "digits": digits})
    return rows


def main() -> None:
    """Read the sample with default and tuned Tesseract and write the per-subset table."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rows = load_orand_sample()
    records = []
    for method in ("tesseract_default", "tesseract_tuned"):
        config = read_configs_for_method(method)["amount_numeric"]
        for row in rows:
            crop = cv2.cvtColor(cv2.imread(row["path"]), cv2.COLOR_BGR2RGB)
            text, _ = read_field_crop_with_tesseract(crop, "amount_numeric", config)
            predicted_digits = re.sub(r"\D", "", text)
            records.append({"method": method, "subset": row["subset"], "exact": predicted_digits == row["digits"],
                            "cer": Levenshtein.distance(predicted_digits, row["digits"]) / len(row["digits"])})
    table = pd.DataFrame(records).groupby(["method", "subset"]).agg(n=("exact", "size"), exact=("exact", "mean"), cer=("cer", "mean")).round(3)
    output_path = REPORTS_ROOT / "real_orand_car" / "tesseract__orand_car.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("# Tesseract on real courtesy amounts (ORAND-CAR-2014 test, digits-only exact)\n\n" + table.to_markdown() + "\n")
    logger.info("\n%s", table)


if __name__ == "__main__":
    main()
