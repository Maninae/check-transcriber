"""Score Tesseract (default and tuned configs) on the 78 hand-labelled real SSBI handwriting crops.

Reuses the learned area's SSBI loader, padding and comparison rules so every method is scored
identically; see `learned/real_ssbi_scoring.py` for the rules and the licence caveat.

Run: python -m experiments.field_reading.ocr_baselines.tesseract_real_ssbi
"""

import logging

import pandas as pd
from rapidfuzz.distance import Levenshtein

from experiments.field_reading.config import REPORTS_ROOT
from experiments.field_reading.learned.line_crop_dataset import read_rgb_image
from experiments.field_reading.learned.real_ssbi_scoring import SSBI_ROOT, comparable_text, load_ssbi_rows, pad_tight_crop
from experiments.field_reading.ocr_baselines.tesseract_predict import read_configs_for_method
from experiments.field_reading.ocr_baselines.tesseract_reader import read_field_crop_with_tesseract

logger = logging.getLogger(__name__)

TESSERACT_METHODS = ("tesseract_default", "tesseract_tuned")


def main() -> None:
    """Read every labelled SSBI crop with each Tesseract method and write a per-field table."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rows = load_ssbi_rows()
    crops = [pad_tight_crop(read_rgb_image(str(SSBI_ROOT / "crops" / name))) for name in rows.crop]
    records = []
    for method in TESSERACT_METHODS:
        configs = read_configs_for_method(method)
        for (_, row), crop in zip(rows.iterrows(), crops):
            text, confidence = read_field_crop_with_tesseract(crop, row.field_name, configs[row.field_name])
            truth, predicted = comparable_text(row.field_name, row.text), comparable_text(row.field_name, text)
            records.append({"method": method, "field_name": row.field_name, "truth": row.text, "pred_text": text,
                            "confidence": confidence, "exact": truth == predicted,
                            "cer": Levenshtein.distance(truth, predicted) / max(1, len(truth))})
    table = pd.DataFrame(records).groupby(["method", "field_name"]).agg(n=("exact", "size"), exact=("exact", "mean"), cer=("cer", "mean")).round(3)
    output_path = REPORTS_ROOT / "real_ssbi" / "tesseract__real_ssbi.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("# Tesseract on real SSBI handwriting (n=78)\n\n" + table.to_markdown() + "\n")
    logger.info("\n%s", table)


if __name__ == "__main__":
    main()
