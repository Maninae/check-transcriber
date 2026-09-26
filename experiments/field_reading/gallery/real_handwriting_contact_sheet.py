"""Contact sheet of REAL handwritten crops (SSBI + ORAND-CAR) with each reader's output.

Reuses `field_read_contact_sheet.render_sheet`; correctness comes from the real-set scorers'
per-row `exact` flags (SSBI literal strings, CAR digits-only), not the synth value parsers.
Both sources are non-commercial licences: the sheet stays on vega and in private messages.

Run: python -m experiments.field_reading.gallery.real_handwriting_contact_sheet
"""

import json
import logging

import pandas as pd

from experiments.field_reading.config import FIELD_READING_OUTPUT_ROOT, REAL_SAMPLES_ROOT, REPORTS_ROOT
from experiments.field_reading.gallery.field_read_contact_sheet import render_sheet

logger = logging.getLogger(__name__)

SSBI_PER_ROW_CSV = REPORTS_ROOT / "learned" / "real_ssbi__methods=trocr_small_handwritten_zs_grey+crnn_general+crnn_amount_route.csv"
CAR_PER_ROW_CSV = REPORTS_ROOT / "learned" / "real_car__per_row.csv"
METHODS = ["trocr_small_handwritten_zs_grey", "crnn_general", "crnn_amount_route"]
SSBI_TILES_PER_FIELD = 2
CAR_TILES = 4
SAMPLE_SEED = 3


def wide_rows(per_row: pd.DataFrame, key_column: str) -> pd.DataFrame:
    """One row per crop with <method>__text / __confidence / __correct columns."""
    per_row = per_row[per_row.method.isin(METHODS)]
    wide = per_row.pivot_table(index=key_column, columns="method", values=["pred", "confidence", "exact"], aggfunc="first")
    output = pd.DataFrame(index=wide.index)
    for method in METHODS:
        if ("pred", method) not in wide.columns:
            continue
        output[f"{method}__text"] = wide[("pred", method)].fillna("").astype(str)
        output[f"{method}__confidence"] = wide[("confidence", method)]
        output[f"{method}__correct"] = wide[("exact", method)]
    return output


def ssbi_tiles() -> pd.DataFrame:
    """A couple of SSBI crops per field, joined to their crop paths and truths."""
    per_row = pd.read_csv(SSBI_PER_ROW_CSV)
    index = pd.DataFrame(json.loads((FIELD_READING_OUTPUT_ROOT / "real_ssbi" / "crops_index.json").read_text()))
    index["row_key"] = "ssbi_" + index.idx.astype(str)
    wide = wide_rows(per_row, "row_key").join(index.set_index("row_key"))
    wide["field_name"] = per_row.drop_duplicates("row_key").set_index("row_key").field.reindex(wide.index)
    wide["field_crop_path"] = str(FIELD_READING_OUTPUT_ROOT / "real_ssbi" / "crops") + "/" + wide.crop
    wide["tile_header"] = "REAL (SSBI)  " + wide.field_name + "  handwritten"
    return wide.groupby("field_name", group_keys=False).apply(lambda g: g.sample(min(SSBI_TILES_PER_FIELD, len(g)), random_state=SAMPLE_SEED))


def car_tiles() -> pd.DataFrame:
    """A few ORAND-CAR courtesy amounts."""
    per_row = pd.read_csv(CAR_PER_ROW_CSV, dtype={"truth": str, "pred": str})
    wide = wide_rows(per_row, "row_key")
    meta = per_row.drop_duplicates("row_key").set_index("row_key")
    wide["text"] = meta.truth.reindex(wide.index)
    wide["field_name"] = "amount_numeric"
    wide["field_crop_path"] = [str(REAL_SAMPLES_ROOT / "orand-car-2014" / "ORAND-CAR-2014" / key.split("_", 1)[0]
                                   / f"{key.split('_', 1)[1][0]}_test_images" / key.split("_", 1)[1]) for key in wide.index]
    wide["tile_header"] = "REAL (ORAND-CAR, digits only)  " + meta.subset.reindex(wide.index)
    return wide.sample(CAR_TILES, random_state=SAMPLE_SEED)


def main() -> None:
    """Render the real-handwriting sheet."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    tiles = pd.concat([ssbi_tiles(), car_tiles()])
    render_sheet(tiles, METHODS, REPORTS_ROOT / "gallery" / "field_reads__real_handwriting__ssbi+orand_car.jpg")


if __name__ == "__main__":
    main()
