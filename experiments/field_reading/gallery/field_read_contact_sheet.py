"""Contact sheets of field crops with ground truth and every method's read, for Owen to eyeball.

Each tile: the field crop (scaled to a fixed height) on the left; on the right one line for the
ground truth and one per method with its text, confidence, and a green/red mark from the
metrics harness's correctness rule. The crop is the payload, so it gets the most pixels;
method names are small and gray.

Two sheets are produced:
- `showcase`: mostly hard handwritten rows (stratified by field), a couple of printed ones.
- `failures`: rows where the best method is confidently wrong (confidence above its median),
  the failure mode the app's blank-with-crop fallback cannot catch.

Run: python -m experiments.field_reading.gallery.field_read_contact_sheet --split eval --methods A__loc=oracle B__loc=oracle
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from experiments.field_reading.config import PREDICTIONS_ROOT, REPORTS_ROOT
from experiments.field_reading.data_access.field_manifest import load_field_rows
from experiments.field_reading.metrics.field_value_parsing import normalize_field_value

logger = logging.getLogger(__name__)

SHEET_WIDTH_PX = 2000
CROP_HEIGHT_PX = 110
CROP_MAX_WIDTH_PX = 900
TEXT_COLUMN_X_PX = CROP_MAX_WIDTH_PX + 40
LINE_HEIGHT_PX = 30
TILE_GAP_PX = 22
SAMPLE_SEED = 11
FONT_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"
MONO_FONT_PATH = "/System/Library/Fonts/Menlo.ttc"
CORRECT_COLOR = (20, 130, 60)
WRONG_COLOR = (190, 40, 30)
LABEL_COLOR = (130, 130, 130)
MAX_SHOWN_READ_CHARACTERS = 46   # runaway hallucinations are cut with an ellipsis so they stay on the sheet
METHOD_LABEL_WIDTH_PX = 190   # short method name column left of each read, so long reads never collide
TEXT_COLOR = (20, 20, 20)
JPEG_QUALITY = 82


def is_read_correct(field_name: str, predicted_text: str, truth_text: str) -> bool:
    """The metrics harness's correctness rule (blank is never correct)."""
    predicted_value = normalize_field_value(field_name, predicted_text or "")
    return predicted_value is not None and predicted_value == normalize_field_value(field_name, truth_text)


def load_method_predictions(split_name: str, method_labels: list[str]) -> pd.DataFrame:
    """Wide table: row_key -> <label>__text, <label>__confidence for each method label."""
    tables = []
    for label in method_labels:
        predictions = pd.read_json(PREDICTIONS_ROOT / split_name / f"{label}.jsonl", lines=True).set_index("row_key")
        tables.append(predictions[["pred_text", "confidence"]].rename(
            columns={"pred_text": f"{label}__text", "confidence": f"{label}__confidence"}))
    return pd.concat(tables, axis=1, join="inner")


def short_method_name(method_label: str) -> str:
    """Readable short name for a prediction-file stem (localization suffix dropped)."""
    stem = method_label.split("__loc=")[0]
    for prefix, short in (("tesseract", "Tesseract"), ("trocr_small_handwritten", "TrOCR-hw"), ("trocr_small_printed", "TrOCR-printed"),
                          ("crnn_amount", "CRNN-amount"), ("crnn_general", "CRNN")):
        if stem.startswith(prefix):
            return short + ("+xcheck" if "+xcheck" in stem else "")
    return stem[:22]


def draw_tile(row: pd.Series, method_labels: list[str], fonts: dict) -> Image.Image:
    """One tile: crop left, ground truth and each method's read right."""
    line_count = 1 + len(method_labels)
    tile_height = max(CROP_HEIGHT_PX, (line_count + 1) * LINE_HEIGHT_PX)
    tile = Image.new("RGB", (SHEET_WIDTH_PX, tile_height), "white")
    crop = Image.open(row.field_crop_path).convert("RGB")
    scale = min(CROP_HEIGHT_PX / crop.height, CROP_MAX_WIDTH_PX / crop.width)
    tile.paste(crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))), Image.LANCZOS), (0, 0))
    draw = ImageDraw.Draw(tile)
    header = row.get("tile_header") or f"{row.field_name}  {'handwritten' if row.handwritten else 'printed'}  {row.text_height_in_photo_px:.0f}px  {row.layout_family}"
    draw.text((TEXT_COLUMN_X_PX, 0), header, fill=LABEL_COLOR, font=fonts["label"])
    draw.text((TEXT_COLUMN_X_PX, LINE_HEIGHT_PX + 4), "truth", fill=LABEL_COLOR, font=fonts["label"])
    draw.text((TEXT_COLUMN_X_PX + METHOD_LABEL_WIDTH_PX, LINE_HEIGHT_PX), f"      {row.text}", fill=TEXT_COLOR, font=fonts["mono"])
    for index, label in enumerate(method_labels, start=2):
        text, confidence = row[f"{label}__text"] or "", row[f"{label}__confidence"]
        precomputed_flag = row.get(f"{label}__correct")
        correct = bool(precomputed_flag) if precomputed_flag is not None and pd.notna(precomputed_flag) else is_read_correct(row.field_name, text, row.text)
        color = CORRECT_COLOR if correct else WRONG_COLOR
        shown_text = text if text else "(blank)"
        if len(shown_text) > MAX_SHOWN_READ_CHARACTERS:
            shown_text = shown_text[:MAX_SHOWN_READ_CHARACTERS - 1] + "…"
        confidence_text = f"{confidence:.2f}" if pd.notna(confidence) else "  - "
        draw.text((TEXT_COLUMN_X_PX, index * LINE_HEIGHT_PX + 4), short_method_name(label), fill=LABEL_COLOR, font=fonts["label"])
        draw.text((TEXT_COLUMN_X_PX + METHOD_LABEL_WIDTH_PX, index * LINE_HEIGHT_PX), f"{confidence_text}  {shown_text}", fill=color, font=fonts["mono"])
    return tile


def render_sheet(rows: pd.DataFrame, method_labels: list[str], output_path: Path) -> None:
    """Stack tiles vertically with whitespace between them and save as JPEG."""
    fonts = {"mono": ImageFont.truetype(MONO_FONT_PATH, 22), "label": ImageFont.truetype(FONT_PATH, 17)}
    tiles = [draw_tile(row, method_labels, fonts) for _, row in rows.iterrows()]
    sheet = Image.new("RGB", (SHEET_WIDTH_PX, sum(t.height + TILE_GAP_PX for t in tiles)), "white")
    y_offset = 0
    for tile in tiles:
        sheet.paste(tile, (0, y_offset))
        y_offset += tile.height + TILE_GAP_PX
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=JPEG_QUALITY, optimize=True)
    logger.info("wrote %s (%.2f MB, %d tiles)", output_path, output_path.stat().st_size / 1e6, len(tiles))


def main() -> None:
    """Build the showcase and failure sheets for a split."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", default="eval")
    parser.add_argument("--methods", nargs="+", required=True, help="prediction file stems, e.g. tesseract_tuned__loc=oracle")
    parser.add_argument("--best-method", default=None, help="method label used to pick confident failures (default: last)")
    parser.add_argument("--showcase-count", type=int, default=10)
    parser.add_argument("--failure-count", type=int, default=10)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    truth = load_field_rows(arguments.split)
    truth = truth[truth.status == "ok"].set_index("row_key")
    joined = truth.join(load_method_predictions(arguments.split, arguments.methods), how="inner")
    handwritten = joined[joined.handwritten & (joined.text_height_in_photo_px < 24)]
    showcase_parts = [group.sample(min(2, len(group)), random_state=SAMPLE_SEED)
                      for _, group in handwritten.groupby("field_name")]
    showcase = pd.concat(showcase_parts).head(arguments.showcase_count - 2)
    printed = joined[~joined.handwritten].sample(2, random_state=SAMPLE_SEED)
    output_directory = REPORTS_ROOT / "gallery"
    render_sheet(pd.concat([showcase, printed]), arguments.methods, output_directory / f"field_reads__showcase__split={arguments.split}.jpg")
    best = arguments.best_method or arguments.methods[-1]
    best_correct = [is_read_correct(f, t or "", g) for f, t, g in zip(joined.field_name, joined[f"{best}__text"], joined.text)]
    confident = joined[f"{best}__confidence"] >= joined[f"{best}__confidence"].median()
    failures = joined[confident & ~pd.Series(best_correct, index=joined.index)]
    failure_sample = failures.sample(min(arguments.failure_count, len(failures)), random_state=SAMPLE_SEED)
    render_sheet(failure_sample, arguments.methods, output_directory / f"field_reads__confident_failures__method={best.split('__loc=')[0]}__split={arguments.split}.jpg")


if __name__ == "__main__":
    main()
