"""Visual QA for the OCR manifest: field crops stacked with their ground-truth text underneath.

Usage: python -m dataset_builder.qa.ocr_crop_grid DATASET_DIR --out /tmp/ocr_grid.png [--split val] [--count 24] [--status ok]

Each tile is one crop scaled to a common width, then a caption: `field [hw|pr] status` and the
label text. Reading the grid answers "does the crop show exactly this text?" at a glance.
"""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from dataset_builder.ocr.ocr_manifest import OCR_DIRECTORY_NAME
from synthetic_checks.fonts.font_registry import load_font

TILE_WIDTH_PX = 760
CAPTION_HEIGHT_PX = 70
CAPTION_FONT_ID = "pt_sans"
CAPTION_FONT_PX = 22
COLUMNS = 2
GUTTER_PX = 14
MAX_CROP_HEIGHT_PX = 260


def crop_tile(dataset_directory: Path, row: dict) -> Image.Image:
    """One crop, scaled to the tile width (capped in height), with a two-line caption."""
    crop = Image.open(dataset_directory / row["field_crop"]).convert("RGB")
    scale = min(TILE_WIDTH_PX / crop.width, MAX_CROP_HEIGHT_PX / crop.height)
    crop = crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (TILE_WIDTH_PX, crop.height + CAPTION_HEIGHT_PX), "white")
    tile.paste(crop, (0, 0))
    draw = ImageDraw.Draw(tile)
    font = load_font(CAPTION_FONT_ID, CAPTION_FONT_PX)
    kind = "hw" if row["handwritten"] else "pr"
    header = f"{row['scene_id']} c{row['check_index']}  {row['field_name']} [{kind}] {row['status']}"
    draw.text((4, crop.height + 6), header, fill=(110, 110, 110), font=font)
    draw.text((4, crop.height + 36), repr(row["text"].replace("\n", " / ")), fill=(0, 0, 0), font=font)
    return tile


def build_ocr_crop_grid(dataset_directory: Path, output_path: Path, split_name: str, count: int, status: str | None) -> Path:
    """Tile the first `count` rows (optionally of one status) that have a field crop."""
    manifest_path = dataset_directory / OCR_DIRECTORY_NAME / f"ocr_fields__split={split_name}.jsonl"
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line]
    rows = [row for row in rows if row["field_crop"] and (status is None or row["status"] == status)][:count]
    tiles = [crop_tile(dataset_directory, row) for row in rows]
    column_heights = [0] * COLUMNS
    placements = []
    for tile in tiles:
        column = column_heights.index(min(column_heights))
        placements.append((tile, column * (TILE_WIDTH_PX + GUTTER_PX), column_heights[column]))
        column_heights[column] += tile.height + GUTTER_PX
    sheet = Image.new("RGB", (COLUMNS * (TILE_WIDTH_PX + GUTTER_PX), max(column_heights)), (235, 235, 235))
    for tile, x, y in placements:
        sheet.paste(tile, (x, y))
    sheet.save(output_path)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_directory", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--status", default=None, help="only rows with this status (e.g. ok, occluded)")
    arguments = parser.parse_args()
    print(build_ocr_crop_grid(arguments.dataset_directory, arguments.out, arguments.split, arguments.count, arguments.status))
