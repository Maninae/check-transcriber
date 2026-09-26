"""Draw GT and every method's boxes on 6 eval crops (one per layout family) into one JPEG under 3 MB.

Green = ground truth; each method gets one colour (legend strip on top). One eval check per family,
chosen deterministically with the most handwritten fields so the hard case is always shown.

Run: python -m experiments.field_reading.field_localization.overlay_visual
"""

import logging

import cv2
import numpy as np

from experiments.field_reading.field_localization.localization_config import (LOCALIZATION_REPORTS_ROOT,
                                                                              localization_prediction_path)
from experiments.field_reading.field_localization.localization_targets import load_localization_checks
from experiments.field_reading.field_localization.prediction_io import read_prediction_rows_by_key
from experiments.field_reading.data_access.field_manifest import make_row_key

logger = logging.getLogger(__name__)

GROUND_TRUTH_COLOUR_BGR = (40, 170, 40)
METHOD_COLOURS_BGR = [(200, 90, 20), (0, 140, 255), (200, 0, 200)]
TILE_WIDTH_PX = 1000
LEGEND_HEIGHT_PX = 44
JPEG_QUALITY = 82
OUTPUT_PATH = LOCALIZATION_REPORTS_ROOT / "localization_overlay__split=eval.jpg"


def pick_one_check_per_family(check_records: list[dict]) -> list[dict]:
    """Per family, the first check (sorted order) with the most handwritten scored fields."""
    picked = {}
    for check_record in check_records:
        handwritten_count = sum(target["handwritten"] and target["status"] == "ok"
                                for target in check_record["fields"].values())
        family = check_record["layout_family"]
        if family not in picked or handwritten_count > picked[family][0]:
            picked[family] = (handwritten_count, check_record)
    return [picked[family][1] for family in sorted(picked)]


def draw_box(image: np.ndarray, box: list[float], colour: tuple[int, int, int], thickness: int, inset: int) -> None:
    """Rectangle inset by a few px per method so coincident boxes stay distinguishable."""
    cv2.rectangle(image, (int(box[0]) - inset, int(box[1]) - inset), (int(box[2]) + inset, int(box[3]) + inset),
                  colour, thickness)


def render_overlay(method_ids: list[str]) -> None:
    """Build and write the overlay JPEG."""
    check_records = pick_one_check_per_family(load_localization_checks("eval"))
    predictions_by_method = {method: read_prediction_rows_by_key(localization_prediction_path("eval", method))
                             for method in method_ids}
    tiles = []
    for check_record in check_records:
        image = cv2.imread(check_record["check_crop_path"])
        for field_name, target in check_record["fields"].items():
            if target["box"] is not None:
                draw_box(image, target["box"], GROUND_TRUTH_COLOUR_BGR, 6, 0)
            for method_index, method_id in enumerate(method_ids):
                row = predictions_by_method[method_id].get(
                    make_row_key(check_record["scene_id"], check_record["check_index"], field_name), {})
                if row.get("pred_box") is not None:
                    draw_box(image, row["pred_box"], METHOD_COLOURS_BGR[method_index], 3, 5 + 5 * method_index)
        tile = cv2.resize(image, (TILE_WIDTH_PX, int(image.shape[0] * TILE_WIDTH_PX / image.shape[1])),
                          interpolation=cv2.INTER_AREA)
        cv2.putText(tile, check_record["layout_family"], (8, tile.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 0), 2)
        tiles.append(cv2.copyMakeBorder(tile, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
    tile_height = max(tile.shape[0] for tile in tiles)
    tiles = [cv2.copyMakeBorder(tile, 0, tile_height - tile.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
             for tile in tiles]
    grid = np.vstack([np.hstack(tiles[row_start:row_start + 2]) for row_start in range(0, len(tiles), 2)])
    legend = np.full((LEGEND_HEIGHT_PX, grid.shape[1], 3), 255, np.uint8)
    legend_x = 10
    for label, colour in [("ground truth", GROUND_TRUTH_COLOUR_BGR), *zip(method_ids, METHOD_COLOURS_BGR)]:
        cv2.rectangle(legend, (legend_x, 12), (legend_x + 30, 32), colour, 4)
        cv2.putText(legend, label, (legend_x + 40, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
        legend_x += 60 + cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0][0]
    LOCALIZATION_REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_PATH), np.vstack([legend, grid]), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    logger.info("wrote %s (%.2f MB)", OUTPUT_PATH, OUTPUT_PATH.stat().st_size / 1e6)


def main() -> None:
    """CLI entry point: overlay every method with eval predictions (classical first)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    method_ids = sorted(path.stem for path in localization_prediction_path("eval", "x").parent.glob("*.jsonl"))
    render_overlay(method_ids)


if __name__ == "__main__":
    main()
