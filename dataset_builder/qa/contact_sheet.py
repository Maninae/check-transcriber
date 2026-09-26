"""Visual QA: a grid of scenes with every check polygon drawn from its label.

Usage: python -m dataset_builder.qa.contact_sheet DATASET_DIR --out /tmp/sheet.png [--count 12] [--split train]

- Green: the paper outline (deformed edge). Small yellow dots: the 4 physical corners.
  Large green dot: the check's own top-left corner (orientation).
- Thin magenta: field quads, so misaligned field labels are visible at a glance.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

TILE_WIDTH = 900
COLUMNS = 4


def draw_scene_labels(image: np.ndarray, scene_label: dict, scale: float) -> np.ndarray:
    """Overlay polygons, top-left dots and field quads on a downscaled scene (BGR)."""
    for check in scene_label["checks"]:
        corners = np.round(np.array(check["corners"]) * scale).astype(np.int32)
        outline = np.round(np.array(check.get("outline") or check["corners"]) * scale).astype(np.int32)
        for field in check["fields"]:
            cv2.polylines(image, [np.round(np.array(field["quad"]) * scale).astype(np.int32)], True, (200, 0, 200), 1)
        cv2.polylines(image, [outline], True, (0, 220, 0), 2)
        for corner in corners:
            cv2.circle(image, tuple(int(v) for v in corner), 4, (0, 230, 255), -1)
        cv2.circle(image, tuple(int(v) for v in corners[0]), 8, (0, 220, 0), -1)
    return image


def build_contact_sheet(dataset_directory: Path, output_path: Path, count: int, split_name: str) -> Path:
    """Tile the first `count` scenes of a split with labels drawn."""
    label_paths = sorted((dataset_directory / split_name / "annotations").glob("*.json"))[:count]
    tiles = []
    for label_path in label_paths:
        scene_label = json.loads(label_path.read_text())
        image = cv2.imread(str(dataset_directory / split_name / "images" / scene_label["image_file"]))
        scale = TILE_WIDTH / max(image.shape[:2])
        image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        tile = np.full((TILE_WIDTH, TILE_WIDTH, 3), 255, np.uint8)
        tile[:image.shape[0], :image.shape[1]] = draw_scene_labels(image, scene_label, scale)
        cv2.putText(tile, f"{scene_label['scene_id']}  n={len(scene_label['checks'])}  {scene_label['layout_mode']}",
                    (10, TILE_WIDTH - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2)
        tiles.append(tile)
    while len(tiles) % COLUMNS:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.hstack(tiles[i:i + COLUMNS]) for i in range(0, len(tiles), COLUMNS)]
    cv2.imwrite(str(output_path), np.vstack(rows))
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_directory", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--split", default="train")
    arguments = parser.parse_args()
    print(build_contact_sheet(arguments.dataset_directory, arguments.out, arguments.count, arguments.split))
