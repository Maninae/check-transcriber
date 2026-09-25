"""YOLO exports (Ultralytics layout): segmentation from the paper outline, OBB from the 4 corners.

- Segmentation: `<split>/labels/<scene>.txt`, class 0, the outline (contract C3) clipped to
  the photo, normalized to [0, 1]. `data.yaml` points Ultralytics at `<split>/images`.
- OBB: `<split>/labels_obb/<scene>.txt`, class 0, the 4 corners TL TR BR BL normalized and
  clamped to [0, 1]. Ultralytics finds labels by swapping `images` for `labels` in the image
  path, so `labels_obb` is reached through `yolo_obb/<split>/{images,labels}`, two directory
  symlinks per split, and `data_obb.yaml`.
- COCO lives in `coco_export.py`.
"""

import os
from pathlib import Path

import numpy as np

from synth.compose.perspective import clip_polygon_to_rect, polygon_area
from synth.dataset.coco_export import MIN_CLIPPED_AREA_PX, check_outline
from synth.dataset.splits import SPLIT_NAMES

YOLO_OBB_DIRECTORY_NAME = "yolo_obb"
YOLO_SPLIT_KEYS = {"train": "train", "val": "val", "eval": "test"}


def clipped_check_polygon(points: list[list[float]], width: int, height: int) -> np.ndarray:
    """A check polygon (outline or corners) clipped to the photo."""
    return clip_polygon_to_rect(np.asarray(points, np.float64), width, height)


def check_is_visible(check: dict, width: int, height: int) -> bool:
    """A check gets a YOLO line when a meaningful part of its outline is in frame."""
    polygon = clipped_check_polygon(check_outline(check), width, height)
    return len(polygon) >= 3 and polygon_area(polygon) >= MIN_CLIPPED_AREA_PX


def yolo_segmentation_lines(scene_label: dict) -> list[str]:
    """One Ultralytics segmentation line per visible check."""
    width, height = scene_label["image_width"], scene_label["image_height"]
    lines = []
    for check in scene_label["checks"]:
        if not check_is_visible(check, width, height):
            continue
        normalized = (clipped_check_polygon(check_outline(check), width, height) / [width, height]).clip(0, 1)
        lines.append("0 " + " ".join(f"{value:.6f}" for value in normalized.ravel()))
    return lines


def yolo_obb_lines(scene_label: dict) -> list[str]:
    """One Ultralytics OBB line (4 corners) per visible check."""
    width, height = scene_label["image_width"], scene_label["image_height"]
    lines = []
    for check in scene_label["checks"]:
        if not check_is_visible(check, width, height):
            continue
        normalized = (np.asarray(check["corners"]) / [width, height]).clip(0, 1)
        lines.append("0 " + " ".join(f"{value:.6f}" for value in normalized.ravel()))
    return lines


def write_yolo_labels(scene_label: dict, split_directory: Path) -> None:
    """Write both YOLO label files for one scene."""
    stem = Path(scene_label["image_file"]).stem
    (split_directory / "labels" / f"{stem}.txt").write_text("\n".join(yolo_segmentation_lines(scene_label)) + "\n")
    (split_directory / "labels_obb" / f"{stem}.txt").write_text("\n".join(yolo_obb_lines(scene_label)) + "\n")


def yolo_data_yaml_text(dataset_path: Path) -> str:
    """An Ultralytics dataset config with our eval split as its `test` set."""
    split_lines = "".join(f"{YOLO_SPLIT_KEYS[name]}: {name}/images\n" for name in SPLIT_NAMES)
    return f"path: {dataset_path}\n{split_lines}names:\n  0: check\n"


def write_yolo_obb_tree(output_directory: Path) -> Path:
    """`yolo_obb/<split>/images -> ../../<split>/images` and `labels -> ../../<split>/labels_obb`."""
    obb_root = output_directory / YOLO_OBB_DIRECTORY_NAME
    for split_name in SPLIT_NAMES:
        split_root = obb_root / split_name
        split_root.mkdir(parents=True, exist_ok=True)
        for link_name, target_name in (("images", "images"), ("labels", "labels_obb")):
            link_path = split_root / link_name
            if not link_path.is_symlink():
                os.symlink(Path("..") / ".." / split_name / target_name, link_path)
    return obb_root


def write_yolo_data_yaml(output_directory: Path) -> tuple[Path, Path]:
    """Write `data.yaml` (segmentation) and `data_obb.yaml` (oriented boxes); returns both paths."""
    segmentation_yaml = output_directory / "data.yaml"
    segmentation_yaml.write_text(yolo_data_yaml_text(output_directory))
    obb_yaml = output_directory / "data_obb.yaml"
    obb_yaml.write_text(yolo_data_yaml_text(write_yolo_obb_tree(output_directory)))
    return segmentation_yaml, obb_yaml
