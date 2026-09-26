"""YOLO exports (Ultralytics layout): segmentation from the paper outline, OBB from the 4 corners.

- Segmentation: `<split>/labels/<scene>.txt`, class 0, the outline (contract C3) clipped to
  the photo, normalized to [0, 1]. `data.yaml` points Ultralytics at `<split>/images`.
- OBB: `<split>/labels_obb/<scene>.txt`, class 0, 4 corners in the check's own TL TR BR BL
  order, normalized. A fully in-frame check writes its true corners. A partly out-of-frame one
  writes the rectangle, aligned with the check's own top edge, that encloses its visible part
  (`visible_check_rectangle`): the rule Ultralytics itself applies when augmentation crops an
  OBB (`Instances.clip(preserve_obb=True)`), and never the trapezoid per-corner clamping makes.
  That rectangle can overhang the frame a little; Ultralytics accepts a label whose axis-aligned
  box (centre, width, height) stays within [-0.01, 1.01], so only a label that would fail that
  check falls back to clamping.
- Ultralytics finds labels by swapping `images` for `labels` in the image path, so `labels_obb`
  is reached through `yolo_obb/<split>/images` (per-image symlinks) beside `yolo_obb/<split>/labels`
  (a directory symlink to `<split>/labels_obb`), and `data_obb.yaml`.
- The yamls carry no `path:` key: Ultralytics then takes the yaml's own directory as the dataset
  root (a relative `path:` would resolve against the cwd), so a dataset can be moved or copied.
- Points are shifted from scene-label pixel centres to pixel edges (`to_pixel_edge_coordinates`).
- COCO lives in `coco_export.py`.
"""

import logging
import os
from pathlib import Path

import numpy as np

from synth.compose.perspective import clip_polygon_to_rect, polygon_area
from synth.dataset.coco_export import MIN_CLIPPED_AREA_PX, check_outline, to_pixel_edge_coordinates
from synth.dataset.splits import SPLIT_NAMES

logger = logging.getLogger(__name__)

YOLO_OBB_DIRECTORY_NAME = "yolo_obb"
YOLO_SPLIT_KEYS = {"train": "train", "val": "val", "eval": "test"}
ULTRALYTICS_LABEL_TOLERANCE = 0.01   # verify_image_label accepts normalized box values in [-0.01, 1.01]


def clipped_check_polygon(points: list[list[float]], width: int, height: int) -> np.ndarray:
    """A scene-label check polygon (outline or corners) in pixel-edge coordinates, clipped to the photo."""
    return clip_polygon_to_rect(to_pixel_edge_coordinates(points), width, height)


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


def visible_check_rectangle(corners: np.ndarray, width: int, height: int) -> np.ndarray:
    """TL TR BR BL of the rectangle, aligned with the check's top edge, enclosing its in-frame part.

    `corners` are pixel-edge coordinates in the check's own order. The axis is the mean of the
    top and bottom edge directions, so the rectangle keeps the check's rotation and TL-first order.
    """
    along_top = (corners[1] - corners[0]) + (corners[2] - corners[3])
    along_top /= np.linalg.norm(along_top)
    down_side = np.array([-along_top[1], along_top[0]])
    if np.dot(down_side, corners[3] - corners[0]) < 0:
        down_side = -down_side
    visible = clip_polygon_to_rect(corners, width, height)
    along, down = visible @ along_top, visible @ down_side
    rectangle_axes = [(along.min(), down.min()), (along.max(), down.min()), (along.max(), down.max()), (along.min(), down.max())]
    return np.array([a * along_top + d * down_side for a, d in rectangle_axes])


def passes_ultralytics_label_check(normalized_points: np.ndarray) -> bool:
    """Ultralytics' loader check on a polygon label: its axis-aligned centre and size lie in [-0.01, 1.01]."""
    low, high = normalized_points.min(axis=0), normalized_points.max(axis=0)
    box = np.concatenate([(low + high) / 2, high - low])
    return bool(box.min() >= -ULTRALYTICS_LABEL_TOLERANCE and box.max() <= 1 + ULTRALYTICS_LABEL_TOLERANCE)


def obb_corners_normalized(check: dict, width: int, height: int) -> np.ndarray:
    """The OBB for one visible check, normalized; see the module docstring for the out-of-frame rule."""
    corners = to_pixel_edge_coordinates(check["corners"])
    in_frame = ((corners >= 0) & (corners <= [width, height])).all()
    rectangle = corners if in_frame else visible_check_rectangle(corners, width, height)
    normalized = rectangle / [width, height]
    if not passes_ultralytics_label_check(normalized):
        logger.info("OBB overhangs past Ultralytics' tolerance; clamping check %s", check["check_index"])
        normalized = normalized.clip(0, 1)
    return normalized


def yolo_obb_lines(scene_label: dict) -> list[str]:
    """One Ultralytics OBB line (4 corners) per visible check."""
    width, height = scene_label["image_width"], scene_label["image_height"]
    return ["0 " + " ".join(f"{value:.6f}" for value in obb_corners_normalized(check, width, height).ravel())
            for check in scene_label["checks"] if check_is_visible(check, width, height)]


def write_yolo_labels(scene_label: dict, split_directory: Path) -> None:
    """Write both YOLO label files for one scene."""
    stem = Path(scene_label["image_file"]).stem
    (split_directory / "labels" / f"{stem}.txt").write_text("\n".join(yolo_segmentation_lines(scene_label)) + "\n")
    (split_directory / "labels_obb" / f"{stem}.txt").write_text("\n".join(yolo_obb_lines(scene_label)) + "\n")


def yolo_data_yaml_text(images_parent: str = "") -> str:
    """An Ultralytics dataset config with our eval split as its `test` set; paths relative to the yaml's directory."""
    split_lines = "".join(f"{YOLO_SPLIT_KEYS[name]}: {images_parent}{name}/images\n" for name in SPLIT_NAMES)
    return f"# no `path:` key: Ultralytics resolves these against this file's directory\n{split_lines}names:\n  0: check\n"


def write_yolo_obb_tree(output_directory: Path) -> Path:
    """`yolo_obb/<split>/images/<scene>.jpg -> ../../../<split>/images/<scene>.jpg`, `labels -> ../../<split>/labels_obb`.

    Images are linked one file at a time inside a real directory: Ultralytics `.resolve()`s each
    split directory from the yaml, which would follow a directory symlink back to `<split>/images`
    and read the segmentation labels beside it instead of the OBB ones.
    """
    obb_root = output_directory / YOLO_OBB_DIRECTORY_NAME
    for split_name in SPLIT_NAMES:
        images_directory = obb_root / split_name / "images"
        images_directory.mkdir(parents=True, exist_ok=True)
        for image_path in sorted((output_directory / split_name / "images").glob("*.jpg")):
            link_path = images_directory / image_path.name
            if not link_path.is_symlink():
                os.symlink(Path("..") / ".." / ".." / split_name / "images" / image_path.name, link_path)
        labels_link = obb_root / split_name / "labels"
        if not labels_link.is_symlink():
            os.symlink(Path("..") / ".." / split_name / "labels_obb", labels_link)
    return obb_root


def write_yolo_data_yaml(output_directory: Path) -> tuple[Path, Path]:
    """Write `data.yaml` (segmentation) and `data_obb.yaml` (oriented boxes); returns both paths."""
    segmentation_yaml = output_directory / "data.yaml"
    segmentation_yaml.write_text(yolo_data_yaml_text())
    obb_root = write_yolo_obb_tree(output_directory)
    obb_yaml = output_directory / "data_obb.yaml"
    obb_yaml.write_text(yolo_data_yaml_text(f"{obb_root.relative_to(output_directory).as_posix()}/"))
    return segmentation_yaml, obb_yaml
