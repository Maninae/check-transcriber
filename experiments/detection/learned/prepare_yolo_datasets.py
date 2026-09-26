"""Build downscaled Ultralytics datasets (OBB and 4-corner pose) from the v1 synthetic set.

Why a copy: the v1 scenes are 3-6 MP JPEGs; decoding them every epoch dominates
training time on this machine. Resizing once to `RESIZED_LONG_SIDE_PIXELS` (above the
1024 training size) keeps the dataloader cheap. Labels are normalized, so they are
independent of the resize.

Two label sets are written from the same images:
- OBB labels: class + 4 normalized points = the minimum-area rectangle around the
  check's corner quad clipped to the frame (Ultralytics turns any 4 points into a
  rotated box anyway, so we give it the in-frame part of the true corner quad).
- pose labels: class, axis-aligned box of the in-frame quad, then the 4 corners as
  keypoints in the check's own TL, TR, BR, BL order. Corners outside the frame get
  visibility 0 (ignored by the loss); in-frame corners, even under an overlapping
  check, get visibility 2.

Layout under the output root: `<variant>/images/<split>/` and `<variant>/labels/<split>/`
for variant in (obb, pose), plus `data_obb.yaml` and `data_pose.yaml`. Each variant
holds its own copy of the images: Ultralytics resolves symlinks and then finds labels
by replacing `images` with `labels` in the real path, so shared image folders break.
"""

import argparse
import logging
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon, box

from experiments.detection.config.paths import DETECTION_EXPERIMENTS_ROOT, SPLIT_NAMES
from experiments.detection.dataset.scene_annotations import (
    SceneAnnotation,
    load_split_scene_annotations,
)

logger = logging.getLogger(__name__)

RESIZED_LONG_SIDE_PIXELS = 1280
RESIZED_JPEG_QUALITY = 93
DEFAULT_OUTPUT_ROOT = DETECTION_EXPERIMENTS_ROOT / "yolo_data" / f"long{RESIZED_LONG_SIDE_PIXELS}"
MIN_IN_FRAME_AREA_PIXELS = 50.0
KEYPOINT_VISIBLE = 2
KEYPOINT_OUTSIDE_FRAME = 0
LABEL_VARIANTS = ("obb", "pose")


def clip_quad_to_frame(corners: np.ndarray, image_width: int, image_height: int) -> np.ndarray | None:
    """Intersect the corner quad with the image rectangle; None if nothing remains."""
    clipped_polygon = Polygon(corners).buffer(0).intersection(box(0, 0, image_width, image_height))
    if clipped_polygon.is_empty or clipped_polygon.area < MIN_IN_FRAME_AREA_PIXELS:
        return None
    if clipped_polygon.geom_type != "Polygon":
        clipped_polygon = max(clipped_polygon.geoms, key=lambda part: part.area)
    return np.asarray(clipped_polygon.exterior.coords[:-1], dtype=np.float64)


def format_obb_label_line(clipped_points: np.ndarray, image_width: int, image_height: int) -> str:
    """One OBB label row: the min-area rectangle of the in-frame quad, normalized."""
    rectangle_points = cv2.boxPoints(cv2.minAreaRect(clipped_points.astype(np.float32)))
    normalized_points = rectangle_points / np.array([image_width, image_height])
    normalized_points = np.clip(normalized_points, 0.0, 1.0)
    return "0 " + " ".join(f"{value:.6f}" for value in normalized_points.reshape(-1))


def format_pose_label_line(
    corners: np.ndarray, clipped_points: np.ndarray, image_width: int, image_height: int
) -> str:
    """One pose label row: axis-aligned in-frame box then 4 ordered corner keypoints."""
    x_min, y_min = clipped_points.min(axis=0)
    x_max, y_max = clipped_points.max(axis=0)
    box_values = [
        (x_min + x_max) / 2 / image_width,
        (y_min + y_max) / 2 / image_height,
        (x_max - x_min) / image_width,
        (y_max - y_min) / image_height,
    ]
    keypoint_values: list[float] = []
    for corner_x, corner_y in corners:
        corner_in_frame = 0 <= corner_x <= image_width and 0 <= corner_y <= image_height
        if corner_in_frame:
            keypoint_values += [corner_x / image_width, corner_y / image_height, KEYPOINT_VISIBLE]
        else:
            keypoint_values += [0.0, 0.0, KEYPOINT_OUTSIDE_FRAME]
    return "0 " + " ".join(f"{value:.6f}" for value in box_values + keypoint_values)


def write_scene_to_yolo_datasets(scene: SceneAnnotation, output_root: Path) -> int:
    """Resize one scene image and write its OBB and pose label files; returns checks written."""
    image_output_path = output_root / "obb" / "images" / scene.split_name / f"{scene.scene_id}.jpg"
    pose_image_output_path = output_root / "pose" / "images" / scene.split_name / image_output_path.name
    if not image_output_path.exists():
        image_bgr = cv2.imread(str(scene.image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"cannot read {scene.image_path}")
        scale = RESIZED_LONG_SIDE_PIXELS / max(image_bgr.shape[:2])
        resized_bgr = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(image_output_path), resized_bgr, [cv2.IMWRITE_JPEG_QUALITY, RESIZED_JPEG_QUALITY])
    if not pose_image_output_path.exists():
        shutil.copyfile(image_output_path, pose_image_output_path)
    obb_lines, pose_lines = [], []
    for check in scene.checks:
        clipped_points = clip_quad_to_frame(check.corners, scene.image_width, scene.image_height)
        if clipped_points is None:
            continue
        obb_lines.append(format_obb_label_line(clipped_points, scene.image_width, scene.image_height))
        pose_lines.append(
            format_pose_label_line(check.corners, clipped_points, scene.image_width, scene.image_height)
        )
    for variant, lines in (("obb", obb_lines), ("pose", pose_lines)):
        label_path = output_root / variant / "labels" / scene.split_name / f"{scene.scene_id}.txt"
        label_path.write_text("\n".join(lines) + "\n")
    return len(obb_lines)


def write_variant_yaml(output_root: Path, variant: str) -> Path:
    """Write the Ultralytics data yaml for one variant tree."""
    variant_root = output_root / variant
    yaml_lines = [
        f"path: {variant_root}",
        "train: images/train",
        "val: images/val",
        "test: images/eval",
        "names:",
        "  0: check",
    ]
    if variant == "pose":
        # Mirroring a check is never realistic, so fliplr is disabled in training; flip_idx
        # is still required by Ultralytics and maps TL<->TR, BR<->BL.
        yaml_lines += ["kpt_shape: [4, 3]", "flip_idx: [1, 0, 3, 2]"]
    yaml_path = output_root / f"data_{variant}.yaml"
    yaml_path.write_text("\n".join(yaml_lines) + "\n")
    return yaml_path


def main() -> None:
    """Build both datasets for all splits."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args()
    output_root: Path = arguments.output_root
    for split_name in SPLIT_NAMES:
        for variant in LABEL_VARIANTS:
            for subdirectory in ("images", "labels"):
                (output_root / variant / subdirectory / split_name).mkdir(parents=True, exist_ok=True)
        scenes = load_split_scene_annotations(split_name)
        with ProcessPoolExecutor(max_workers=arguments.workers) as executor:
            checks_written = sum(
                executor.map(write_scene_to_yolo_datasets, scenes, [output_root] * len(scenes), chunksize=16)
            )
        logger.info("%s: %d scenes, %d checks", split_name, len(scenes), checks_written)
    for variant in LABEL_VARIANTS:
        logger.info("wrote %s", write_variant_yaml(output_root, variant))


if __name__ == "__main__":
    main()
