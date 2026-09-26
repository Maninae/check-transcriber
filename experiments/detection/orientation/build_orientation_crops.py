"""Build the upright/upside-down crop dataset from GT corners on the resized YOLO images.

For every check with at least `MIN_IN_FRAME_FRACTION` in frame, the GT corners are
jittered (to mimic detector error), warped to a landscape grayscale crop, and stored
either as-is (label 0, upright) or rotated 180 degrees via the corner order (label 1,
upside down), chosen at random. Output per split: `<split>__crops.npy` (N, H, W) uint8
and `<split>__labels.npy` (N,) uint8 under the orientation experiments directory.
"""

import argparse
import logging
import zlib
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from experiments.detection.config.paths import DETECTION_EXPERIMENTS_ROOT, SPLIT_NAMES
from experiments.detection.dataset.scene_annotations import SceneAnnotation, load_split_scene_annotations
from experiments.detection.learned.prepare_yolo_datasets import DEFAULT_OUTPUT_ROOT
from experiments.detection.orientation.rectify_check_crop import warp_quadrilateral_to_crop

logger = logging.getLogger(__name__)

ORIENTATION_EXPERIMENTS_DIRECTORY = DETECTION_EXPERIMENTS_ROOT / "orientation"
MIN_IN_FRAME_FRACTION = 0.8
CORNER_JITTER_FRACTION_OF_SHORT_SIDE = 0.03


def build_scene_crops(scene: SceneAnnotation) -> tuple[list[np.ndarray], list[int]]:
    """Crops and labels for every usable check of one scene (deterministic per scene)."""
    random_generator = np.random.default_rng(zlib.crc32(scene.scene_id.encode()))
    resized_image_path = DEFAULT_OUTPUT_ROOT / "obb" / "images" / scene.split_name / f"{scene.scene_id}.jpg"
    image_gray = cv2.imread(str(resized_image_path), cv2.IMREAD_GRAYSCALE)
    scale = image_gray.shape[1] / scene.image_width
    crops, labels = [], []
    for check in scene.checks:
        if check.in_frame_fraction < MIN_IN_FRAME_FRACTION:
            continue
        corners = check.corners * scale
        short_side = min(np.linalg.norm(corners[3] - corners[0]), np.linalg.norm(corners[1] - corners[0]))
        jitter = random_generator.uniform(-1, 1, size=(4, 2)) * CORNER_JITTER_FRACTION_OF_SHORT_SIDE * short_side
        is_upside_down = int(random_generator.random() < 0.5)
        crop_corners = np.roll(corners + jitter, -2 * is_upside_down, axis=0)
        crops.append(warp_quadrilateral_to_crop(image_gray, crop_corners))
        labels.append(is_upside_down)
    return crops, labels


def main() -> None:
    """Write crops for every split."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=3)
    arguments = parser.parse_args()
    ORIENTATION_EXPERIMENTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for split_name in SPLIT_NAMES:
        scenes = load_split_scene_annotations(split_name)
        all_crops, all_labels = [], []
        with ProcessPoolExecutor(max_workers=arguments.workers) as executor:
            for crops, labels in executor.map(build_scene_crops, scenes, chunksize=16):
                all_crops += crops
                all_labels += labels
        np.save(ORIENTATION_EXPERIMENTS_DIRECTORY / f"{split_name}__crops.npy", np.stack(all_crops))
        np.save(ORIENTATION_EXPERIMENTS_DIRECTORY / f"{split_name}__labels.npy", np.array(all_labels, np.uint8))
        logger.info("%s: %d crops, %.3f upside down", split_name, len(all_labels), np.mean(all_labels))


if __name__ == "__main__":
    main()
