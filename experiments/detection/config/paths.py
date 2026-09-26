"""Filesystem locations for the detection experiments (dataset in, runs out).

Everything heavy lives on the external `vega` drive: the v1 synthetic dataset is
read from there and every model, run directory and rendered image is written there.
Override either root with an environment variable when running elsewhere.
"""

import os
from pathlib import Path

CHECK_TRANSCRIBER_DATA_ROOT = Path(
    os.environ.get("CHECK_TRANSCRIBER_DATA_ROOT", "/Volumes/vega/datasets/check-transcriber")
)
SYNTHETIC_DATASET_V1_ROOT = CHECK_TRANSCRIBER_DATA_ROOT / "synth" / "v1"
REAL_SAMPLE_PHOTOS_ROOT = CHECK_TRANSCRIBER_DATA_ROOT / "samples"
DETECTION_EXPERIMENTS_ROOT = Path(
    os.environ.get(
        "CHECK_DETECTION_EXPERIMENTS_ROOT",
        str(CHECK_TRANSCRIBER_DATA_ROOT / "experiments" / "detection"),
    )
)

SPLIT_NAMES = ("train", "val", "eval")


def split_images_directory(split_name: str) -> Path:
    """Full-resolution scene JPEGs for one split of the v1 dataset."""
    return SYNTHETIC_DATASET_V1_ROOT / split_name / "images"


def split_annotations_directory(split_name: str) -> Path:
    """Per-scene JSON annotations (corners, outline, orientation, effects) for one split."""
    return SYNTHETIC_DATASET_V1_ROOT / split_name / "annotations"
