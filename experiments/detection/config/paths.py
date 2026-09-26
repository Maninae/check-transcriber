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
SYNTHETIC_DATASET_V1_1_CLOSEUP_EVAL_ROOT = CHECK_TRANSCRIBER_DATA_ROOT / "synth" / "v1.1-closeup-eval"
# Which dataset every loader reads. Set CHECK_DETECTION_DATASET_ROOT to score another set
# with the same layout (e.g. the v1.1 close-up eval set, eval split only).
ACTIVE_SYNTHETIC_DATASET_ROOT = Path(
    os.environ.get("CHECK_DETECTION_DATASET_ROOT", str(SYNTHETIC_DATASET_V1_ROOT))
)
REAL_SAMPLE_PHOTOS_ROOT = CHECK_TRANSCRIBER_DATA_ROOT / "samples"
DETECTION_EXPERIMENTS_ROOT = Path(
    os.environ.get(
        "CHECK_DETECTION_EXPERIMENTS_ROOT",
        str(CHECK_TRANSCRIBER_DATA_ROOT / "experiments" / "detection"),
    )
)

SPLIT_NAMES = ("train", "val", "eval")


V1_DATASET_NAME = "v1"


def synthetic_dataset_root(dataset_name: str | None = None) -> Path:
    """Root of a named synthetic dataset under synth/ (e.g. "v1.1-closeup-train"); None = active."""
    if dataset_name is None:
        return ACTIVE_SYNTHETIC_DATASET_ROOT
    return CHECK_TRANSCRIBER_DATA_ROOT / "synth" / dataset_name


def split_images_directory(split_name: str, dataset_name: str | None = None) -> Path:
    """Full-resolution scene JPEGs for one split of a dataset (default: the active one)."""
    return synthetic_dataset_root(dataset_name) / split_name / "images"


def split_annotations_directory(split_name: str, dataset_name: str | None = None) -> Path:
    """Per-scene JSON annotations (corners, outline, orientation, effects) for one split."""
    return synthetic_dataset_root(dataset_name) / split_name / "annotations"
