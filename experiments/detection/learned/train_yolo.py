"""Train an Ultralytics YOLO model (OBB or 4-corner pose) on the downscaled v1 dataset.

Licensing: Ultralytics code and its pretrained weights are AGPL-3.0. That is fine for
training and evaluation here; shipping a model trained this way inside the public app
has AGPL implications (see experiments/detection/README.md).

Run from the worktree root, e.g.
    python -m experiments.detection.learned.train_yolo --variant obb --model yolo26n-obb.pt
Outputs land in `<experiments root>/runs/<variant>/<run name>/` on vega.

- `fliplr` is 0 for pose because a mirrored check is never real and would teach the
  model a mirrored corner order; OBB boxes are order-free so flips stay on there.
- Rotation augmentation is modest: the data already covers 0/90/180/270 with jitter.
"""

import argparse
import logging
import os
from pathlib import Path

from ultralytics import YOLO

from experiments.detection.config.paths import DETECTION_EXPERIMENTS_ROOT
from experiments.detection.learned.prepare_yolo_datasets import DEFAULT_OUTPUT_ROOT

logger = logging.getLogger(__name__)

PRETRAINED_WEIGHTS_DIRECTORY = DETECTION_EXPERIMENTS_ROOT / "pretrained"
DEFAULT_IMAGE_SIZE = 1024
DEFAULT_EPOCHS = 8
DEFAULT_BATCH_SIZE = 8
DEFAULT_DATALOADER_WORKERS = 4
EARLY_STOP_PATIENCE_EPOCHS = 15
CLOSE_MOSAIC_FINAL_EPOCHS = 2  # last epochs train on un-mosaicked, full scenes


def build_training_arguments(arguments: argparse.Namespace) -> dict:
    """Assemble the keyword arguments for `YOLO.train`."""
    is_pose_variant = arguments.variant == "pose"
    return {
        "data": arguments.data or str(DEFAULT_OUTPUT_ROOT / f"data_{arguments.variant}.yaml"),
        "imgsz": arguments.image_size,
        "epochs": arguments.epochs,
        "batch": arguments.batch_size,
        "workers": arguments.workers,
        "device": "mps",
        "project": str(DETECTION_EXPERIMENTS_ROOT / "runs" / arguments.variant),
        "name": arguments.run_name,
        "exist_ok": False,
        "patience": EARLY_STOP_PATIENCE_EPOCHS,
        "cache": False,
        "degrees": 10.0,
        "fliplr": 0.0 if is_pose_variant else 0.5,
        "mosaic": 1.0,
        "close_mosaic": CLOSE_MOSAIC_FINAL_EPOCHS,
        "plots": True,
        "seed": 42,
        "deterministic": False,
        "amp": False,
    }


def main() -> None:
    """Parse arguments and run one training job."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", choices=["obb", "pose"], required=True)
    parser.add_argument("--model", required=True, help="pretrained weights name, e.g. yolo26n-obb.pt")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=DEFAULT_DATALOADER_WORKERS)
    parser.add_argument("--data", default=None, help="data yaml; default is the v1 yaml for the variant")
    arguments = parser.parse_args()
    PRETRAINED_WEIGHTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    os.chdir(PRETRAINED_WEIGHTS_DIRECTORY)  # Ultralytics downloads pretrained weights into cwd
    model = YOLO(arguments.model)
    training_arguments = build_training_arguments(arguments)
    logger.info("training %s with %s", arguments.model, training_arguments)
    model.train(**training_arguments)


if __name__ == "__main__":
    main()
