"""Configuration for the CenterNet check detector: fixed model geometry plus training knobs.

Geometry constants (input size, output stride, channel counts) are shared by the
model, the target encoder, the decoder and the ONNX export, so they live here as
one source of truth. `CenterNetTrainingConfig` is a plain dataclass saved as JSON
into every run directory; every field can be overridden from the training CLI.
"""

from dataclasses import dataclass

from experiments.detection.config.paths import DETECTION_EXPERIMENTS_ROOT

DEFAULT_INPUT_SIZE_PIXELS = 768
OUTPUT_STRIDE = 4
CORNER_COUNT = 4
CORNER_OFFSET_CHANNELS = 2 * CORNER_COUNT  # TL, TR, BR, BL as (dx, dy) from the anchor cell
LETTERBOX_FILL_VALUE = 114
# The offset head's raw output is multiplied by this, so targets of 25-110 cells (at 768)
# need raw values of ~1-7 instead of starting ~100x too small.
CORNER_OFFSET_OUTPUT_SCALE_CELLS = 16.0

# ImageNet statistics: the backbone's pretrained weights expect this normalization.
IMAGENET_MEAN_RGB = (0.485, 0.456, 0.406)
IMAGENET_STD_RGB = (0.229, 0.224, 0.225)

CENTERNET_RUNS_ROOT = DETECTION_EXPERIMENTS_ROOT / "runs" / "centernet"
PRETRAINED_WEIGHTS_DIRECTORY = DETECTION_EXPERIMENTS_ROOT / "pretrained" / "torchvision"
TRAINING_IMAGES_ROOT = (
    DETECTION_EXPERIMENTS_ROOT / "yolo_data" / "long1280" / "obb" / "images"
)  # 1280-long-side copies; GT is scaled by the image-width ratio


@dataclass
class CenterNetTrainingConfig:
    """Every knob of one training run; saved to `<run>/config.json`."""

    run_name: str = "mnv3l_768"
    input_size_pixels: int = DEFAULT_INPUT_SIZE_PIXELS
    backbone_name: str = "mobilenet_v3_large"
    neck_channels: int = 96
    epochs: int = 30
    batch_size: int = 8
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 300
    heatmap_loss_weight: float = 1.0
    corner_offset_loss_weight: float = 5.0
    dataloader_workers: int = 2
    device: str = "mps"
    seed: int = 42
    train_scene_limit: int | None = None  # None = the whole train split
    val_scene_limit: int = 200  # quick val metric subset per dataset, used for checkpoint selection
    extra_train_datasets: str = ""  # comma-separated synth/<name> sets added to v1 train, e.g. "v1.1-closeup-train"
    extra_val_datasets: str = ""  # same for the selection val set, e.g. "v1.1-closeup-val"
    validate_every_epochs: int = 1
    augment: bool = True
    log_every_steps: int = 50
    max_steps: int | None = None  # stop early (smoke tests, timing runs)
