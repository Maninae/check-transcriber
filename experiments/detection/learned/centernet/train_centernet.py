"""Train the CenterNet check detector on the v1 synthetic train split.

    python -m experiments.detection.learned.centernet.train_centernet --run-name mnv3l_768 --epochs 30

Every `CenterNetTrainingConfig` field is a `--kebab-case` flag. Outputs go to
`<vega>/experiments/detection/runs/centernet/<run name>/`: `config.json`,
`training_log.jsonl` (one line per logged step / validation), `last.pt`, `best.pt`
(best quick-val F1@0.90, see `quick_validation.py`).

- AdamW, linear warmup then cosine decay to zero over all steps.
- `--validation-split train` scores the quick validation on the training scenes
  themselves (overfit smoke tests).
"""

import argparse
import dataclasses
import json
import logging
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.detection.dataset.scene_annotations import load_split_scene_annotations
from experiments.detection.learned.centernet.centernet_config import CENTERNET_RUNS_ROOT, CenterNetTrainingConfig
from experiments.detection.learned.centernet.centernet_losses import centernet_total_loss
from experiments.detection.learned.centernet.centernet_model import CenterNetCheckDetector
from experiments.detection.learned.centernet.check_scene_dataset import TARGET_KEYS, CheckSceneCenterNetDataset
from experiments.detection.learned.centernet.quick_validation import run_quick_validation

logger = logging.getLogger(__name__)


def parse_training_config() -> tuple[CenterNetTrainingConfig, str]:
    """Build the config from CLI flags (one per dataclass field) plus `--validation-split`."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for config_field in dataclasses.fields(CenterNetTrainingConfig):
        flag = "--" + config_field.name.replace("_", "-")
        if config_field.type is bool:
            parser.add_argument(flag, type=lambda text: text.lower() in ("1", "true", "yes"), default=config_field.default)
        elif config_field.type in (int, float, str):
            parser.add_argument(flag, type=config_field.type, default=config_field.default)
        else:  # `int | None` fields
            parser.add_argument(flag, type=int, default=config_field.default)
    parser.add_argument("--validation-split", default="val", choices=("val", "train"))
    arguments = vars(parser.parse_args())
    validation_split = arguments.pop("validation_split")
    return CenterNetTrainingConfig(**arguments), validation_split


def learning_rate_multiplier(step: int, warmup_steps: int, total_steps: int) -> float:
    """Linear warmup to 1, then cosine to 0."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def append_log_line(log_path: Path, record: dict) -> None:
    """Append one JSON record to the run's training log."""
    with open(log_path, "a") as log_file:
        log_file.write(json.dumps(record) + "\n")


def save_checkpoint(path: Path, model: torch.nn.Module, config: CenterNetTrainingConfig, epoch: int, selection_metric: float) -> None:
    """Weights plus the config needed to rebuild the model."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "training_config": dataclasses.asdict(config),
            "epoch": epoch,
            "selection_metric": selection_metric,
        },
        path,
    )


def train(config: CenterNetTrainingConfig, validation_split: str) -> Path:
    """Run training; returns the run directory."""
    seed_everything(config.seed)
    run_directory = CENTERNET_RUNS_ROOT / config.run_name
    run_directory.mkdir(parents=True, exist_ok=True)
    (run_directory / "config.json").write_text(json.dumps(dataclasses.asdict(config) | {"validation_split": validation_split}, indent=2))
    log_path = run_directory / "training_log.jsonl"

    train_scenes = load_split_scene_annotations("train", limit=config.train_scene_limit)
    validation_scenes = (
        train_scenes[: config.val_scene_limit]
        if validation_split == "train"
        else load_split_scene_annotations("val", limit=config.val_scene_limit)
    )
    train_dataset = CheckSceneCenterNetDataset(train_scenes, config.input_size_pixels, config.augment, config.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=len(train_scenes) > config.batch_size,
        num_workers=config.dataloader_workers,
    )
    model = CenterNetCheckDetector(config.backbone_name, config.neck_channels).to(config.device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    total_steps = config.max_steps or config.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: learning_rate_multiplier(step, config.warmup_steps, total_steps)
    )
    logger.info("run %s: %d train scenes, %d steps/epoch, %d total steps", config.run_name, len(train_scenes), len(train_loader), total_steps)

    step, best_selection_metric, window_start_time = 0, -1.0, time.perf_counter()
    for epoch in range(config.epochs):
        train_dataset.epoch = epoch
        for batch in train_loader:
            batch = {key: batch[key].to(config.device) for key in ("input_image", *TARGET_KEYS)}
            losses = centernet_total_loss(model(batch["input_image"]), batch, config.heatmap_loss_weight, config.corner_offset_loss_weight)
            optimizer.zero_grad(set_to_none=True)
            losses["total_loss"].backward()
            optimizer.step()
            scheduler.step()
            step += 1
            if step % config.log_every_steps == 0 or step == total_steps:
                seconds_per_step = (time.perf_counter() - window_start_time) / config.log_every_steps
                record = {"step": step, "epoch": epoch, "seconds_per_step": round(seconds_per_step, 3),
                          "learning_rate": scheduler.get_last_lr()[0]} | {key: round(float(value), 5) for key, value in losses.items()}
                append_log_line(log_path, record)
                logger.info("%s", record)
                window_start_time = time.perf_counter()
            if step >= total_steps:
                break
        is_last_epoch = step >= total_steps or epoch == config.epochs - 1
        if (epoch + 1) % config.validate_every_epochs == 0 or is_last_epoch:
            selection_metric, headline = run_quick_validation(model, validation_scenes, config.input_size_pixels, config.device)
            append_log_line(log_path, {"step": step, "epoch": epoch, "selection_f1_at_0.90": selection_metric, "headline": headline})
            logger.info("epoch %d val: %s", epoch, headline)
            save_checkpoint(run_directory / "last.pt", model, config, epoch, selection_metric)
            if selection_metric > best_selection_metric:
                best_selection_metric = selection_metric
                save_checkpoint(run_directory / "best.pt", model, config, epoch, selection_metric)
            window_start_time = time.perf_counter()
        if is_last_epoch:
            break
    logger.info("done; best F1@0.90 %.4f; outputs in %s", best_selection_metric, run_directory)
    return run_directory


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    config, validation_split = parse_training_config()
    train(config, validation_split)


if __name__ == "__main__":
    main()
