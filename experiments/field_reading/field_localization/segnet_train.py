"""Train the field-mask localizer on train crops; select the checkpoint by val mean IoU.

Holds the area's MPS lock for the whole run. Loss = BCE + batch-level soft Dice per field channel.
Stops early if the wall-clock budget is spent or after `--stop-after-epoch`. Checkpoints under
LOCALIZATION_MODEL_ROOT: `<method>.pt` (best val) and `<method>__last.pt` (resumable: model, optimizer,
scheduler, epoch). `--resume` continues the same cosine schedule from the last checkpoint; a model-only
checkpoint resumes with fresh optimizer moments and the scheduler fast-forwarded to the right step.

Run: nohup python -m experiments.field_reading.field_localization.segnet_train > <log> 2>&1 &
"""

import argparse
import logging
import os
import random
import time

import numpy as np
import torch
from torch.nn import functional
from torch.utils.data import DataLoader

from experiments.field_reading.field_localization.localization_config import (LOCALIZATION_MODEL_ROOT,
                                                                              PRETRAINED_WEIGHTS_CACHE_ROOT)
from experiments.field_reading.field_localization.localization_metrics import (score_prediction_rows,
                                                                               summarize_scored_records)
from experiments.field_reading.field_localization.localization_targets import load_localization_checks
from experiments.field_reading.field_localization.mps_lock import hold_mps_lock
from experiments.field_reading.field_localization.segnet_config import SEGNET_METHOD_ID
from experiments.field_reading.field_localization.segnet_dataset import FieldMaskDataset
from experiments.field_reading.field_localization.segnet_inference import predict_field_rows
from experiments.field_reading.field_localization.segnet_model import FieldMaskSegmentationNet

logger = logging.getLogger(__name__)

DICE_SMOOTHING = 1.0
WARMUP_ITERATIONS = 300
VAL_SELECTION_CHECK_COUNT = 600
DATALOADER_WORKER_COUNT = 2   # machine-wide cap while the detector owner trains (16 GB, heavy swap)


def field_mask_loss(predicted_logits: torch.Tensor, target_masks: torch.Tensor) -> torch.Tensor:
    """BCE over all cells + (1 - soft Dice) per channel pooled over the batch."""
    bce = functional.binary_cross_entropy_with_logits(predicted_logits, target_masks)
    probabilities = torch.sigmoid(predicted_logits)
    intersection = (probabilities * target_masks).sum(dim=(0, 2, 3))
    total = probabilities.sum(dim=(0, 2, 3)) + target_masks.sum(dim=(0, 2, 3))
    dice = (2 * intersection + DICE_SMOOTHING) / (total + DICE_SMOOTHING)
    return bce + (1 - dice).mean()


def evaluate_val_mean_iou(model: torch.nn.Module, val_checks: list[dict], device: torch.device) -> float:
    """Mean over fields of val mean IoU (status ok), no confidence gate."""
    prediction_rows = predict_field_rows(model, val_checks, device)
    scored = score_prediction_rows(val_checks, {row["row_key"]: row for row in prediction_rows})
    summary = summarize_scored_records(scored[scored.status == "ok"], ["field_name"])
    logger.info("val per-field IoU: %s", dict(zip(summary.field_name, summary.mean_iou.round(3))))
    return float(summary.mean_iou.mean())


def restore_training_state(model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler, best_checkpoint_path,
                           last_checkpoint_path, iterations_per_epoch: int) -> tuple[int, float]:
    """Load the newest checkpoint into model/optimizer/scheduler; returns (next epoch, best val IoU so far)."""
    source_path = last_checkpoint_path if last_checkpoint_path.exists() else best_checkpoint_path
    checkpoint = torch.load(source_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    if "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    else:
        logger.warning("model-only checkpoint: fresh optimizer moments, scheduler fast-forwarded")
        for _ in range((checkpoint["epoch"] + 1) * iterations_per_epoch):
            scheduler.step()
    best_val_iou = checkpoint.get("best_val_mean_iou", checkpoint["val_mean_iou"])
    logger.info("resumed from %s after epoch %d (lr now %.2e)", source_path, checkpoint["epoch"],
                optimizer.param_groups[0]["lr"])
    return checkpoint["epoch"] + 1, best_val_iou


def train_field_mask_net(epochs: int, batch_size: int, learning_rate: float, budget_minutes: float, resume: bool,
                         stop_after_epoch: int | None) -> None:
    """Full training loop (see module docstring)."""
    torch.manual_seed(0), np.random.seed(0), random.seed(0)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train_checks = load_localization_checks("train")
    val_checks = load_localization_checks("val")
    val_checks = [val_checks[index] for index in np.random.default_rng(0).permutation(len(val_checks))[:VAL_SELECTION_CHECK_COUNT]]
    train_dataset = FieldMaskDataset(train_checks, augment=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=DATALOADER_WORKER_COUNT, drop_last=True,
                              persistent_workers=False)
    model = FieldMaskSegmentationNet(pretrained_encoder=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    total_iterations = epochs * len(train_loader)
    # float(): numpy scalars in scheduler state would make the checkpoint unloadable with weights_only=True.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: float(min(1.0, (step + 1) / WARMUP_ITERATIONS) *
                                                  0.5 * (1 + np.cos(np.pi * min(step, total_iterations) / total_iterations))))
    checkpoint_path = LOCALIZATION_MODEL_ROOT / f"{SEGNET_METHOD_ID}.pt"
    last_checkpoint_path = LOCALIZATION_MODEL_ROOT / f"{SEGNET_METHOD_ID}__last.pt"
    start_time, best_val_iou, first_epoch = time.time(), -1.0, 0
    if resume:
        first_epoch, best_val_iou = restore_training_state(model, optimizer, scheduler, checkpoint_path,
                                                           last_checkpoint_path, len(train_loader))
    for epoch_index in range(first_epoch, epochs):
        train_dataset.epoch_index = epoch_index
        model.train()
        running_loss, epoch_start = 0.0, time.time()
        for iteration_index, batch in enumerate(train_loader):
            loss = field_mask_loss(model(batch["canvas_image"].to(device)), batch["field_masks"].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            if iteration_index % 100 == 0:
                logger.info("epoch %d iter %d/%d loss %.4f (%.1f s)", epoch_index, iteration_index, len(train_loader),
                            loss.item(), time.time() - epoch_start)
        val_iou = evaluate_val_mean_iou(model, val_checks, device)
        logger.info("epoch %d done: train loss %.4f, val mean IoU %.4f, epoch %.0f s, total %.1f min", epoch_index,
                    running_loss / len(train_loader), val_iou, time.time() - epoch_start, (time.time() - start_time) / 60)
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save({"model_state": model.state_dict(), "epoch": epoch_index, "val_mean_iou": val_iou}, checkpoint_path)
            logger.info("saved best checkpoint to %s", checkpoint_path)
        torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(), "epoch": epoch_index, "val_mean_iou": val_iou,
                    "best_val_mean_iou": best_val_iou}, last_checkpoint_path)
        logger.info("saved resumable checkpoint to %s", last_checkpoint_path)
        if stop_after_epoch is not None and epoch_index >= stop_after_epoch:
            logger.info("stopping after epoch %d as requested", epoch_index)
            break
        if (time.time() - start_time) / 60 > budget_minutes:
            logger.warning("time budget of %.0f min spent; stopping after epoch %d", budget_minutes, epoch_index)
            break
    logger.info("training finished in %.1f min, best val mean IoU %.4f", (time.time() - start_time) / 60, best_val_iou)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--budget-minutes", type=float, default=80)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int, default=None)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.environ.setdefault("HF_HOME", str(PRETRAINED_WEIGHTS_CACHE_ROOT))
    LOCALIZATION_MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    with hold_mps_lock("segnet training"):
        train_field_mask_net(arguments.epochs, arguments.batch_size, arguments.learning_rate, arguments.budget_minutes,
                             arguments.resume, arguments.stop_after_epoch)


if __name__ == "__main__":
    main()
