"""Train the upside-down classifier on the GT-derived crops (see build_orientation_crops).

    python -m experiments.detection.orientation.train_upside_down_classifier --epochs 8

Selects the epoch with the best val accuracy; eval is never read here. Augmentation
mimics what a real detector plus phone photo does to a crop: brightness/contrast
changes, blur, noise, and a small random translation/scale of the crop.
"""

import argparse
import json
import logging

import numpy as np
import torch
from torch import nn
from torch.nn import functional

from experiments.detection.orientation.build_orientation_crops import ORIENTATION_EXPERIMENTS_DIRECTORY
from experiments.detection.orientation.upside_down_classifier import UpsideDownCheckClassifier

logger = logging.getLogger(__name__)

CLASSIFIER_WEIGHTS_PATH = ORIENTATION_EXPERIMENTS_DIRECTORY / "upside_down_classifier.pt"
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 128
MAX_TRANSLATION_FRACTION = 0.04
MAX_SCALE_CHANGE = 0.06


def load_split_crops(split_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """(N, 1, H, W) float crops in [0, 1] and (N,) float labels."""
    crops = np.load(ORIENTATION_EXPERIMENTS_DIRECTORY / f"{split_name}__crops.npy")
    labels = np.load(ORIENTATION_EXPERIMENTS_DIRECTORY / f"{split_name}__labels.npy")
    return torch.from_numpy(crops).float().unsqueeze(1) / 255.0, torch.from_numpy(labels).float()


def augment_crop_batch(crop_batch: torch.Tensor) -> torch.Tensor:
    """Photometric jitter plus a small random affine shift/scale, applied per sample."""
    batch_size = crop_batch.shape[0]
    device = crop_batch.device
    contrast = torch.empty(batch_size, 1, 1, 1, device=device).uniform_(0.6, 1.4)
    brightness = torch.empty(batch_size, 1, 1, 1, device=device).uniform_(-0.2, 0.2)
    crop_batch = (crop_batch - 0.5) * contrast + 0.5 + brightness
    scale = 1 + torch.empty(batch_size, device=device).uniform_(-MAX_SCALE_CHANGE, MAX_SCALE_CHANGE)
    affine = torch.zeros(batch_size, 2, 3, device=device)
    affine[:, 0, 0] = scale
    affine[:, 1, 1] = scale
    affine[:, :, 2] = torch.empty(batch_size, 2, device=device).uniform_(-2, 2) * MAX_TRANSLATION_FRACTION
    grid = functional.affine_grid(affine, list(crop_batch.shape), align_corners=False)
    crop_batch = functional.grid_sample(crop_batch, grid, padding_mode="border", align_corners=False)
    if torch.rand(1).item() < 0.5:
        crop_batch = functional.avg_pool2d(crop_batch, 3, stride=1, padding=1)
    crop_batch = crop_batch + torch.randn_like(crop_batch) * torch.empty(1, device=device).uniform_(0, 0.05)
    return crop_batch.clamp(0, 1)


def evaluate_accuracy(model: nn.Module, crops: torch.Tensor, labels: torch.Tensor, device: str) -> float:
    """Fraction of crops classified correctly."""
    model.eval()
    correct_count = 0
    with torch.no_grad():
        for start in range(0, len(crops), BATCH_SIZE * 4):
            logits = model(crops[start : start + BATCH_SIZE * 4].to(device)).cpu()
            correct_count += int(((logits > 0).float() == labels[start : start + BATCH_SIZE * 4]).sum())
    return correct_count / len(crops)


def main() -> None:
    """Train, keep the best-val checkpoint, write a small JSON summary."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-steps", type=int, default=None, help="smoke-test cap")
    arguments = parser.parse_args()
    torch.manual_seed(0)
    train_crops, train_labels = load_split_crops("train")
    val_crops, val_labels = load_split_crops("val")
    model = UpsideDownCheckClassifier().to(arguments.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = len(train_crops) // BATCH_SIZE
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, LEARNING_RATE, total_steps=steps_per_epoch * arguments.epochs
    )
    best_val_accuracy, step_count = -1.0, 0
    for epoch in range(arguments.epochs):
        model.train()
        permutation = torch.randperm(len(train_crops))
        for batch_start in range(0, steps_per_epoch * BATCH_SIZE, BATCH_SIZE):
            batch_indices = permutation[batch_start : batch_start + BATCH_SIZE]
            crop_batch = augment_crop_batch(train_crops[batch_indices].to(arguments.device))
            loss = functional.binary_cross_entropy_with_logits(
                model(crop_batch), train_labels[batch_indices].to(arguments.device)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            step_count += 1
            if arguments.max_steps and step_count >= arguments.max_steps:
                logger.info("smoke test stopped at step %d, loss %.4f", step_count, loss.item())
                return
        val_accuracy = evaluate_accuracy(model, val_crops, val_labels, arguments.device)
        logger.info("epoch %d loss %.4f val accuracy %.4f", epoch, loss.item(), val_accuracy)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), CLASSIFIER_WEIGHTS_PATH)
    summary = {"best_val_accuracy": best_val_accuracy, "epochs": arguments.epochs, "weights": str(CLASSIFIER_WEIGHTS_PATH)}
    (ORIENTATION_EXPERIMENTS_DIRECTORY / "training_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("best val accuracy %.4f", best_val_accuracy)


if __name__ == "__main__":
    main()
