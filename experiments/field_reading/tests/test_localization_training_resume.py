"""Resuming the field-mask trainer continues the same schedule instead of restarting it."""

import numpy as np
import torch

from experiments.field_reading.field_localization.segnet_train import restore_training_state

ITERATIONS_PER_EPOCH = 10
TOTAL_ITERATIONS = 30


def make_model_optimizer_scheduler():
    """Tiny linear model with the trainer's cosine LambdaLR shape (no warm-up)."""
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: float(0.5 * (1 + np.cos(np.pi * min(step, TOTAL_ITERATIONS) / TOTAL_ITERATIONS))))
    return model, optimizer, scheduler


def test_model_only_checkpoint_fast_forwards_scheduler(tmp_path):
    """A weights-only checkpoint after epoch 0 resumes at epoch 1 with the lr of step 10."""
    model, _, _ = make_model_optimizer_scheduler()
    best_path = tmp_path / "best.pt"
    torch.save({"model_state": model.state_dict(), "epoch": 0, "val_mean_iou": 0.8}, best_path)
    resumed_model, optimizer, scheduler = make_model_optimizer_scheduler()
    next_epoch, best_iou = restore_training_state(resumed_model, optimizer, scheduler, best_path,
                                                  tmp_path / "missing_last.pt", ITERATIONS_PER_EPOCH)
    assert next_epoch == 1 and best_iou == 0.8
    assert optimizer.param_groups[0]["lr"] == float(0.5 * (1 + np.cos(np.pi * 10 / 30)))
    assert torch.equal(resumed_model.weight, model.weight)


def test_full_checkpoint_restores_scheduler_state(tmp_path):
    """A resumable checkpoint restores the scheduler step exactly."""
    model, optimizer, scheduler = make_model_optimizer_scheduler()
    for _ in range(20):
        optimizer.step()
        scheduler.step()
    last_path = tmp_path / "last.pt"
    torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(), "epoch": 1, "val_mean_iou": 0.7,
                "best_val_mean_iou": 0.9}, last_path)
    _, resumed_optimizer, resumed_scheduler = make_model_optimizer_scheduler()
    next_epoch, best_iou = restore_training_state(model, resumed_optimizer, resumed_scheduler, tmp_path / "best.pt",
                                                  last_path, ITERATIONS_PER_EPOCH)
    assert next_epoch == 2 and best_iou == 0.9
    assert resumed_scheduler.last_epoch == 20
    assert resumed_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
