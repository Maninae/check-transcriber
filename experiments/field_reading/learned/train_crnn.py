"""Train a CRNN-CTC line recognizer on train field crops; select the checkpoint on a val subset.

Examples (from the worktree root):
    python -m experiments.field_reading.learned.train_crnn --run-name crnn_general_h32 \
        --variant crnn_small_h32 --charset general --epochs 12
    python -m experiments.field_reading.learned.train_crnn --run-name crnn_amount_h32 --variant crnn_small_h32 \
        --charset amount --fields amount_numeric --epochs 3 --init-from crnn_general_h32 --learning-rate 3e-4

- Holds the area MPS lock for the whole run (released on exit or failure).
- CTC loss is computed on CPU (MPS has no native CTC kernel); the network runs on MPS.
- `--init-from <run>` starts from another run's best.pt; tensors whose shape differs (the
  classifier head when the charset changes) keep their fresh initialization.
- Writes `<run>/best.pt`, `<run>/last.pt` and `<run>/history.jsonl` under RECOGNIZER_CHECKPOINT_ROOT.
"""

import argparse
import json
import logging
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from experiments.field_reading.config import FIELD_READING_MODEL_ROOT
from experiments.field_reading.data_access.field_manifest import load_field_rows
from experiments.field_reading.learned.context_crop_export import CONTEXT_MANIFEST_PATH, SCORED_STATUSES
from experiments.field_reading.learned.crnn_model import CRNN_VARIANT_REGISTRY, CrnnLineRecognizer, count_parameters
from experiments.field_reading.learned.ctc_decoding import decode_ctc_batch
from experiments.field_reading.learned.line_crop_dataset import (ContextCropTrainingDataset, FieldCropEvaluationDataset,
                                                                 WidthBucketedBatchSampler, collate_line_batch)
from experiments.field_reading.learned.mps_lock import hold_mps_lock
from experiments.field_reading.learned.quick_text_scoring import character_error_rate, is_exact_match
from experiments.field_reading.learned.text_charset import CHARSET_REGISTRY, CTC_BLANK_INDEX

logger = logging.getLogger(__name__)

RECOGNIZER_CHECKPOINT_ROOT = FIELD_READING_MODEL_ROOT / "recognizers"
VALIDATION_SUBSET_SIZE = 3000
VALIDATION_SUBSET_SEED = 7
GRADIENT_CLIP_NORM = 5.0
LOG_EVERY_STEPS = 200


def build_validation_rows(field_names: list[str]) -> pd.DataFrame:
    """A fixed random subset of scored val rows for per-epoch model selection."""
    rows = load_field_rows("val")
    rows = rows[rows.status.isin(SCORED_STATUSES) & rows.field_name.isin(field_names)]
    return rows.sample(min(VALIDATION_SUBSET_SIZE, len(rows)), random_state=VALIDATION_SUBSET_SEED)


def load_matching_weights(model: nn.Module, checkpoint_path) -> None:
    """Copy every same-shaped tensor from a checkpoint into `model` (others stay freshly initialized)."""
    source_state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["state_dict"]
    target_state = model.state_dict()
    matching = {key: value for key, value in source_state.items() if key in target_state and value.shape == target_state[key].shape}
    model.load_state_dict(matching, strict=False)
    logger.info("initialized %d/%d tensors from %s", len(matching), len(target_state), checkpoint_path)


@torch.no_grad()
def evaluate_on_loader(model: nn.Module, loader: DataLoader, charset, device: str) -> dict:
    """Per-field exact match / CER on an un-augmented loader, plus their field means."""
    model.eval()
    records = []
    for batch in loader:
        log_probabilities = model(batch["line_images"].to(device)).float().cpu().numpy()
        decoded = decode_ctc_batch(log_probabilities, batch["input_timesteps"].tolist(), charset)
        for (text, _), truth, field_name in zip(decoded, batch["texts"], batch["field_names"]):
            records.append({"field_name": field_name, "exact": is_exact_match(text, truth),
                            "cer": character_error_rate(text, truth)})
    model.train()
    per_field = pd.DataFrame(records).groupby("field_name")[["exact", "cer"]].mean()
    return {"exact_mean": float(per_field.exact.mean()), "cer_mean": float(per_field.cer.mean()),
            "per_field": per_field.round(4).to_dict("index")}


def train(arguments: argparse.Namespace) -> None:
    """Run the full training loop under the MPS lock."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    variant = CRNN_VARIANT_REGISTRY[arguments.variant]
    charset = CHARSET_REGISTRY[arguments.charset]
    run_directory = RECOGNIZER_CHECKPOINT_ROOT / arguments.run_name
    run_directory.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_json(CONTEXT_MANIFEST_PATH, lines=True)
    field_names = arguments.fields or sorted(manifest.field_name.unique())
    manifest = manifest[manifest.field_name.isin(field_names)]
    training_dataset = ContextCropTrainingDataset(manifest, variant, charset, seed=arguments.seed)
    validation_fields = [name for name in field_names if name != "bank_name"]
    validation_dataset = FieldCropEvaluationDataset(build_validation_rows(validation_fields), variant, charset)
    training_loader = DataLoader(training_dataset, collate_fn=collate_line_batch, num_workers=arguments.workers,
                                 persistent_workers=True, batch_sampler=WidthBucketedBatchSampler(
                                     training_dataset.aspect_ratios, arguments.batch_size, arguments.seed))
    validation_loader = DataLoader(validation_dataset, batch_size=64, collate_fn=collate_line_batch, num_workers=0)
    torch.manual_seed(arguments.seed)
    model = CrnnLineRecognizer(variant, charset.num_classes)
    if arguments.init_from:
        load_matching_weights(model, RECOGNIZER_CHECKPOINT_ROOT / arguments.init_from / "best.pt")
    model = model.to(device)
    logger.info("%s: %d train rows, %d val rows, %.2fM params, device %s", arguments.run_name, len(training_dataset),
                len(validation_dataset), count_parameters(model) / 1e6, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=arguments.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=arguments.learning_rate, pct_start=0.1,
                                                    total_steps=arguments.epochs * len(training_loader))
    ctc_loss = nn.CTCLoss(blank=CTC_BLANK_INDEX, zero_infinity=True)
    best_exact_mean, started = -1.0, time.time()
    checkpoint_common = {"variant_name": arguments.variant, "charset_name": arguments.charset, "fields": field_names}
    for epoch in range(arguments.epochs):
        running_losses = []
        for step, batch in enumerate(training_loader):
            log_probabilities = model(batch["line_images"].to(device))
            loss = ctc_loss(log_probabilities.permute(1, 0, 2).float().cpu(), batch["targets"],
                            batch["input_timesteps"], batch["target_lengths"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()
            scheduler.step()
            running_losses.append(loss.item())
            if step % LOG_EVERY_STEPS == 0:
                logger.info("epoch %d step %d/%d loss %.4f lr %.2e elapsed %.1f min", epoch, step, len(training_loader),
                            np.mean(running_losses[-LOG_EVERY_STEPS:]), scheduler.get_last_lr()[0], (time.time() - started) / 60)
        scores = evaluate_on_loader(model, validation_loader, charset, device)
        record = {"epoch": epoch, "train_loss": float(np.mean(running_losses)), "elapsed_min": (time.time() - started) / 60, **scores}
        logger.info("epoch %d val exact_mean %.4f cer_mean %.4f per_field %s", epoch, scores["exact_mean"],
                    scores["cer_mean"], {k: v["exact"] for k, v in scores["per_field"].items()})
        with open(run_directory / "history.jsonl", "a") as handle:
            handle.write(json.dumps(record) + "\n")
        checkpoint = {**checkpoint_common, "epoch": epoch, "state_dict": model.state_dict(), "val": scores}
        torch.save(checkpoint, run_directory / "last.pt")
        if scores["exact_mean"] > best_exact_mean:
            best_exact_mean = scores["exact_mean"]
            torch.save(checkpoint, run_directory / "best.pt")
    logger.info("done in %.1f min, best val exact_mean %.4f", (time.time() - started) / 60, best_exact_mean)


def main() -> None:
    """Parse arguments and train under the MPS lock."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--variant", default="crnn_small_h32", choices=sorted(CRNN_VARIANT_REGISTRY))
    parser.add_argument("--charset", default="general", choices=sorted(CHARSET_REGISTRY))
    parser.add_argument("--fields", nargs="*", default=None, help="restrict to these field names (default: all)")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init-from", default=None, help="run name whose best.pt initializes matching tensors")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with hold_mps_lock(arguments.run_name):
        train(arguments)


if __name__ == "__main__":
    main()
