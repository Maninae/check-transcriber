"""Fine-tune TrOCR-small-handwritten on check field crops within a wall-clock budget.

NOT used for the shipped results: the coordinator ruled out fine-tuning on synth v1 handwriting
(its 24 fonts are font recall, not handwriting; see the U4 report). Kept for a future run on
better synthetic or real handwriting, where the SSBI score logged per checkpoint is the criterion.

Run (from the worktree root):
    python -m experiments.field_reading.learned.finetune_trocr --max-minutes 150

- Holds the area MPS lock. Cosine schedule over the planned steps (`--epochs`), but stops at
  `--max-minutes` regardless; validation every `--validate-every` steps on a fixed subset of
  handwritten val rows, scored with the harness's `normalize_field_value` (field_correct).
- Every validation also scores the 78 real SSBI handwriting crops (real_ssbi_scoring rules), the
  actual shipping criterion: synth val alone rewards forgetting real handwriting.
- Every validated checkpoint -> RECOGNIZER_ROOT/trocr_small_handwritten_ft/step_<n>; the synth-best
  one is also written to .../best (save_pretrained + the sentencepiece model, so
  `TrocrCropReader("trocr_small_handwritten_ft")` loads it).
"""

import argparse
import json
import logging
import math
import shutil
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import VisionEncoderDecoderModel

from experiments.field_reading.data_access.field_manifest import load_field_rows
from experiments.field_reading.learned.context_crop_export import CONTEXT_MANIFEST_PATH, SCORED_STATUSES
from experiments.field_reading.learned.line_crop_dataset import read_rgb_image
from experiments.field_reading.learned.mps_lock import hold_mps_lock
from experiments.field_reading.learned.real_ssbi_scoring import comparable_text, load_ssbi_rows, pad_tight_crop
from experiments.field_reading.learned.reading_methods import RECOGNIZER_ROOT
from experiments.field_reading.learned.trocr_finetune_data import (IGNORED_TARGET_ID, TrocrFinetuneDataset,
                                                                   collate_trocr_batch, select_finetune_rows)
from experiments.field_reading.learned.trocr_reader import TrocrCropReader, resolve_trocr_directory
from experiments.field_reading.learned.trocr_tokenizer import SENTENCEPIECE_FILE_NAME, XlmRobertaSentencePieceCodec
from experiments.field_reading.metrics.field_value_parsing import normalize_field_value

logger = logging.getLogger(__name__)

BASE_MODEL_ID = "trocr_small_handwritten"
OUTPUT_DIRECTORY = RECOGNIZER_ROOT / "trocr_small_handwritten_ft"
VALIDATION_ROW_COUNT = 600
WARMUP_STEPS = 300


def validation_accuracy(reader: TrocrCropReader, rows: pd.DataFrame, crops: list) -> float:
    """field_correct rate of greedy reads on the validation subset."""
    reader.model.eval()
    reads = reader.read_crops(crops)
    reader.model.train()
    correct = [normalize_field_value(field, text) is not None and normalize_field_value(field, text) == normalize_field_value(field, truth)
               for (text, _), field, truth in zip(reads, rows.field_name, rows.text)]
    return float(sum(correct) / len(correct))


def ssbi_exact_by_field(reader: TrocrCropReader, ssbi_rows: pd.DataFrame, ssbi_crops: list) -> dict[str, dict[str, float]]:
    """Exact-match rate per field on the real SSBI crops, for RGB and grey+autocontrast input."""
    reader.model.eval()
    by_input = {}
    for input_name, grey_input in [("rgb", False), ("grey", True)]:
        reader.grey_input = grey_input
        reads = reader.read_crops(ssbi_crops)
        exact = [comparable_text(field, text) == comparable_text(field, truth)
                 for (text, _), field, truth in zip(reads, ssbi_rows.field_name, ssbi_rows.text)]
        by_input[input_name] = pd.Series(exact, index=ssbi_rows.field_name.values).groupby(level=0).mean().round(3).to_dict()
    reader.grey_input = False
    reader.model.train()
    return by_input


def save_checkpoint(model: VisionEncoderDecoderModel, base_directory, checkpoint_directory) -> None:
    """save_pretrained plus the sentencepiece model the reader needs."""
    model.save_pretrained(checkpoint_directory)
    shutil.copy(base_directory / SENTENCEPIECE_FILE_NAME, checkpoint_directory / SENTENCEPIECE_FILE_NAME)


def finetune(arguments: argparse.Namespace) -> None:
    """Training loop (runs under the MPS lock)."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(arguments.seed)
    base_directory = resolve_trocr_directory(BASE_MODEL_ID)
    codec = XlmRobertaSentencePieceCodec(base_directory / SENTENCEPIECE_FILE_NAME)
    training_rows = select_finetune_rows(pd.read_json(CONTEXT_MANIFEST_PATH, lines=True), arguments.printed_fraction, arguments.seed)
    loader = DataLoader(TrocrFinetuneDataset(training_rows, codec, arguments.seed), batch_size=arguments.batch_size, shuffle=True,
                        num_workers=arguments.workers, collate_fn=collate_trocr_batch, persistent_workers=True, drop_last=True)
    validation_rows = load_field_rows("val")
    validation_rows = validation_rows[validation_rows.status.isin(SCORED_STATUSES) & validation_rows.handwritten]
    validation_rows = validation_rows.sample(VALIDATION_ROW_COUNT, random_state=arguments.seed)
    validation_crops = [read_rgb_image(path) for path in validation_rows.field_crop_path]
    ssbi_rows = load_ssbi_rows()
    ssbi_crops = [pad_tight_crop(read_rgb_image(path)) for path in ssbi_rows.crop_path]
    reader = TrocrCropReader(BASE_MODEL_ID, device)
    model: VisionEncoderDecoderModel = reader.model
    model.train()
    planned_steps = arguments.epochs * len(loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=arguments.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / WARMUP_STEPS) *
                                                  0.5 * (1 + math.cos(math.pi * min(1.0, step / planned_steps))))
    logger.info("fine-tuning on %d rows (%d hw), %d planned steps, device %s", len(training_rows),
                int(training_rows.handwritten.sum()), planned_steps, device)
    best_accuracy = validation_accuracy(reader, validation_rows, validation_crops)
    logger.info("zero-shot val field accuracy on the %d handwritten rows: %.4f; SSBI exact %s", len(validation_rows),
                best_accuracy, ssbi_exact_by_field(reader, ssbi_rows, ssbi_crops))
    started, step, history = time.time(), 0, []
    out_of_time = False
    while step < planned_steps and not out_of_time:
        for batch in loader:
            output = model(pixel_values=batch["pixel_values"].to(device), decoder_input_ids=batch["decoder_input_ids"].to(device),
                           decoder_attention_mask=batch["decoder_attention_mask"].to(device))
            loss = torch.nn.functional.cross_entropy(output.logits.float().flatten(0, 1), batch["target_ids"].to(device).flatten(),
                                                     ignore_index=IGNORED_TARGET_ID, label_smoothing=0.05)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            elapsed_minutes = (time.time() - started) / 60
            if step % 100 == 0:
                logger.info("step %d/%d loss %.4f lr %.2e elapsed %.1f min", step, planned_steps, loss.item(),
                            scheduler.get_last_lr()[0], elapsed_minutes)
            out_of_time = elapsed_minutes > arguments.max_minutes
            if step % arguments.validate_every == 0 or out_of_time or step >= planned_steps:
                accuracy = validation_accuracy(reader, validation_rows, validation_crops)
                ssbi_exact = ssbi_exact_by_field(reader, ssbi_rows, ssbi_crops)
                history.append({"step": step, "elapsed_min": elapsed_minutes, "val_hw_field_accuracy": accuracy,
                                "ssbi_exact": ssbi_exact})
                logger.info("step %d val hw field accuracy %.4f (best %.4f) SSBI exact %s", step, accuracy, best_accuracy, ssbi_exact)
                save_checkpoint(model, base_directory, OUTPUT_DIRECTORY / f"step_{step}")
                (OUTPUT_DIRECTORY / "history.json").write_text(json.dumps(history, indent=1))
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    save_checkpoint(model, base_directory, OUTPUT_DIRECTORY / "best")
            if out_of_time or step >= planned_steps:
                break
    (OUTPUT_DIRECTORY / "history.json").write_text(json.dumps(history, indent=1))
    logger.info("done: %d steps in %.1f min, best val hw field accuracy %.4f", step, (time.time() - started) / 60, best_accuracy)


def main() -> None:
    """Parse arguments and fine-tune under the MPS lock."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-minutes", type=float, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--printed-fraction", type=float, default=0.2)
    parser.add_argument("--validate-every", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    with hold_mps_lock("finetune_trocr"):
        finetune(arguments)


if __name__ == "__main__":
    main()
