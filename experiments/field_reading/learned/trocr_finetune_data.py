"""Training/validation data for fine-tuning TrOCR-small on check field crops.

- Train: context crops (see context_crop_export) of handwritten target-field rows, plus a
  `printed_fraction` share of printed rows from the same fields so the fine-tuned model stays
  usable on printed crops when a non-oracle router sends them its way. Same box jitter and
  photometric augmentation as the CRNN.
- Targets follow the pretrained generation layout: decoder input = [</s>] + pieces, target =
  pieces + [</s>] (no <s>), padded with -100 in the target and <pad> in the input.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from experiments.field_reading.config import TARGET_FIELD_NAMES
from experiments.field_reading.learned.crop_augmentation import augment_line_crop, jitter_box_and_crop
from experiments.field_reading.learned.line_crop_dataset import AUGMENT_PROBABILITY, JITTER_PROBABILITY, read_rgb_image
from experiments.field_reading.learned.trocr_reader import trocr_pixel_values
from experiments.field_reading.learned.trocr_tokenizer import EOS_ID, PAD_ID, XlmRobertaSentencePieceCodec

IGNORED_TARGET_ID = -100
MAX_TARGET_TOKENS = 48


def select_finetune_rows(manifest: pd.DataFrame, printed_fraction: float, seed: int) -> pd.DataFrame:
    """All handwritten target-field rows plus printed rows amounting to `printed_fraction` of the total."""
    target_rows = manifest[manifest.field_name.isin(TARGET_FIELD_NAMES)]
    handwritten_rows = target_rows[target_rows.handwritten]
    printed_rows = target_rows[~target_rows.handwritten]
    printed_count = int(len(handwritten_rows) * printed_fraction / max(1e-9, 1 - printed_fraction))
    printed_sample = printed_rows.sample(min(printed_count, len(printed_rows)), random_state=seed)
    return pd.concat([handwritten_rows, printed_sample]).sample(frac=1.0, random_state=seed).reset_index(drop=True)


class TrocrFinetuneDataset(Dataset):
    """Augmented (crop, token ids) items from context crops."""

    def __init__(self, rows: pd.DataFrame, codec: XlmRobertaSentencePieceCodec, seed: int):
        self.rows, self.codec, self.seed = rows.reset_index(drop=True), codec, seed

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows.iloc[index]
        generator = np.random.default_rng((self.seed, index, torch.randint(0, 2**31, ()).item()))
        context_image = read_rgb_image(row.context_crop_path)
        crop = jitter_box_and_crop(context_image, row.box_in_context, generator,
                                   0.10 if generator.random() < JITTER_PROBABILITY else 0.0)
        if generator.random() < AUGMENT_PROBABILITY:
            crop = augment_line_crop(crop, generator)
        target_ids = self.codec.encode(row.text)[1:][:MAX_TARGET_TOKENS]
        return {"crop": crop, "target_ids": target_ids, "text": row.text}


def collate_trocr_batch(items: list[dict]) -> dict:
    """Pixel values plus teacher-forcing decoder inputs and padded targets."""
    longest = max(len(item["target_ids"]) for item in items)
    decoder_input_ids = torch.full((len(items), longest), PAD_ID, dtype=torch.long)
    target_ids = torch.full((len(items), longest), IGNORED_TARGET_ID, dtype=torch.long)
    for row_index, item in enumerate(items):
        targets = item["target_ids"]
        decoder_input_ids[row_index, :len(targets)] = torch.tensor([EOS_ID] + targets[:-1])
        target_ids[row_index, :len(targets)] = torch.tensor(targets)
    return {"pixel_values": trocr_pixel_values([item["crop"] for item in items]),
            "decoder_input_ids": decoder_input_ids, "decoder_attention_mask": (target_ids != IGNORED_TARGET_ID).long(),
            "target_ids": target_ids, "texts": [item["text"] for item in items]}
