"""Datasets of field crops for the CTC recognizers, plus a width-bucketed batch sampler.

- `ContextCropTrainingDataset`: train rows from `context_crop_export`; each item jitters the box
  inside the context crop, augments, and preprocesses. Items are data_dicts.
- `FieldCropEvaluationDataset`: val/eval rows read from synth's ground-truth `field_crop_path`
  (oracle localization), no augmentation.
- `WidthBucketedBatchSampler`: groups rows of similar aspect ratio so padding stays small.
"""

import logging

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from experiments.field_reading.learned.crnn_model import CrnnVariant, timesteps_for_width
from experiments.field_reading.learned.crop_augmentation import augment_line_crop, jitter_box_and_crop
from experiments.field_reading.learned.line_image_preprocessing import (MPS_WIDTH_MULTIPLE, pad_line_batch,
                                                                       preprocess_line_image)
from experiments.field_reading.learned.text_charset import TextCharset

logger = logging.getLogger(__name__)

JITTER_PROBABILITY = 0.85
AUGMENT_PROBABILITY = 0.9
BUCKET_SPAN_BATCHES = 50


def read_rgb_image(image_path: str) -> np.ndarray:
    """Load an image file as RGB uint8."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"unreadable image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class ContextCropTrainingDataset(Dataset):
    """Augmented training lines cut from wide-margin context crops."""

    def __init__(self, manifest_rows: pd.DataFrame, variant: CrnnVariant, charset: TextCharset, seed: int):
        covered = manifest_rows.text.map(charset.covers)
        if (~covered).any():
            logger.warning("dropping %d rows with characters outside charset %s", (~covered).sum(), charset.name)
        self.rows = manifest_rows[covered].reset_index(drop=True)
        self.variant, self.charset, self.seed = variant, charset, seed
        boxes = np.array(self.rows.box_in_context.tolist())
        self.aspect_ratios = (boxes[:, 2] - boxes[:, 0]) / np.maximum(1.0, boxes[:, 3] - boxes[:, 1])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows.iloc[index]
        generator = np.random.default_rng((self.seed, index, torch.randint(0, 2**31, ()).item()))
        context_image = read_rgb_image(row.context_crop_path)
        jitter = 0.10 if generator.random() < JITTER_PROBABILITY else 0.0
        crop = jitter_box_and_crop(context_image, row.box_in_context, generator, jitter)
        if generator.random() < AUGMENT_PROBABILITY:
            crop = augment_line_crop(crop, generator)
        line_image = preprocess_line_image(crop, self.variant.input_height, self.variant.min_width, self.variant.max_width)
        return {"line_image": line_image, "label_indices": self.charset.encode(row.text), "text": row.text,
                "field_name": row.field_name, "row_key": row.row_key}


class FieldCropEvaluationDataset(Dataset):
    """Un-augmented lines from ground-truth field crops (or any RGB crops supplied by path)."""

    def __init__(self, rows: pd.DataFrame, variant: CrnnVariant, charset: TextCharset, crop_path_column: str = "field_crop_path"):
        self.rows = rows.reset_index(drop=True)
        self.variant, self.charset, self.crop_path_column = variant, charset, crop_path_column

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows.iloc[index]
        crop = read_rgb_image(row[self.crop_path_column])
        line_image = preprocess_line_image(crop, self.variant.input_height, self.variant.min_width, self.variant.max_width)
        return {"line_image": line_image, "label_indices": self.charset.encode(str(row.text)), "text": str(row.text),
                "field_name": row.field_name, "row_key": row.row_key}


def collate_line_batch(items: list[dict]) -> dict:
    """Pad lines (width rounded up to MPS_WIDTH_MULTIPLE) and flatten CTC targets."""
    line_batch, widths = pad_line_batch([item["line_image"] for item in items], MPS_WIDTH_MULTIPLE)
    targets = [label for item in items for label in item["label_indices"]]
    return {"line_images": torch.from_numpy(line_batch),
            "input_timesteps": torch.tensor([timesteps_for_width(width) for width in widths], dtype=torch.long),
            "targets": torch.tensor(targets, dtype=torch.long),
            "target_lengths": torch.tensor([len(item["label_indices"]) for item in items], dtype=torch.long),
            "texts": [item["text"] for item in items], "field_names": [item["field_name"] for item in items],
            "row_keys": [item["row_key"] for item in items]}


class WidthBucketedBatchSampler(Sampler):
    """Shuffle, sort each span of BUCKET_SPAN_BATCHES batches by aspect ratio, cut, shuffle batches."""

    def __init__(self, aspect_ratios: np.ndarray, batch_size: int, seed: int):
        self.aspect_ratios, self.batch_size, self.seed, self.epoch = aspect_ratios, batch_size, seed, 0

    def __len__(self) -> int:
        return (len(self.aspect_ratios) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        generator = np.random.default_rng((self.seed, self.epoch))
        self.epoch += 1
        order = generator.permutation(len(self.aspect_ratios))
        span = self.batch_size * BUCKET_SPAN_BATCHES
        batches = []
        for start in range(0, len(order), span):
            chunk = order[start:start + span]
            chunk = chunk[np.argsort(self.aspect_ratios[chunk], kind="stable")]
            batches += [chunk[i:i + self.batch_size].tolist() for i in range(0, len(chunk), self.batch_size)]
        for batch_index in generator.permutation(len(batches)):
            yield batches[batch_index]
