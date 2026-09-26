"""Printed-vs-handwritten crop classifier: the non-oracle routing signal for the readers.

The app has no handwritten flag, and on real crops neither reader's confidence tells the styles
apart (the CRNN is confidently wrong on real handwriting). "Does this ink look handwritten" should
transfer from synth to real far better than reading synthetic fonts does, so a tiny CNN learns it
from synth train crops (same jitter/augmentation as the CRNN) and is checked on real crops.

- Model: CRNN-style conv trunk on the CRNN input (grey, height 32) + global average pool + logit.
- `train`: STYLE_TRAIN_ROWS random train rows, one pass, under the MPS lock.
- `predict`: p_handwritten for every scored row of a split/localization, plus the real SSBI and
  CAR crops (all handwritten), written to FIELD_READING_OUTPUT_ROOT/learned_style/
  (`<split>__loc=<loc>.jsonl`, `real_SSBI.jsonl`, `real_ORAND-CAR.jsonl`; rows {row_key, p_handwritten}).

Run: python -m experiments.field_reading.learned.handwriting_style_classifier train
     python -m experiments.field_reading.learned.handwriting_style_classifier predict --split val --localization oracle
"""

import argparse
import logging

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from experiments.field_reading.config import FIELD_READING_OUTPUT_ROOT, TARGET_FIELD_NAMES
from experiments.field_reading.learned.context_crop_export import CONTEXT_MANIFEST_PATH
from experiments.field_reading.learned.crnn_model import CRNN_VARIANT_REGISTRY, conv_bn_relu
from experiments.field_reading.learned.field_crop_sources import load_field_crops
from experiments.field_reading.learned.line_crop_dataset import ContextCropTrainingDataset, read_rgb_image
from experiments.field_reading.learned.line_image_preprocessing import MPS_WIDTH_MULTIPLE, pad_line_batch, preprocess_line_image
from experiments.field_reading.learned.mps_lock import hold_mps_lock
from experiments.field_reading.learned.predict import select_scored_rows
from experiments.field_reading.learned.reading_methods import RECOGNIZER_ROOT
from experiments.field_reading.learned.real_car_scoring import load_car_rows
from experiments.field_reading.learned.real_ssbi_scoring import load_ssbi_rows, pad_tight_crop
from experiments.field_reading.learned.text_charset import CHARSET_REGISTRY

logger = logging.getLogger(__name__)

STYLE_CHECKPOINT_PATH = RECOGNIZER_ROOT / "style_classifier_h32" / "model.pt"
STYLE_OUTPUT_ROOT = FIELD_READING_OUTPUT_ROOT / "learned_style"
STYLE_VARIANT = CRNN_VARIANT_REGISTRY["crnn_small_h32"]
STYLE_TRAIN_ROWS = 40000
STYLE_BATCH_SIZE = 64


class HandwritingStyleClassifier(nn.Module):
    """Grey line (B, 1, 32, W) -> logit of 'handwritten'; padding is masked out of the pooling."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(*conv_bn_relu(1, 32), nn.MaxPool2d(2, 2), *conv_bn_relu(32, 64), nn.MaxPool2d(2, 2),
                                      *conv_bn_relu(64, 96), nn.MaxPool2d((2, 1), (2, 1)), *conv_bn_relu(96, 128))
        self.classifier = nn.Linear(128, 1)

    def forward(self, line_images: torch.Tensor, valid_width_fraction: torch.Tensor) -> torch.Tensor:
        features = self.features(line_images).mean(dim=2)                     # (B, C, W')
        positions = torch.arange(features.shape[-1], device=features.device)[None] / features.shape[-1]
        mask = (positions < valid_width_fraction[:, None]).float()[:, None]     # ignore right padding
        pooled = (features * mask).sum(-1) / mask.sum(-1).clamp(min=1)
        return self.classifier(pooled).squeeze(-1)


class StyleTrainingDataset(Dataset):
    """(augmented grey line, handwritten label) pairs over a CRNN training dataset."""

    def __init__(self, line_dataset: ContextCropTrainingDataset):
        self.line_dataset = line_dataset
        self.labels = line_dataset.rows.handwritten.astype(float).to_numpy()

    def __len__(self) -> int:
        return len(self.line_dataset)

    def __getitem__(self, index: int) -> tuple[np.ndarray, float]:
        return self.line_dataset[index]["line_image"], float(self.labels[index])


def collate_style_batch(items: list[tuple[np.ndarray, float]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Padded images, valid width fractions, labels."""
    images, fractions = batch_from_lines([line for line, _ in items])
    return images, fractions, torch.tensor([label for _, label in items], dtype=torch.float32)


def batch_from_lines(lines: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    """Padded batch and each line's valid width fraction."""
    batch, widths = pad_line_batch(lines, MPS_WIDTH_MULTIPLE)
    return torch.from_numpy(batch), torch.tensor(widths, dtype=torch.float32) / batch.shape[-1]


def train_style_classifier(device: str) -> None:
    """One pass over a random subset of augmented train crops."""
    manifest = pd.read_json(CONTEXT_MANIFEST_PATH, lines=True)
    manifest = manifest[manifest.field_name.isin(TARGET_FIELD_NAMES)].sample(STYLE_TRAIN_ROWS, random_state=0)
    dataset = StyleTrainingDataset(ContextCropTrainingDataset(manifest, STYLE_VARIANT, CHARSET_REGISTRY["general"], seed=1))
    loader = DataLoader(dataset, batch_size=STYLE_BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=collate_style_batch)
    model = HandwritingStyleClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    for step, (images, fractions, labels) in enumerate(loader):
        loss = nn.functional.binary_cross_entropy_with_logits(model(images.to(device), fractions.to(device)), labels.to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            logger.info("step %d/%d loss %.4f", step, len(loader), loss.item())
    STYLE_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), STYLE_CHECKPOINT_PATH)


@torch.no_grad()
def predict_handwritten_probability(model: nn.Module, crops: list[np.ndarray], device: str) -> list[float]:
    """p_handwritten per RGB crop (width-sorted batches)."""
    lines = [preprocess_line_image(crop, STYLE_VARIANT.input_height, STYLE_VARIANT.min_width, STYLE_VARIANT.max_width) for crop in crops]
    order = np.argsort([line.shape[1] for line in lines], kind="stable")
    probabilities = np.zeros(len(lines))
    for start in range(0, len(order), 128):
        indices = order[start:start + 128]
        images, fractions = batch_from_lines([lines[i] for i in indices])
        probabilities[indices] = torch.sigmoid(model(images.to(device), fractions.to(device))).cpu().numpy()
    return probabilities.tolist()


def main() -> None:
    """train | predict."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["train", "predict"])
    parser.add_argument("--split", default="val")
    parser.add_argument("--localization", default="oracle")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    with hold_mps_lock(f"style classifier {arguments.action}"):
        if arguments.action == "train":
            train_style_classifier(device)
            return
        model = HandwritingStyleClassifier().to(device).eval()
        model.load_state_dict(torch.load(STYLE_CHECKPOINT_PATH, map_location=device))
        rows = select_scored_rows(arguments.split, None)
        crops, _ = load_field_crops(rows, arguments.split, arguments.localization)
        present = [crop is not None for crop in crops]
        probabilities = iter(predict_handwritten_probability(model, [c for c in crops if c is not None], device))
        rows["p_handwritten"] = [next(probabilities) if flag else None for flag in present]
        STYLE_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        rows[["row_key", "p_handwritten"]].to_json(STYLE_OUTPUT_ROOT / f"{arguments.split}__loc={arguments.localization}.jsonl",
                                                  orient="records", lines=True)
        scored = rows[rows.p_handwritten.notna()]
        logger.info("%s %s style accuracy vs GT flag: %.4f", arguments.split, arguments.localization,
                    ((scored.p_handwritten > 0.5) == scored.handwritten.astype(bool)).mean())
        if arguments.split == "val" and arguments.localization == "oracle":
            for name, real_rows in [("SSBI", load_ssbi_rows()), ("ORAND-CAR", load_car_rows())]:
                real_probabilities = predict_handwritten_probability(model, [pad_tight_crop(read_rgb_image(p)) for p in real_rows.crop_path], device)
                real_rows.assign(p_handwritten=real_probabilities)[["row_key", "p_handwritten"]].to_json(
                    STYLE_OUTPUT_ROOT / f"real_{name}.jsonl", orient="records", lines=True)
                logger.info("%s (all handwritten): %.3f called handwritten, median p %.3f", name,
                            np.mean(np.array(real_probabilities) > 0.5), np.median(real_probabilities))


if __name__ == "__main__":
    main()
