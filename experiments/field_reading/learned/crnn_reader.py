"""Load a trained CRNN checkpoint and read batches of RGB crops -> (text, confidences).

Checkpoints are dicts {variant_name, charset_name, state_dict, ...} written by `train_crnn`.
`read_crops` sorts by width internally so padding is small, then restores input order.
Optional `allowed_characters` applies constrained decoding (e.g. the amount vocabulary).
"""

import logging
from pathlib import Path

import numpy as np
import torch

from experiments.field_reading.learned.crnn_model import CRNN_VARIANT_REGISTRY, CrnnLineRecognizer, timesteps_for_width
from experiments.field_reading.learned.ctc_decoding import (CtcConfidenceKind, decode_ctc_batch,
                                                            restrict_log_probabilities_to_charset)
from experiments.field_reading.learned.line_image_preprocessing import (MPS_WIDTH_MULTIPLE, pad_line_batch,
                                                                       preprocess_line_image)
from experiments.field_reading.learned.text_charset import CHARSET_REGISTRY

logger = logging.getLogger(__name__)

READ_BATCH_SIZE = 64


class CrnnCropReader:
    """Inference wrapper around one CRNN checkpoint."""

    def __init__(self, checkpoint_path: Path, device: str, confidence_kind: CtcConfidenceKind = CtcConfidenceKind.MEAN_CHAR,
                 allowed_characters: str | None = None):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.variant = CRNN_VARIANT_REGISTRY[checkpoint["variant_name"]]
        self.charset = CHARSET_REGISTRY[checkpoint["charset_name"]]
        self.model = CrnnLineRecognizer(self.variant, self.charset.num_classes)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(device).eval()
        self.device, self.confidence_kind, self.allowed_characters = device, confidence_kind, allowed_characters
        logger.info("loaded CRNN %s (%s charset) from %s", checkpoint["variant_name"], self.charset.name, checkpoint_path)

    def preprocess(self, rgb_crop: np.ndarray) -> np.ndarray:
        """The model's exact input recipe for one crop."""
        return preprocess_line_image(rgb_crop, self.variant.input_height, self.variant.min_width, self.variant.max_width)

    @torch.no_grad()
    def log_probabilities_for_lines(self, line_images: list[np.ndarray]) -> tuple[np.ndarray, list[int]]:
        """(B, T, C) log-probs and valid timesteps for already-preprocessed lines."""
        width_multiple = MPS_WIDTH_MULTIPLE if self.device == "mps" else 1
        batch, widths = pad_line_batch(line_images, width_multiple)
        output = self.model(torch.from_numpy(batch).to(self.device)).float().cpu().numpy()
        return output, [timesteps_for_width(width) for width in widths]

    def read_crops(self, rgb_crops: list[np.ndarray]) -> list[tuple[str, dict[str, float]]]:
        """Read crops in any order; returns (text, confidences) aligned with the input."""
        line_images = [self.preprocess(crop) for crop in rgb_crops]
        order = np.argsort([image.shape[1] for image in line_images], kind="stable")
        results: list = [None] * len(line_images)
        for start in range(0, len(order), READ_BATCH_SIZE):
            batch_indices = order[start:start + READ_BATCH_SIZE]
            log_probabilities, timesteps = self.log_probabilities_for_lines([line_images[i] for i in batch_indices])
            if self.allowed_characters is not None:
                log_probabilities = restrict_log_probabilities_to_charset(log_probabilities, self.charset,
                                                                          self.allowed_characters)
            for index, decoded in zip(batch_indices, decode_ctc_batch(log_probabilities, timesteps, self.charset,
                                                                      self.confidence_kind)):
                results[index] = decoded
        return results
