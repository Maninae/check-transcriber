"""TrOCR (VisionEncoderDecoder) reading with a sequence confidence, zero-shot or fine-tuned.

Confidence = exp(mean token log-prob) over the greedy-decoded tokens including EOS
(`compute_transition_scores` with normalized logits; pad positions after EOS are excluded).
Greedy decoding (num_beams=1) keeps latency and the confidence definition simple.

Model ids resolve through TROCR_MODEL_REGISTRY (hub checkpoints cached under HF_HOME on vega) or
any local checkpoint directory (e.g. a finetune_trocr output).
Zero-shot trocr-small-handwritten needs `grey_input=True` on real crops: with RGB input it
hallucinates on SSBI (amount exact 0.09 vs 0.82 grey).
"""

import logging
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from PIL import Image, ImageOps
from transformers import VisionEncoderDecoderModel

from experiments.field_reading.config import FIELD_READING_MODEL_ROOT
from experiments.field_reading.learned.trocr_tokenizer import SENTENCEPIECE_FILE_NAME, XlmRobertaSentencePieceCodec

logger = logging.getLogger(__name__)

TROCR_MODEL_REGISTRY: dict[str, str] = {
    "trocr_small_handwritten": "microsoft/trocr-small-handwritten",
    "trocr_small_printed": "microsoft/trocr-small-printed",
}
TROCR_READ_BATCH_SIZE = 24
TROCR_MAX_NEW_TOKENS = 48
TROCR_INPUT_SIZE = 384   # DeiT image processor: bicubic resize to 384 x 384, mean = std = 0.5


def resolve_trocr_directory(model_id: str) -> Path:
    """Local directory of a registered TrOCR checkpoint (hub snapshots come from the vega cache)."""
    source = TROCR_MODEL_REGISTRY.get(model_id, model_id)
    return Path(source) if Path(source).exists() else Path(snapshot_download(source))


GREY_AUTOCONTRAST_CUTOFF_PERCENT = 2


def grey_autocontrast(rgb_crop: np.ndarray) -> Image.Image:
    """Grey, contrast-stretched copy (closer to IAM's dark-ink-on-white lines), back as RGB."""
    grey = ImageOps.autocontrast(Image.fromarray(rgb_crop).convert("L"), cutoff=GREY_AUTOCONTRAST_CUTOFF_PERCENT)
    return grey.convert("RGB")


def trocr_pixel_values(rgb_crops: list[np.ndarray], grey_input: bool = False) -> torch.Tensor:
    """(B, 3, 384, 384) float input, identical to the checkpoints' DeiTImageProcessor.

    `grey_input` first applies `grey_autocontrast` (an optional zero-shot domain tweak).
    """
    images = [grey_autocontrast(crop) if grey_input else Image.fromarray(crop) for crop in rgb_crops]
    resized = [np.asarray(image.resize((TROCR_INPUT_SIZE, TROCR_INPUT_SIZE), Image.BICUBIC), dtype=np.float32)
               for image in images]
    batch = (np.stack(resized) / 255.0 - 0.5) / 0.5
    return torch.from_numpy(batch.transpose(0, 3, 1, 2).copy())


class TrocrCropReader:
    """Batch reader for one TrOCR checkpoint."""

    def __init__(self, model_id: str, device: str, grey_input: bool = False):
        model_directory = resolve_trocr_directory(model_id)
        self.codec = XlmRobertaSentencePieceCodec(model_directory / SENTENCEPIECE_FILE_NAME)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_directory).to(device).eval()
        self.device, self.model_id, self.grey_input = device, model_id, grey_input
        logger.info("loaded TrOCR %s from %s on %s", model_id, model_directory, device)

    @torch.no_grad()
    def read_crops(self, rgb_crops: list[np.ndarray]) -> list[tuple[str, float]]:
        """(text, confidence) per crop, in input order."""
        results: list[tuple[str, float]] = []
        pad_token_id = self.codec.pad_token_id
        for start in range(0, len(rgb_crops), TROCR_READ_BATCH_SIZE):
            pixel_values = trocr_pixel_values(rgb_crops[start:start + TROCR_READ_BATCH_SIZE], self.grey_input).to(self.device)
            output = self.model.generate(pixel_values, max_new_tokens=TROCR_MAX_NEW_TOKENS, num_beams=1, do_sample=False,
                                         output_scores=True, return_dict_in_generate=True)
            token_log_probabilities = self.model.compute_transition_scores(output.sequences, output.scores,
                                                                           normalize_logits=True).float().cpu().numpy()
            generated_tokens = output.sequences[:, 1:].cpu().numpy()
            texts = self.codec.batch_decode(output.sequences.cpu().tolist())
            for text, token_ids, log_probabilities in zip(texts, generated_tokens, token_log_probabilities):
                valid = (token_ids != pad_token_id) & np.isfinite(log_probabilities)
                confidence = float(np.exp(log_probabilities[valid].mean())) if valid.any() else 0.0
                results.append((text.strip(), confidence))
        return results
