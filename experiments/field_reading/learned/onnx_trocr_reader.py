"""TrOCR reader backed by onnxruntime (CPU): the browser path, for accuracy checks of ONNX/int8.

Uses the same preprocessing and confidence definition as TrocrCropReader (grey+autocontrast
optional; confidence = exp(mean chosen-token log-prob), EOS included), one crop at a time with
greedy decoding (`trocr_onnx_decoding`). Variants come from trocr_onnx_benchmark.variant_paths().
"""

import logging

import numpy as np

from experiments.field_reading.learned.onnx_benchmark import ort_session
from experiments.field_reading.learned.trocr_onnx_benchmark import variant_paths
from experiments.field_reading.learned.trocr_onnx_decoding import onnx_trocr_greedy
from experiments.field_reading.learned.trocr_reader import resolve_trocr_directory, trocr_pixel_values
from experiments.field_reading.learned.trocr_tokenizer import SENTENCEPIECE_FILE_NAME, XlmRobertaSentencePieceCodec

logger = logging.getLogger(__name__)

ACCURACY_RUN_THREADS = 2   # shared machine: accuracy runs stay light; latency is measured separately


class OnnxTrocrCropReader:
    """read_crops() over one ONNX (encoder, decoder) pair of trocr-small-handwritten."""

    def __init__(self, variant_label: str, grey_input: bool, threads: int | None = ACCURACY_RUN_THREADS):
        encoder_path, decoder_path = variant_paths()[variant_label]
        self.encoder, self.decoder = ort_session(encoder_path, threads), ort_session(decoder_path, threads)
        self.codec = XlmRobertaSentencePieceCodec(resolve_trocr_directory("trocr_small_handwritten") / SENTENCEPIECE_FILE_NAME)
        self.grey_input = grey_input
        logger.info("loaded ONNX TrOCR %s (grey=%s)", variant_label, grey_input)

    def read_crops(self, rgb_crops: list[np.ndarray]) -> list[tuple[str, float]]:
        """(text, confidence) per crop, in order."""
        results = []
        for crop in rgb_crops:
            tokens, _, _, confidence = onnx_trocr_greedy(self.encoder, self.decoder, trocr_pixel_values([crop], self.grey_input).numpy())
            results.append((self.codec.decode(tokens), confidence))
        return results
