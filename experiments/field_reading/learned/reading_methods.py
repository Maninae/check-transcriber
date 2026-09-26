"""Registry of reading methods: which reader reads which row, and how the answer is chosen.

A method maps (field rows, RGB crops) -> (text, confidence, latency_ms) per row. Readers are
loaded lazily and shared through `ReaderPool`, so a router that uses two models loads each once.

Routing rules:
- `_oraclehw` methods route on the ground-truth `handwritten` flag, which the app does NOT have;
  they are upper bounds, never shippable as is.
- non-oracle routers use only the field name and the readers' own confidences.
- latency_ms = the row's share of the batched wall time of every reader that read it, on the
  device used (the ONNX CPU numbers in onnx_benchmark are the browser-relevant ones).
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from experiments.field_reading.config import FIELD_READING_MODEL_ROOT
from experiments.field_reading.learned.crnn_reader import CrnnCropReader
from experiments.field_reading.learned.text_charset import AMOUNT_CHARSET
from experiments.field_reading.learned.trocr_reader import TrocrCropReader

logger = logging.getLogger(__name__)

RECOGNIZER_ROOT = FIELD_READING_MODEL_ROOT / "recognizers"
CRNN_CHECKPOINTS = {
    "crnn_general": RECOGNIZER_ROOT / "crnn_general_h32" / "best.pt",
    "crnn_amount": RECOGNIZER_ROOT / "crnn_amount_h32" / "best.pt",
}
AMOUNT_FIELD_NAME = "amount_numeric"


@dataclass
class ReaderOutput:
    """One reader's answers for a list of crops, aligned with the input, plus per-row latency."""

    texts: list[str]
    confidences: list[float]
    latency_ms_per_row: float


class ReaderPool:
    """Lazily constructed readers keyed by a short reader id (see `read_with`)."""

    def __init__(self, device: str):
        self.device = device
        self.readers: dict[str, object] = {}

    def get(self, reader_id: str):
        """Build or reuse a reader. Ids: crnn_general[_amountmask], crnn_amount, trocr ids [+ `:grey`]."""
        if reader_id not in self.readers:
            if reader_id.startswith("crnn_"):
                base_id = reader_id.removesuffix("_amountmask")
                allowed = AMOUNT_CHARSET if reader_id.endswith("_amountmask") else None
                self.readers[reader_id] = CrnnCropReader(CRNN_CHECKPOINTS[base_id], self.device, allowed_characters=allowed)
            else:
                model_id, _, option = reader_id.partition(":")
                self.readers[reader_id] = TrocrCropReader(model_id, self.device, grey_input=option == "grey")
        return self.readers[reader_id]

    def read_with(self, reader_id: str, crops: list[np.ndarray]) -> ReaderOutput:
        """Run one reader on `crops`; CRNN readers return their configured CTC confidence."""
        if not crops:
            return ReaderOutput([], [], 0.0)
        started = time.perf_counter()
        results = self.get(reader_id).read_crops(crops)
        elapsed_ms = (time.perf_counter() - started) * 1000
        texts = [text for text, _ in results]
        confidences = [extra["confidence"] if isinstance(extra, dict) else extra for _, extra in results]
        return ReaderOutput(texts, [float(np.clip(value, 0.0, 1.0)) for value in confidences], elapsed_ms / len(crops))


MethodRunner = Callable[[pd.DataFrame, list[np.ndarray], ReaderPool], list[tuple[str, float, float]]]


def read_subset(rows: pd.DataFrame, crops: list[np.ndarray], mask: np.ndarray, reader_id: str, pool: ReaderPool,
                results: list) -> None:
    """Read rows where `mask` is true with one reader, writing (text, confidence, latency) into `results`."""
    indices = np.flatnonzero(mask)
    output = pool.read_with(reader_id, [crops[i] for i in indices])
    for position, index in enumerate(indices):
        results[index] = (output.texts[position], output.confidences[position], output.latency_ms_per_row)


def single_reader(reader_id: str) -> MethodRunner:
    """Every row read by one reader."""
    def run(rows, crops, pool):
        results: list = [None] * len(rows)
        read_subset(rows, crops, np.ones(len(rows), bool), reader_id, pool, results)
        return results
    return run


def route_by_mask(mask_function: Callable[[pd.DataFrame], np.ndarray], reader_when_true: str, reader_when_false: str) -> MethodRunner:
    """Rows where `mask_function(rows)` holds go to one reader, the rest to the other."""
    def run(rows, crops, pool):
        results: list = [None] * len(rows)
        mask = mask_function(rows)
        read_subset(rows, crops, mask, reader_when_true, pool, results)
        read_subset(rows, crops, ~mask, reader_when_false, pool, results)
        return results
    return run


def higher_confidence(reader_a: str, reader_b: str) -> MethodRunner:
    """Both readers read every row; keep the answer with the higher confidence (latency = sum)."""
    def run(rows, crops, pool):
        all_rows = np.ones(len(rows), bool)
        results_a: list = [None] * len(rows)
        results_b: list = [None] * len(rows)
        read_subset(rows, crops, all_rows, reader_a, pool, results_a)
        read_subset(rows, crops, all_rows, reader_b, pool, results_b)
        return [(a if a[1] >= b[1] else b)[:2] + (a[2] + b[2],) for a, b in zip(results_a, results_b)]
    return run


def is_handwritten(rows: pd.DataFrame) -> np.ndarray:
    """ORACLE: the ground-truth handwritten flag."""
    return rows.handwritten.fillna(False).to_numpy(bool)


def is_amount_field(rows: pd.DataFrame) -> np.ndarray:
    """Courtesy-amount rows (a field-name route, available to the app)."""
    return (rows.field_name == AMOUNT_FIELD_NAME).to_numpy()


READING_METHOD_REGISTRY: dict[str, MethodRunner] = {
    "trocr_small_printed_zs": single_reader("trocr_small_printed"),
    "trocr_small_handwritten_zs": single_reader("trocr_small_handwritten"),
    "trocr_small_handwritten_zs_grey": single_reader("trocr_small_handwritten:grey"),
    "trocr_small_zs_route_oraclehw": route_by_mask(is_handwritten, "trocr_small_handwritten:grey", "trocr_small_printed"),
    "trocr_small_zs_maxconf": higher_confidence("trocr_small_handwritten:grey", "trocr_small_printed"),
    "crnn_general": single_reader("crnn_general"),
    "crnn_general_amountmask": route_by_mask(is_amount_field, "crnn_general_amountmask", "crnn_general"),
    "crnn_amount_route": route_by_mask(is_amount_field, "crnn_amount", "crnn_general"),
    "trocr_small_hw_ft": single_reader("trocr_small_handwritten_ft"),
    "trocr_ft_crnn_route_oraclehw": route_by_mask(is_handwritten, "trocr_small_handwritten_ft", "crnn_general"),
    "trocr_ft_crnn_maxconf": higher_confidence("trocr_small_handwritten_ft", "crnn_general"),
}
