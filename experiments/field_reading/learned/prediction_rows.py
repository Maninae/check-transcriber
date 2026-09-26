"""The reading-prediction row contract (PLAN.md) and its JSONL writer.

One row per scored target-field row:
`{row_key, scene_id, check_index, field_name, method, localization, pred_text, confidence,
pred_box, latency_ms}`. `pred_text == ""` means blank; `confidence` is in [0, 1] or null.
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from experiments.field_reading.config import PREDICTIONS_ROOT

logger = logging.getLogger(__name__)

PREDICTION_ROW_KEYS = ("row_key", "scene_id", "check_index", "field_name", "method", "localization",
                       "pred_text", "confidence", "pred_box", "latency_ms")


@dataclass
class ReadingPredictionRow:
    """One field's reading prediction, serialized exactly in contract order."""

    row_key: str
    scene_id: str
    check_index: int
    field_name: str
    method: str
    localization: str
    pred_text: str
    confidence: float | None
    pred_box: list[float] | None
    latency_ms: float | None

    def __post_init__(self) -> None:
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence out of [0, 1] for {self.row_key}: {self.confidence}")


def prediction_file_path(split_name: str, method_id: str, localization: str) -> Path:
    """`predictions/<split>/<method>__loc=<localization>.jsonl`."""
    return PREDICTIONS_ROOT / split_name / f"{method_id}__loc={localization}.jsonl"


def write_prediction_rows(rows: list[ReadingPredictionRow], output_path: Path) -> None:
    """Write rows as JSONL (atomically, via a temp file rename)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".jsonl.tmp")
    with open(temporary_path, "w") as handle:
        for row in rows:
            record = asdict(row)
            handle.write(json.dumps({key: record[key] for key in PREDICTION_ROW_KEYS}) + "\n")
    temporary_path.replace(output_path)
    logger.info("wrote %d prediction rows to %s", len(rows), output_path)
