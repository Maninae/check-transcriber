"""Write and read localization prediction JSONL files in the U2 contract shape.

Row: {row_key, scene_id, check_index, field_name, pred_box (list | null), confidence}, one per
(check, target field), present on the check or not.
"""

import json
from pathlib import Path

from experiments.field_reading.data_access.field_manifest import make_row_key


def make_prediction_row(scene_id: str, check_index: int, field_name: str, predicted_box: list[float] | None,
                        confidence: float | None) -> dict:
    """One contract row; boxes rounded to 0.1 px, confidence to 4 decimals."""
    return {
        "row_key": make_row_key(scene_id, check_index, field_name),
        "scene_id": scene_id, "check_index": int(check_index), "field_name": field_name,
        "pred_box": [round(float(value), 1) for value in predicted_box] if predicted_box is not None else None,
        "confidence": round(float(confidence), 4) if confidence is not None else None,
    }


def write_prediction_rows(prediction_rows: list[dict], output_path: Path) -> None:
    """Write rows as JSONL (parent directories created)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as output_file:
        for prediction_row in prediction_rows:
            output_file.write(json.dumps(prediction_row) + "\n")


def read_prediction_rows_by_key(prediction_path: Path) -> dict[str, dict]:
    """row_key -> prediction row."""
    with open(prediction_path) as prediction_file:
        return {row["row_key"]: row for row in map(json.loads, prediction_file)}
