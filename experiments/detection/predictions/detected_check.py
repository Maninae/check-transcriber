"""The detector output contract shared by every detector, the metrics and the overlays.

Every detector returns `list[DetectedCheck]` per image, in FULL-RESOLUTION pixel
coordinates of the original scene image (detectors that work on a downscaled copy
must scale back up). A run's predictions for a split are saved as one JSON file:

    {"detector_name": str, "split_name": str, "config": {...},
     "predictions": {scene_id: [{"corners": [[x, y] x4], "score": float,
                                 "orientation_known": bool}, ...]}}

- `corners` are in clockwise image order. When `orientation_known` is True they must
  start at the check's own top-left (TL, TR, BR, BL of the printed check), so the
  metrics can score orientation. When False the starting corner is arbitrary and
  orientation is not scored (only the landscape axis is).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class DetectedCheck:
    """One predicted check quadrilateral."""

    corners: np.ndarray  # (4, 2) float, clockwise in image coordinates
    score: float = 1.0
    orientation_known: bool = False
    extras: dict = field(default_factory=dict)  # detector-specific diagnostics, not scored

    def to_json_dict(self) -> dict:
        """Serialize for the predictions file (extras are dropped)."""
        return {
            "corners": [[round(float(x), 3), round(float(y), 3)] for x, y in self.corners],
            "score": round(float(self.score), 5),
            "orientation_known": bool(self.orientation_known),
        }

    @classmethod
    def from_json_dict(cls, detection_json: dict) -> "DetectedCheck":
        """Inverse of `to_json_dict`."""
        return cls(
            corners=np.asarray(detection_json["corners"], dtype=np.float64),
            score=float(detection_json.get("score", 1.0)),
            orientation_known=bool(detection_json.get("orientation_known", False)),
        )


def save_predictions_file(
    output_json_path: Path,
    detector_name: str,
    split_name: str,
    predictions_by_scene_id: dict[str, list[DetectedCheck]],
    detector_config: dict | None = None,
) -> None:
    """Write a split's predictions to one JSON file."""
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "detector_name": detector_name,
        "split_name": split_name,
        "config": detector_config or {},
        "predictions": {
            scene_id: [detection.to_json_dict() for detection in detections]
            for scene_id, detections in predictions_by_scene_id.items()
        },
    }
    with open(output_json_path, "w") as output_file:
        json.dump(payload, output_file)


def load_predictions_file(predictions_json_path: Path) -> tuple[dict, dict[str, list[DetectedCheck]]]:
    """Read a predictions file; returns (header without predictions, predictions by scene id)."""
    with open(predictions_json_path) as predictions_file:
        payload = json.load(predictions_file)
    predictions_by_scene_id = {
        scene_id: [DetectedCheck.from_json_dict(item) for item in detections]
        for scene_id, detections in payload.pop("predictions").items()
    }
    return payload, predictions_by_scene_id
