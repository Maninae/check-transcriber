"""Typed view of the v1 synthetic scene annotations, the ground truth for detection.

Each scene JSON (`<split>/annotations/<scene_id>.json`) describes one phone photo
holding 1-12 checks. We keep only what detection needs:

- `corners`: the check's four physical corners in its OWN order (top-left, top-right,
  bottom-right, bottom-left of the check as printed), in full-resolution image pixels.
  Corners may lie outside the image when the check is partly out of frame.
- `outline`: the dense deformed silhouette (folds, curl, waves bend the edges), the
  polygon IoU is scored against.
- `orientation_class`: rotation clockwise snapped to 0/90/180/270; the corner order
  already encodes it, this is the convenience label.

Labels were measured exact to ~0.1 px, so they are treated as ground truth.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from experiments.detection.config.paths import (
    split_annotations_directory,
    split_images_directory,
)

# Keyword -> coarse surface category, first match wins. Used only for metric breakdowns.
BACKGROUND_KEYWORD_TO_CATEGORY: list[tuple[str, str]] = [
    ("rug", "carpet_rug"),
    ("carpet", "carpet_rug"),
    ("towel", "carpet_rug"),
    ("bedsheet", "bedding"),
    ("bed", "bedding"),
    ("duvet", "bedding"),
    ("comforter", "bedding"),
    ("blanket", "fabric_pattern"),
    ("crochet", "fabric_pattern"),
    ("tartan", "fabric_pattern"),
    ("gingham", "fabric_pattern"),
    ("jacquard", "fabric_pattern"),
    ("checkered", "fabric_pattern"),
    ("diamond", "fabric_pattern"),
    ("fabric", "fabric_plain"),
    ("cloth", "fabric_plain"),
    ("chino", "fabric_plain"),
    ("boucle", "fabric_plain"),
    ("cardboard", "cardboard"),
    ("marble", "stone_tile"),
    ("granite", "stone_tile"),
    ("terrazzo", "stone_tile"),
    ("slate", "stone_tile"),
    ("tile", "stone_tile"),
    ("stone", "stone_tile"),
    ("speckled", "stone_tile"),
    ("brick", "stone_tile"),
    ("linoleum", "stone_tile"),
    ("wood", "wood"),
    ("veneer", "wood"),
    ("plank", "wood"),
    ("laminate", "wood"),
    ("chipboard", "wood"),
    ("strand", "wood"),
    ("pine", "wood"),
    ("oak", "wood"),
    ("cork", "wood"),
]


def categorize_background_id(background_id: str) -> str:
    """Map a background file id to a coarse surface category ("wood", "bedding", ...)."""
    lowered_background_id = background_id.lower()
    for keyword, category in BACKGROUND_KEYWORD_TO_CATEGORY:
        if keyword in lowered_background_id:
            return category
    return "other"


@dataclass
class CheckAnnotation:
    """Ground truth for one check in a scene."""

    check_index: int
    corners: np.ndarray  # (4, 2) float, TL/TR/BR/BL of the check itself, image pixels
    outline: np.ndarray  # (N, 2) float, deformed silhouette, may leave the frame
    orientation_class: int  # 0, 90, 180 or 270 (clockwise)
    rotation_degrees_clockwise: float
    fully_in_frame: bool
    in_frame_fraction: float
    visible_fraction: float  # < 1 when another check overlaps it
    size_kind: str  # personal / business / money_order
    deformation_kinds: tuple[str, ...]  # subset of ("corner_lift", "curl", "fold", "waves")


@dataclass
class SceneAnnotation:
    """Ground truth for one scene (one phone photo)."""

    scene_id: str
    split_name: str
    image_path: Path
    image_width: int
    image_height: int
    background_id: str
    background_category: str
    background_source: str  # "web" (real photo texture) or "flux" (generated)
    layout_mode: str  # grid / loose_overlap / loose_fan
    cast_shadow_kind: str  # "none", "hand", "phone_and_hand"
    framing_regime: str = "wide"  # "wide" (v1 default), "close" (4-6 checks fill the frame), "single"
    checks: list[CheckAnnotation] = field(default_factory=list)


def parse_check_annotation(check_json: dict) -> CheckAnnotation:
    """Build a CheckAnnotation from one entry of a scene JSON's `checks` list."""
    deformation_json = check_json.get("deformation") or {}
    deformation_kinds = tuple(
        sorted(
            kind
            for kind, value in deformation_json.items()
            if value and kind not in ("width_inches", "height_inches")
        )
    )
    return CheckAnnotation(
        check_index=int(check_json["check_index"]),
        corners=np.asarray(check_json["corners"], dtype=np.float64),
        outline=np.asarray(check_json["outline"], dtype=np.float64),
        orientation_class=int(check_json["orientation_class"]),
        rotation_degrees_clockwise=float(check_json["rotation_degrees_clockwise"]),
        fully_in_frame=bool(check_json["fully_in_frame"]),
        in_frame_fraction=float(check_json["in_frame_fraction"]),
        visible_fraction=float(check_json["visible_fraction"]),
        size_kind=str(check_json["size_kind"]),
        deformation_kinds=deformation_kinds,
    )


def parse_scene_annotation(scene_json: dict, split_name: str) -> SceneAnnotation:
    """Build a SceneAnnotation from a parsed scene JSON dict."""
    background_id = scene_json["background_id"]
    effects_json = scene_json.get("effects") or {}
    cast_shadow_json = effects_json.get("cast_shadow") or {}
    framing_json = effects_json.get("framing") or {}
    return SceneAnnotation(
        scene_id=scene_json["scene_id"],
        split_name=split_name,
        image_path=split_images_directory(split_name) / scene_json["image_file"],
        image_width=int(scene_json["image_width"]),
        image_height=int(scene_json["image_height"]),
        background_id=background_id,
        background_category=categorize_background_id(background_id),
        background_source=background_id.split("/", 1)[0],
        layout_mode=scene_json["layout_mode"],
        cast_shadow_kind=cast_shadow_json.get("kind") or "none",
        framing_regime=framing_json.get("framing_regime") or "wide",
        checks=[parse_check_annotation(check_json) for check_json in scene_json["checks"]],
    )


def load_scene_annotation(annotation_json_path: Path, split_name: str) -> SceneAnnotation:
    """Read one scene annotation file."""
    with open(annotation_json_path) as annotation_file:
        return parse_scene_annotation(json.load(annotation_file), split_name)


def load_split_scene_annotations(split_name: str, limit: int | None = None) -> list[SceneAnnotation]:
    """Load every scene of a split, sorted by scene id; `limit` keeps the first N."""
    annotation_paths = sorted(split_annotations_directory(split_name).glob("*.json"))
    if not annotation_paths:
        raise FileNotFoundError(f"no annotations for split {split_name!r}")
    if limit is not None:
        annotation_paths = annotation_paths[:limit]
    return [load_scene_annotation(path, split_name) for path in annotation_paths]
