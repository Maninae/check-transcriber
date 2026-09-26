"""Per-check and per-scene score records, plus the constants every metrics module shares.

The records are plain dataclasses so they serialize straight into `metrics.json`
(`dataclasses.asdict`) and can be mined later for failures. Leaf module: no imports
from the rest of the package, so geometry, matching, scoring and reporting all
depend on it without cycles.
"""

from dataclasses import dataclass, field

# IoU thresholds the precision/recall/F1 table is reported at.
MATCHING_IOU_THRESHOLDS: tuple[float, ...] = (0.5, 0.75, 0.9)
# The matching that per-check localization stats (corner error, orientation) come from.
PRIMARY_MATCHING_IOU_THRESHOLD = 0.5
# Loose matching kept only so near-misses still get a per-check record to inspect.
DIAGNOSTIC_MATCHING_IOU_THRESHOLD = 0.1
# "Corner error below N px" fractions reported in the headline.
CORNER_ERROR_PIXEL_CUTOFFS: tuple[float, ...] = (2.0, 5.0, 10.0)
# A check is "overlapped" when less than this fraction of it is visible.
OVERLAPPED_VISIBLE_FRACTION_THRESHOLD = 0.98


def threshold_key(iou_threshold: float) -> str:
    """Stable dict key for an IoU threshold, e.g. 0.75 -> "0.75"."""
    return f"{iou_threshold:.2f}"


@dataclass
class CheckScoreRecord:
    """Everything scored about one ground-truth check.

    Localization fields are filled from the diagnostic (IoU >= 0.1) match so near-misses
    can be inspected; aggregates only use them when `matched_iou_threshold_keys`
    contains the primary 0.5 key.
    """

    scene_id: str
    check_index: int
    # Check-level attributes used by the breakdowns.
    size_kind: str
    orientation_class: int
    deformation_kinds: list[str]
    fully_in_frame: bool
    overlapped: bool
    visible_fraction: float
    # Matching outcome. Keys are `threshold_key(t)` for every threshold this check matched at.
    matched_iou_threshold_keys: list[str] = field(default_factory=list)
    matched_prediction_index: int | None = None  # index into the scene's kept predictions
    iou_with_outline: float | None = None
    iou_with_corner_quad: float | None = None
    # Corner error at the best cyclic shift of the predicted corner order.
    best_cyclic_shift: int | None = None  # 0 = predicted order equals the GT order
    corner_error_mean_px: float | None = None
    corner_error_max_px: float | None = None
    corner_error_mean_percent_short_side: float | None = None
    corners_used_for_error: int = 0  # GT corners inside the image
    prediction_orientation_known: bool | None = None
    orientation_correct: bool | None = None  # only when orientation_known
    axis_correct: bool | None = None
    prediction_clockwise: bool | None = None  # contract says clockwise; False flags a bug


@dataclass
class SceneScoreRecord:
    """Everything scored about one scene."""

    scene_id: str
    # Scene-level attributes used by the breakdowns.
    background_id: str
    background_category: str
    background_source: str
    layout_mode: str
    cast_shadow_kind: str
    check_count_bucket: str
    ground_truth_count: int
    predicted_count: int  # predictions kept after the score threshold
    count_exact: bool
    # threshold_key -> number of matched checks / unmatched predictions at that threshold.
    true_positives_by_threshold: dict[str, int] = field(default_factory=dict)
    false_positives_by_threshold: dict[str, int] = field(default_factory=dict)
    scene_perfect: bool = False  # every GT matched at 0.5 and no false positives
    had_predictions_entry: bool = True  # False when the predictions file lacked this scene
