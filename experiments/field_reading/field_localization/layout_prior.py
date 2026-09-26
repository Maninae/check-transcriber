"""Per-field layout priors estimated from train boxes, grouped by layout family or by size kind.

A prior for (group, field) is a set of normalized ([0, 1] of the crop) statistics:
- `median_box`: the typical box; its centre is the refinement's expected text position.
- `search_region`: the low/high quantile envelope of all train boxes, where refinement looks for ink.
- `median_height`: typical text-box height, which sets the refinement's ink-size filters.
- `presence_rate`: fraction of the group's checks that carry the field.

Grouping by `layout_family` is the oracle-family upper bound; by `size_kind` is what the app can infer
from the crop aspect alone.
"""

import json
import logging
from pathlib import Path

import numpy as np

from experiments.field_reading.config import TARGET_FIELD_NAMES
from experiments.field_reading.field_localization.box_geometry import normalize_box
from experiments.field_reading.field_localization.localization_config import BOX_TARGET_STATUSES

logger = logging.getLogger(__name__)

SEARCH_REGION_LOW_QUANTILE = 0.005
SEARCH_REGION_HIGH_QUANTILE = 0.995


def estimate_layout_priors(check_records: list[dict], group_column: str) -> dict[str, dict[str, dict]]:
    """group value -> field name -> prior statistics (see module docstring).

    Args:
        check_records: output of `localization_targets.load_localization_checks` (train split).
        group_column: "layout_family" or "size_kind".
    """
    normalized_boxes_by_group_field: dict[tuple[str, str], list[list[float]]] = {}
    check_count_by_group: dict[str, int] = {}
    for check_record in check_records:
        group_value = check_record[group_column]
        check_count_by_group[group_value] = check_count_by_group.get(group_value, 0) + 1
        crop_width, crop_height = check_record["check_crop_size"]
        for field_name, field_target in check_record["fields"].items():
            if field_target["box"] is None or field_target["status"] not in BOX_TARGET_STATUSES:
                continue
            normalized_boxes_by_group_field.setdefault((group_value, field_name), []).append(
                normalize_box(field_target["box"], crop_width, crop_height))
    layout_priors: dict[str, dict[str, dict]] = {}
    for (group_value, field_name), normalized_boxes in sorted(normalized_boxes_by_group_field.items()):
        box_array = np.asarray(normalized_boxes)
        low_corner = np.quantile(box_array[:, :2], SEARCH_REGION_LOW_QUANTILE, axis=0)
        high_corner = np.quantile(box_array[:, 2:], SEARCH_REGION_HIGH_QUANTILE, axis=0)
        layout_priors.setdefault(group_value, {})[field_name] = {
            "median_box": np.median(box_array, axis=0).tolist(),
            "search_region": [*low_corner.tolist(), *high_corner.tolist()],
            "median_height": float(np.median(box_array[:, 3] - box_array[:, 1])),
            "presence_rate": len(normalized_boxes) / check_count_by_group[group_value],
            "box_count": len(normalized_boxes),
        }
    for group_value, field_priors in layout_priors.items():
        missing_fields = set(TARGET_FIELD_NAMES) - set(field_priors)
        if missing_fields:
            logger.warning("group %s has no train boxes for %s", group_value, sorted(missing_fields))
    return layout_priors


def save_layout_priors(priors_by_group_column: dict[str, dict], output_path: Path) -> None:
    """Write {group_column: priors} as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(priors_by_group_column, indent=1))


def load_layout_priors(prior_path: Path) -> dict[str, dict]:
    """Inverse of `save_layout_priors`."""
    return json.loads(prior_path.read_text())
