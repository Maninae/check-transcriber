"""Per-check ground-truth view for localization: one record per rectified check crop.

`load_localization_checks(split)` returns a list of dicts:
    {scene_id, check_index, check_crop_path, check_crop_size (w, h), layout_family, size_kind, template_id,
     fields: {field_name: {"box": [..] | None, "status": str, "handwritten": bool}}}
A target field missing from `fields` is genuinely absent on the check (e.g. no memo written).
"""

import logging

from experiments.field_reading.data_access.field_manifest import load_field_rows

logger = logging.getLogger(__name__)


def load_localization_checks(split_name: str) -> list[dict]:
    """Group the split's target-field rows into one record per check crop, sorted by (scene, check)."""
    field_rows = load_field_rows(split_name, target_fields_only=True)
    check_records: dict[tuple[str, int], dict] = {}
    for row in field_rows.itertuples(index=False):
        key = (row.scene_id, int(row.check_index))
        if key not in check_records:
            check_records[key] = {
                "scene_id": row.scene_id, "check_index": int(row.check_index),
                "check_crop_path": row.check_crop_path, "check_crop_size": tuple(row.check_crop_size),
                "layout_family": row.layout_family, "size_kind": row.size_kind, "template_id": row.template_id,
                "fields": {},
            }
        box = row.box_in_check_crop if isinstance(row.box_in_check_crop, list) else None
        check_records[key]["fields"][row.field_name] = {"box": box, "status": row.status,
                                                        "handwritten": bool(row.handwritten)}
    logger.info("loaded %d %s checks for localization", len(check_records), split_name)
    return [check_records[key] for key in sorted(check_records)]
