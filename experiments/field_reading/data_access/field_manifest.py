"""Load per-field OCR rows for a split, joined with each check's canonical values.

One row per (scene, check, field) with the synth v1 OCR manifest's columns plus:
- `canonical_amount_cents`, `canonical_date_iso`, `canonical_check_number`: parsed ground truth,
  so money/date/digit metrics compare values rather than strings.
- `check_crop_path` / `field_crop_path`: absolute paths.
- `row_key`: "<scene_id>__check=<i>__field=<name>", the join key for every prediction file.

val/eval come from synth v1's own `ocr/` export; train comes from our re-cut of the train
scenes (`data_access/train_crop_export.py`), which writes the identical schema.
"""

import json
import logging
from functools import lru_cache
from pathlib import Path

import pandas as pd

from experiments.field_reading.config import SYNTH_V1_ROOT, TARGET_FIELD_NAMES, TRAIN_CROPS_ROOT

logger = logging.getLogger(__name__)

SPLITS_WITH_SYNTH_OCR_EXPORT = ("val", "eval")


def dataset_root_for_split(split_name: str) -> Path:
    """Root that the manifest's relative crop paths hang off."""
    return SYNTH_V1_ROOT if split_name in SPLITS_WITH_SYNTH_OCR_EXPORT else TRAIN_CROPS_ROOT


def make_row_key(scene_id: str, check_index: int, field_name: str) -> str:
    """Stable join key between ground-truth rows and predictions."""
    return f"{scene_id}__check={check_index}__field={field_name}"


@lru_cache(maxsize=4)
def load_canonical_table(split_name: str) -> pd.DataFrame:
    """One row per (scene_id, check_index) with the check's canonical values from the scene annotations."""
    records = []
    for annotation_path in sorted((SYNTH_V1_ROOT / split_name / "annotations").glob("*.json")):
        scene_label = json.loads(annotation_path.read_text())
        for check in scene_label["checks"]:
            canonical = check["canonical"]
            records.append({
                "scene_id": scene_label["scene_id"],
                "check_index": check["check_index"],
                "canonical_amount_cents": canonical.get("amount_cents"),
                "canonical_date_iso": canonical.get("date_iso"),
                "canonical_check_number": canonical.get("check_number"),
                "canonical_payee": canonical.get("payee_canonical"),
                "field_names_present": [field["field_name"] for field in check["fields"]],
            })
    logger.info("loaded canonical values for %d %s checks", len(records), split_name)
    return pd.DataFrame.from_records(records)


def load_field_rows(split_name: str, target_fields_only: bool = True) -> pd.DataFrame:
    """Every field row of a split (all statuses), with canonical values and absolute paths joined in.

    Args:
        split_name: "train", "val" or "eval".
        target_fields_only: keep only the fields the app reads (config.TARGET_FIELD_NAMES).
    """
    root = dataset_root_for_split(split_name)
    manifest_path = root / "ocr" / f"ocr_fields__split={split_name}.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"no OCR manifest for split {split_name}: {manifest_path}")
    rows = pd.read_json(manifest_path, lines=True)
    if target_fields_only:
        rows = rows[rows.field_name.isin(TARGET_FIELD_NAMES)].copy()
    rows = rows.merge(load_canonical_table(split_name).drop(columns=["field_names_present"]),
                      on=["scene_id", "check_index"], how="left", validate="many_to_one")
    rows["check_crop_path"] = rows.check_crop.map(lambda relative: str(root / relative))
    rows["field_crop_path"] = rows.field_crop.map(lambda relative: str(root / relative) if isinstance(relative, str) else None)
    rows["row_key"] = [make_row_key(s, c, f) for s, c, f in zip(rows.scene_id, rows.check_index, rows.field_name)]
    return rows.reset_index(drop=True)


def load_check_rows(split_name: str) -> pd.DataFrame:
    """One row per rectified check crop: path, size, family, and which target fields it carries (for localization)."""
    field_rows = load_field_rows(split_name, target_fields_only=True)
    grouped = field_rows.groupby(["scene_id", "check_index"], sort=True)
    return grouped.agg(check_crop_path=("check_crop_path", "first"), check_crop_size=("check_crop_size", "first"),
                       layout_family=("layout_family", "first"), template_id=("template_id", "first"),
                       size_kind=("size_kind", "first")).reset_index()
