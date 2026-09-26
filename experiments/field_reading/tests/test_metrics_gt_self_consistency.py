"""Parsing the ground-truth text must reproduce the annotation's canonical value for >= 99.5% of rows."""

import pytest

from experiments.field_reading.config import SYNTH_V1_ROOT
from experiments.field_reading.data_access.field_manifest import load_field_rows
from experiments.field_reading.metrics.field_value_parsing import (
    FIELD_NAME_TO_CANONICAL_COLUMN,
    canonical_value_for_field,
    normalize_field_value,
)

MINIMUM_SELF_CONSISTENCY = 0.995

pytestmark = pytest.mark.skipif(not (SYNTH_V1_ROOT / "ocr").exists(), reason="synth v1 on vega not mounted")


@pytest.fixture(scope="module")
def val_field_rows():
    """The val manifest, loaded once for every field's check."""
    return load_field_rows("val")


@pytest.mark.parametrize("field_name", sorted(FIELD_NAME_TO_CANONICAL_COLUMN))
def test_val_ground_truth_text_parses_to_canonical_value(field_name: str, val_field_rows) -> None:
    """Val manifest: parsed GT text == canonical amount_cents / date_iso / check_number."""
    rows = val_field_rows[val_field_rows.field_name.eq(field_name)]
    canonical_column = FIELD_NAME_TO_CANONICAL_COLUMN[field_name]
    mismatches = [
        (text, canonical)
        for text, canonical in zip(rows.text, rows[canonical_column])
        if normalize_field_value(field_name, text) != canonical_value_for_field(field_name, canonical)
    ]
    agreement = 1 - len(mismatches) / len(rows)
    assert agreement >= MINIMUM_SELF_CONSISTENCY, f"{field_name}: {agreement:.4f}, e.g. {mismatches[:5]}"
