"""Hard-set slice definitions on hand-built rows (no crops, so low_contrast is never set)."""

import pandas as pd

from experiments.field_reading.metrics.hard_set import build_hard_set_flags


def test_slices_follow_their_definitions() -> None:
    """handwritten_degraded / printed_degraded / in_hard_set per the module docstring; non-ok rows are never hard."""
    rows = pd.DataFrame({
        "row_key": ["hw_clean", "hw_small", "pr_clean", "pr_small", "pr_money_order", "hw_money_order", "tiny"],
        "field_name": ["payee"] * 7,
        "status": ["ok"] * 6 + ["too_small"],
        "field_crop_path": [None] * 7,
        "handwritten": [True, True, False, False, False, True, True],
        "text_height_in_photo_px": [30.0, 15.0, 30.0, 15.0, 30.0, 30.0, 10.0],
        "layout_family": ["personal_classic"] * 4 + ["money_order", "money_order", "personal_classic"],
    })
    flags = build_hard_set_flags(rows, worker_count=1).set_index("row_key")
    assert flags.handwritten_degraded[flags.handwritten_degraded].index.tolist() == ["hw_small"]
    assert flags.printed_degraded[flags.printed_degraded].index.tolist() == ["pr_small", "pr_money_order"]
    assert flags.in_hard_set[flags.in_hard_set].index.tolist() == ["hw_small", "pr_small", "pr_money_order"]
    assert flags.too_small["tiny"] and not flags.in_hard_set["tiny"]
