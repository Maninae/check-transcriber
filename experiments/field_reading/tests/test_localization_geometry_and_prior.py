"""Box arithmetic and layout-prior estimation on hand-built records."""

import pytest

from experiments.field_reading.field_localization.box_geometry import box_coverage_of_target, box_iou
from experiments.field_reading.field_localization.layout_prior import estimate_layout_priors


def test_box_iou_known_values():
    """Identical boxes -> 1, disjoint -> 0, half overlap of equal boxes -> 1/3."""
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)
    assert box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert box_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)
    assert box_iou([0, 0, 0, 0], [0, 0, 0, 0]) == 0.0


def test_coverage_counts_generous_boxes_as_covering():
    """A box strictly containing the target covers it fully; a half box covers half."""
    assert box_coverage_of_target([0, 0, 100, 100], [10, 10, 20, 20]) == pytest.approx(1.0)
    assert box_coverage_of_target([0, 0, 15, 100], [10, 10, 20, 20]) == pytest.approx(0.5)


def make_check_record(family: str, size_kind: str, payee_box: list[float] | None, crop_width: int = 1600,
                      crop_height: int = 800) -> dict:
    """Minimal localization check record carrying at most a payee field."""
    fields = {} if payee_box is None else {"payee": {"box": payee_box, "status": "ok", "handwritten": True}}
    return {"layout_family": family, "size_kind": size_kind, "check_crop_size": (crop_width, crop_height),
            "fields": fields}


def test_prior_statistics_normalize_and_group():
    """Median box and presence are per group, in [0, 1] crop fractions."""
    records = [make_check_record("fam_a", "personal", [160, 400, 800, 480]),
               make_check_record("fam_a", "personal", [320, 400, 960, 480]),
               make_check_record("fam_a", "personal", None),
               make_check_record("fam_b", "personal", [0, 0, 1600, 80])]
    priors = estimate_layout_priors(records, "layout_family")
    family_a_payee = priors["fam_a"]["payee"]
    assert family_a_payee["median_box"] == pytest.approx([0.15, 0.5, 0.55, 0.6])
    assert family_a_payee["median_height"] == pytest.approx(0.1)
    assert family_a_payee["presence_rate"] == pytest.approx(2 / 3)
    assert family_a_payee["search_region"][0] == pytest.approx(0.1, abs=0.002)
    assert family_a_payee["search_region"][2] == pytest.approx(0.6, abs=0.002)
    by_size_kind = estimate_layout_priors(records, "size_kind")
    assert by_size_kind["personal"]["payee"]["box_count"] == 3
