"""Soft box targets and the mask -> box post-process of the learned localizer."""

import numpy as np
import pytest

from experiments.field_reading.field_localization.segnet_dataset import box_to_soft_mask
from experiments.field_reading.field_localization.segnet_postprocess import field_probability_to_box

GRID_WIDTH, GRID_HEIGHT = 384, 176
CANVAS_SCALE = 768 / 1600
CROP_SIZE = (1600, 733)


def test_soft_mask_has_fractional_edges_and_exact_area():
    """Mass equals box area in grid units; edge cells carry the covered fraction."""
    mask = box_to_soft_mask([10.25, 5.5, 20.0, 8.0], GRID_WIDTH, GRID_HEIGHT)
    assert mask.sum() == pytest.approx(9.75 * 2.5)
    assert mask[6, 10] == pytest.approx(0.75)
    assert mask[5, 12] == pytest.approx(0.5)
    assert mask[8, 12] == 0.0


def test_mask_to_box_recovers_box_and_keeps_largest_component():
    """A strong block plus a weaker speck: the block's extent comes back in crop pixels."""
    probability = np.zeros((GRID_HEIGHT, GRID_WIDTH), np.float32)
    probability[50:70, 100:200] = 0.95
    probability[10:12, 10:12] = 0.9
    box, confidence = field_probability_to_box(probability, CANVAS_SCALE, 0.5, CROP_SIZE)
    expected_box = [100 * 2 / CANVAS_SCALE, 50 * 2 / CANVAS_SCALE, 200 * 2 / CANVAS_SCALE, 70 * 2 / CANVAS_SCALE]
    assert box == pytest.approx(expected_box, abs=2 / CANVAS_SCALE + 1e-6)
    assert confidence > 0.8


def test_weak_channel_yields_no_box():
    """Peak probability under the presence threshold -> field reported absent."""
    probability = np.full((GRID_HEIGHT, GRID_WIDTH), 0.2, np.float32)
    box, confidence = field_probability_to_box(probability, CANVAS_SCALE, 0.5, CROP_SIZE)
    assert box is None and confidence == pytest.approx(0.2)
