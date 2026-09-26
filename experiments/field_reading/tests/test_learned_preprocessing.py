"""Tests for CRNN input preprocessing, batch padding, box jitter and SSBI comparison rules."""

import numpy as np

from experiments.field_reading.learned.crnn_model import timesteps_for_width
from experiments.field_reading.learned.crop_augmentation import jitter_box_and_crop
from experiments.field_reading.learned.line_image_preprocessing import PAD_VALUE, pad_line_batch, preprocess_line_image
from experiments.field_reading.learned.real_ssbi_scoring import comparable_text, pad_tight_crop


def test_preprocess_keeps_aspect_and_clamps_width_and_range():
    crop = np.zeros((40, 400, 3), np.uint8)
    line = preprocess_line_image(crop, 32, 48, 200)
    assert line.shape == (32, 200) and line.dtype == np.float32
    assert line.min() == -1.0
    assert preprocess_line_image(np.full((40, 20, 3), 255, np.uint8), 32, 48, 800).shape == (32, 48)
    assert preprocess_line_image(np.full((40, 80, 3), 255, np.uint8), 32, 48, 800).max() == 1.0


def test_pad_line_batch_rounds_width_up_and_reports_valid_widths():
    lines = [np.zeros((32, 50), np.float32), np.zeros((32, 130), np.float32)]
    batch, widths = pad_line_batch(lines, width_multiple=128)
    assert batch.shape == (2, 1, 32, 256) and widths == [50, 130]
    assert (batch[0, 0, :, 50:] == PAD_VALUE).all() and (batch[1, 0, :, :130] == 0).all()
    assert timesteps_for_width(130) == 32


def test_jitter_crop_stays_near_box_and_inside_context():
    context = np.zeros((100, 300, 3), np.uint8)
    box = [50.0, 40.0, 250.0, 60.0]
    generator = np.random.default_rng(0)
    for _ in range(50):
        crop = jitter_box_and_crop(context, box, generator)
        # box width 200 +- 2 * 2 px jitter + 2 * 6 px margin
        assert 200 <= crop.shape[1] <= 220 and 20 <= crop.shape[0] <= 40
    exact = jitter_box_and_crop(context, box, generator, jitter_fraction=0.0)
    assert exact.shape[:2] == (20 + 12, 200 + 12)


def test_ssbi_comparison_rules():
    assert comparable_text("date", " 06/04/ 2016") == "06/04/2016"
    assert comparable_text("amount_numeric", "35, 000") == "35,000"
    assert comparable_text("payee", "Dr.  Sheldon COOPER") == "dr. sheldon cooper"
    assert pad_tight_crop(np.zeros((30, 50, 3), np.uint8)).shape == (42, 62, 3)


def test_ssbi_date_digits_ignore_day_month_order():
    from experiments.field_reading.learned.real_ssbi_scoring import date_digits_match_any_order
    assert date_digits_match_any_order("06/04/2016", "06/04/2016")
    assert date_digits_match_any_order("04-06-2016", "06/04/2016")
    assert not date_digits_match_any_order("06/04/2018", "06/04/2016")
