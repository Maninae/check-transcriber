"""Stage 2 invariants: labels follow pixels through every warp."""

import cv2
import numpy as np
import pytest
from PIL import Image

from synth.compose.compose_scene import SceneConfig, compose_scene
from synth.compose.perspective import clip_polygon_to_rect, distort_points, polygon_area, radial_distortion_maps
from synth.render.check_fields import CheckLabel, FieldLabel
from synth.render.check_templates import build_template_catalog
from synth.render.fake_data import sample_check_content
from synth.render.render_check import render_check

NO_GLARE = SceneConfig(glare_probability=0.0, photo_long_side_range=(1400, 1800))
GRAY_BACKGROUND = np.full((1200, 1600, 3), 0.55, np.float32)


def solid_green_check() -> tuple[Image.Image, CheckLabel]:
    """A pure-green 6 x 2.75 in 'check' with one field, so its pixels are easy to find."""
    width, height = 1800, 825
    label = CheckLabel("solid", "personal", width, height, 300, [FieldLabel("payee", "X", (100, 100, 700, 200), False)])
    return Image.new("RGB", (width, height), (20, 220, 30)), label


@pytest.mark.parametrize("seed", range(8))
def test_corner_polygon_covers_pasted_pixels(seed):
    photo, label = compose_scene("t", [solid_green_check()], GRAY_BACKGROUND, "gray", np.random.default_rng(seed), NO_GLARE)
    height, width = photo.shape[:2]
    rgb = photo.astype(int)
    green_mask = (rgb[..., 1] - np.maximum(rgb[..., 0], rgb[..., 2])) > 60
    polygon_mask = np.zeros((height, width), np.uint8)
    cv2.fillPoly(polygon_mask, [np.round(np.array(label.checks[0].corners)).astype(np.int32)], 1)
    intersection = np.count_nonzero(green_mask & (polygon_mask > 0))
    union = np.count_nonzero(green_mask | (polygon_mask > 0))
    assert intersection / union > 0.95, f"IoU {intersection / union:.3f}"


@pytest.mark.parametrize("seed", range(6))
def test_field_boxes_stay_inside_photo(seed):
    rng = np.random.default_rng(seed)
    catalog = build_template_catalog()
    checks = [render_check(t, sample_check_content(t, rng), rng) for t in catalog[:6]]
    photo, label = compose_scene("t", checks, GRAY_BACKGROUND, "gray", rng, SceneConfig(photo_long_side_range=(1400, 1800)))
    height, width = photo.shape[:2]
    for check in label.checks:
        corners = np.array(check.corners)
        if check.fully_in_frame:
            assert (corners >= -0.5).all() and (corners[:, 0] <= width + 0.5).all() and (corners[:, 1] <= height + 0.5).all()
        for field in check.fields:
            if field.bbox_clipped is not None:
                x0, y0, x1, y1 = field.bbox_clipped
                assert 0 <= x0 <= x1 <= width and 0 <= y0 <= y1 <= height
            if check.fully_in_frame:
                quad = np.array(field.quad)
                assert (quad >= -0.5).all() and (quad[:, 0] <= width + 0.5).all() and (quad[:, 1] <= height + 0.5).all()


def test_point_distortion_inverts_the_image_remap():
    width, height, k1 = 1600, 1200, -0.03
    map_x, map_y = radial_distortion_maps(width, height, k1)
    source_points = np.array([[100.0, 80.0], [1500.0, 1100.0], [800.0, 600.0], [30.0, 1150.0]])
    photo_points = distort_points(source_points, width, height, k1)
    for (source_x, source_y), (photo_x, photo_y) in zip(source_points, photo_points):
        sampled_x = map_x[int(round(photo_y)), int(round(photo_x))]
        sampled_y = map_y[int(round(photo_y)), int(round(photo_x))]
        assert abs(sampled_x - source_x) < 1.5 and abs(sampled_y - source_y) < 1.5


def test_polygon_clipping():
    square = np.array([[-10.0, -10.0], [10.0, -10.0], [10.0, 10.0], [-10.0, 10.0]])
    clipped = clip_polygon_to_rect(square, 100, 100)
    assert polygon_area(clipped) == pytest.approx(100.0)
    assert len(clip_polygon_to_rect(square - 50, 100, 100)) == 0
