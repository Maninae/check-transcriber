"""Scene geometry invariants: deformation labels stay exact, alpha is honoured, framing fills the photo."""

import cv2
import numpy as np
import pytest
from PIL import Image

from scene_composer.compose_scene import SceneConfig, compose_scene
from scene_composer.geometry.check_plane_map import CheckPlaneMap, inverse_residual_px
from scene_composer.geometry.paper_deformation import PaperDeformation
from scene_composer.geometry.placement import plan_fan_layout, plan_loose_overlap_layout, visible_fractions
from scene_composer.tests.test_compose_scene import GRAY_BACKGROUND, STRONGLY_DEFORMED, filled_polygon_mask, green_mask_of, solid_green_check
from synthetic_checks.check_fields import CheckLabel, FieldLabel
from synthetic_checks.check_templates import build_template_catalog
from synthetic_checks.content.fake_data import sample_check_content
from synthetic_checks.render_check import render_check

EXTREME_DEFORMATION = PaperDeformation(
    6.0, 2.75, curl={"edge": "right", "length": 1.0, "lift": 0.35}, corner_lift={"corner": 0, "reach": 1.2, "lift": 0.3},
    fold={"axis": 0, "knots": [0.0, 3.0, 6.0], "heights": [0.0, 0.4, 0.0], "rounding": 0.01, "crease_line": 0.1},
    waves=[{"amplitude": 0.04, "wavelength": 2.0, "angle": 0.3, "phase": 0.0}])


def test_inverse_map_is_accurate_under_heavy_deformation():
    plane_map = CheckPlaneMap(1800, 825, 300, (1500.0, 900.0), 17.0, 250.0, EXTREME_DEFORMATION, (600.0, 300.0), 2000.0)
    assert inverse_residual_px(plane_map, 600, 300, 1800, 1200) < 0.5


def test_flat_map_is_the_rigid_affine():
    plane_map = CheckPlaneMap(1800, 825, 300, (900.0, 500.0), 90.0, 150.0, PaperDeformation(6.0, 2.75), (0.0, 0.0), 2000.0)
    corners = plane_map.check_to_plane(np.array([[0.0, 0.0], [1800.0, 825.0]]))
    assert np.allclose(corners, [[900 - 1.375 * 150, 500 + 3 * 150], [900 + 1.375 * 150, 500 - 3 * 150]])


def ink_only_check(seed: int, field_name: str) -> tuple[Image.Image, CheckLabel]:
    """A real render reduced to white paper plus pure-red ink inside one field's box."""
    rng = np.random.default_rng(seed)
    template = build_template_catalog()[seed % 6]
    image, label = render_check(template, sample_check_content(template, rng), rng)
    pixels = np.asarray(image.convert("RGB")).astype(int)
    field = next(f for f in label.fields if f.field_name == field_name)
    x0, y0, x1, y1 = field.box
    ink = np.zeros(pixels.shape[:2], bool)
    ink[y0:y1, x0:x1] = pixels[y0:y1, x0:x1].mean(axis=2) < 110
    painted = np.full_like(pixels, 255)
    painted[ink] = (230, 0, 0)
    label.fields = [field]
    return Image.fromarray(painted.astype(np.uint8), "RGB"), label


@pytest.mark.parametrize("seed", range(6))
def test_field_quad_covers_its_ink_after_curl_and_fold(seed):
    check = ink_only_check(seed, "amount_words")
    photo, label = compose_scene("t", [check], GRAY_BACKGROUND, "gray", np.random.default_rng(100 + seed), STRONGLY_DEFORMED)
    rgb = photo.astype(int)
    red = (rgb[..., 0] - np.maximum(rgb[..., 1], rgb[..., 2])) > 90
    assert np.count_nonzero(red) > 50
    quad_mask = filled_polygon_mask(label.checks[0].fields[0].quad, *photo.shape[:2])
    near_quad = cv2.dilate(quad_mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    assert np.count_nonzero(red & ~near_quad) / np.count_nonzero(red) < 0.005


def rgba_green_check_with_torn_end() -> tuple[Image.Image, CheckLabel]:
    """The green test check as RGBA with its right fifth transparent (paper missing there)."""
    image, label = solid_green_check()
    rgba = np.dstack([np.asarray(image), np.full(image.size[::-1], 255, np.uint8)])
    rgba[:, int(label.width_px * 0.8):, 3] = 0
    return Image.fromarray(rgba, "RGBA"), label


@pytest.mark.parametrize("seed", range(4))
def test_rgba_alpha_is_paper_coverage(seed):
    flat = SceneConfig(glare_probability=0.0, photo_long_side_range=(1400, 1800), deformation_strength=0.0)
    photo_rgba, label_rgba = compose_scene("t", [rgba_green_check_with_torn_end()], GRAY_BACKGROUND, "g", np.random.default_rng(seed), flat)
    photo_rgb, label_rgb = compose_scene("t", [solid_green_check()], GRAY_BACKGROUND, "g", np.random.default_rng(seed), flat)
    assert label_rgba.checks[0].corners == label_rgb.checks[0].corners
    ratio = np.count_nonzero(green_mask_of(photo_rgba)) / np.count_nonzero(green_mask_of(photo_rgb))
    assert 0.7 < ratio < 0.88, f"RGBA green area ratio {ratio:.3f}"  # a fifth of the paper, give or take foreshortening


def test_scene_depends_only_on_rng():
    checks = [solid_green_check(), solid_green_check()]
    photo_a, label_a = compose_scene("t", checks, GRAY_BACKGROUND, "g", np.random.default_rng(7), STRONGLY_DEFORMED)
    photo_b, label_b = compose_scene("t", checks, GRAY_BACKGROUND, "g", np.random.default_rng(7), STRONGLY_DEFORMED)
    assert np.array_equal(photo_a, photo_b) and label_a.to_dict() == label_b.to_dict()


@pytest.mark.parametrize("seed", range(12))
def test_group_fills_the_frame(seed):
    rng = np.random.default_rng(seed)
    checks = [solid_green_check() for _ in range(int(rng.integers(2, 7)))]
    photo, label = compose_scene("t", checks, GRAY_BACKGROUND, "g", rng, SceneConfig(photo_long_side_range=(1000, 1200)))
    framing = label.effects["framing"]
    if framing["wide_shot"] or framing["pushed_out_check_index"] is not None:
        return
    points = np.vstack([np.array(c.outline) for c in label.checks])
    fill = max(np.ptp(points[:, 0]) / label.image_width, np.ptp(points[:, 1]) / label.image_height)
    assert 0.66 <= fill <= 1.0 and all(c.fully_in_frame for c in label.checks), f"fill {fill:.2f}"


@pytest.mark.parametrize("seed", range(20))
def test_loose_layouts_keep_seventy_percent_visible(seed):
    rng = np.random.default_rng(seed)
    sizes = [(6.0, 2.75) if rng.random() < 0.7 else (8.5, 3.5) for _ in range(int(rng.integers(2, 9)))]
    for planner in (plan_loose_overlap_layout, plan_fan_layout):
        placements = planner(sizes, rng)
        assert min(visible_fractions(sizes, placements, list(range(len(sizes))))) >= 0.7
