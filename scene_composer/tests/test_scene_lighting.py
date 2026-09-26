"""Lighting invariants: one scene light, shadows keep the ambient colour, no shadow hides a check, paper stays white."""

import cv2
import numpy as np
import pytest

from scene_composer.geometry.check_plane_map import CheckPlaneMap
from scene_composer.geometry.paper_deformation import PaperDeformation
from scene_composer.harmonization.harmonize_color_transfer import MAX_CHROMA_SHIFT, MAX_LUMINANCE_DROP, luminance_preserving_transfer
from scene_composer.lighting.baked_background_light import MAX_INTERREFLECTION_TINT, background_color_gains
from scene_composer.lighting.cast_shadows import UMBRA_LEVEL, MAX_UMBRA_SHARE_OF_A_CHECK, cast_occluder_shadow
from scene_composer.lighting.paper_shading import PaperSurface, paper_geometry_factors
from scene_composer.lighting.scene_irradiance import LightBuffers, apply_scene_light
from scene_composer.lighting.scene_light import (LUMINANCE_WEIGHTS, SceneLight, background_brightness_azimuth, kelvin_to_linear_rgb,
                                                 sample_scene_light, srgb_to_linear)

VISIBLE_QUAD = np.array([[100.0, 100.0], [1500.0, 100.0], [1500.0, 1150.0], [100.0, 1150.0]])
SURFACE = PaperSurface(base_lift_inches=0.02, drop_shadow_opacity=0.9, crease_ridge_slope=0.5,
                       crease_ridge_half_width_inches=0.03, contact_occlusion=0.4)


def overhead_light(**changes) -> SceneLight:
    """A neutral daylight key from +x at 45 degrees, no fill, neutral white balance."""
    values = dict(key_azimuth_radians=0.0, key_elevation_radians=np.pi / 4, key_kelvin=6500, key_angular_radius=0.03,
                  key_falloff=0.0, ambient_kelvin=6500, ambient_fraction=0.3, fill=None, white_balance_gains=(1.0, 1.0, 1.0),
                  exposure=1.0, azimuth_from_background=False)
    values.update(changes)
    return SceneLight(**values)


def test_black_body_colours_have_unit_luminance_and_warm_to_cool_order():
    warm, cool = kelvin_to_linear_rgb(2800), kelvin_to_linear_rgb(8000)
    assert warm @ LUMINANCE_WEIGHTS == pytest.approx(1.0) and cool @ LUMINANCE_WEIGHTS == pytest.approx(1.0)
    assert warm[0] / warm[2] > 2 and cool[2] > cool[0]


def test_key_light_aims_from_the_backgrounds_bright_side():
    x_ramp = np.linspace(0.7, 1.3, 400, dtype=np.float32)
    brighter_on_right = np.tile(x_ramp, (300, 1))
    assert abs(background_brightness_azimuth(brighter_on_right)) < 0.1
    assert background_brightness_azimuth(np.ones((300, 400), np.float32)) is None
    light = sample_scene_light(brighter_on_right, np.random.default_rng(0))
    assert light.azimuth_from_background and abs(np.angle(np.exp(1j * light.key_azimuth_radians))) < np.deg2rad(30)


def test_shadow_keeps_the_ambient_colour():
    canvas = np.full((1250, 1600, 3), 0.8, np.float32)
    buffers = LightBuffers.open_sheet(1250, 1600)
    buffers.key_factor[500:700, 600:900] = 0
    light = overhead_light(key_kelvin=6500, ambient_kelvin=2800)
    lit = apply_scene_light(canvas, buffers, light, VISIBLE_QUAD)
    shadow, open_sheet = srgb_to_linear(lit[600, 750]), srgb_to_linear(lit[300, 300])
    assert shadow @ LUMINANCE_WEIGHTS < 0.5 * (open_sheet @ LUMINANCE_WEIGHTS)
    assert shadow[0] / shadow[2] > 1.5 * open_sheet[0] / open_sheet[2], "shadow should turn toward the warm ambient"


@pytest.mark.parametrize("seed", range(12))
def test_cast_shadow_never_hides_a_check(seed):
    id_map = np.zeros((1250, 1600), np.uint8)
    for index, (x0, y0) in enumerate([(200, 200), (850, 200), (200, 650), (850, 650)]):
        id_map[y0:y0 + 380, x0:x0 + 560] = index + 1
    light = sample_scene_light(np.ones((1250, 1600), np.float32), np.random.default_rng(seed))
    shadow, record = cast_occluder_shadow(id_map.shape, VISIBLE_QUAD, (800.0, 1400.0), 16.0, 60.0, id_map, light,
                                          np.random.default_rng(seed))
    if shadow is None:
        pytest.skip("no rule-abiding placement for this seed")
    full = cv2.resize(shadow, (1600, 1250), interpolation=cv2.INTER_LINEAR)
    for owner_id in range(1, 5):
        assert (full[id_map == owner_id] > UMBRA_LEVEL).mean() <= MAX_UMBRA_SHARE_OF_A_CHECK + 0.05
    assert 0 < record["frame_share"] <= 0.3


def test_harmonization_never_darkens_white_paper_toward_the_sheet():
    paper = np.full((20, 20, 3), 0.93, np.float32)
    network_output = np.full((20, 20, 3), [0.45, 0.55, 0.40], np.float32)  # pulled toward a green sheet, much darker
    result = luminance_preserving_transfer(paper, network_output, blend=1.0)
    assert (result @ LUMINANCE_WEIGHTS).min() >= 0.93 - MAX_LUMINANCE_DROP - 0.01
    chroma_shift = cv2.cvtColor(result, cv2.COLOR_RGB2YCrCb)[..., 1:] - cv2.cvtColor(paper, cv2.COLOR_RGB2YCrCb)[..., 1:]
    assert np.linalg.norm(chroma_shift, axis=-1).max() <= MAX_CHROMA_SHIFT + 1e-3
    assert result[..., 1].mean() > result[..., 0].mean(), "the green cast should still come through"


def test_fold_panels_and_crease_answer_the_key_light():
    tent = PaperDeformation(6.0, 2.75, fold={"axis": 0, "knots": [0.0, 3.0, 6.0], "heights": [0.0, 0.3, 0.0],
                                             "rounding": 0.01, "crease_line": 0.0})
    plane_map = CheckPlaneMap(1800, 825, 300, (1000.0, 600.0), 0.0, 150.0, tent, (1000.0, 600.0), 3000.0)
    map_x, map_y = np.meshgrid(np.linspace(0, 1800, 721, dtype=np.float32), np.full(8, 400, np.float32))
    key, _ = paper_geometry_factors(plane_map, map_x, map_y, overhead_light(), SURFACE)
    # Light from +s (the right): the left panel rises toward the crease, so its normal tips away and it darkens;
    # the right panel slopes down toward the light, faces it, and brightens.
    left_panel, right_panel = key[:, 100].mean(), key[:, 620].mean()
    assert right_panel > 1.05 and left_panel < 0.95, "tilted panels must brighten toward and darken away from the key"
    crease_band = key[:, 350:371].mean(axis=0)
    assert crease_band.max() - crease_band.min() > 0.2, "the crease ridge should catch light on one side and darken on the other"


def test_a_strongly_coloured_surface_only_tints_paper_slightly():
    """A saturated purple background moves white paper by at most the inter-reflection cap per channel."""
    purple = np.zeros((64, 64, 3), np.float32)
    purple[..., 0], purple[..., 2] = 0.7, 0.6
    gains = background_color_gains(purple, strength=0.6)
    assert np.all(np.abs(gains - 1) <= MAX_INTERREFLECTION_TINT + 1e-6)

