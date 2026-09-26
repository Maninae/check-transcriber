"""Framing regimes: wide is unchanged byte for byte, close and single labels stay exact, the mix is exact per block."""

import hashlib
import json
from collections import Counter
from dataclasses import replace

import cv2
import numpy as np
import pytest
from PIL import Image

from scene_composer.compose_scene import SceneConfig, compose_scene, sample_check_count
from scene_composer.framing_regime_mix import (
    REGIME_BLOCK_SIZE,
    framing_regime_for_scene,
    parse_framing_regime_mix,
    regime_counts_per_block,
)
from scene_composer.geometry import framing_regimes
from scene_composer.geometry.closeup_placement import plan_single_layout
from scene_composer.geometry.framing_regimes import FramingRegime, framing_regime_of_label
from scene_composer.on_demand import compose_scene_on_demand
from scene_composer.tests.test_compose_scene import GRAY_BACKGROUND, filled_polygon_mask, green_mask_of, solid_green_check
from synthetic_backgrounds.procedural import write_procedural_backgrounds
from synthetic_checks.check_fields import CheckLabel, FieldLabel

# sha256[:16] of (photo bytes, sorted label JSON), composed by the v0.5.0 generator (commit 95ebb71) before
# regimes existed: seed 5, procedural backgrounds (9, seed 2), photo long side 900-1100.
PRE_REGIME_SCENE_HASHES = {
    ("train", 0): ("11b977d9f22fe21d", "bb29434efc5c5913"),
    ("train", 1): ("2c12e567647a0074", "59f9b624a501b63d"),
    ("train", 2): ("8bb32365e1080e7d", "06c1662a0a5b7358"),
    ("eval", 3): ("05c4982fd467f130", "daeb0a06c842c4b0"),
    ("val", 4): ("ac6adad3a20a56da", "2e36e0cf9b2643c4"),
}
SMALL_CLOSEUP = dict(glare_probability=0.0, photo_long_side_range=(1400, 1800), closeup_photo_long_side_range=(1400, 1800))
CLOSE_CONFIG = SceneConfig(framing_regime="close", **SMALL_CLOSEUP)
SINGLE_CONFIG = SceneConfig(framing_regime="single", **SMALL_CLOSEUP)
CLOSEUP_DEFORMED = {name: replace(config, deformation_strength=4.0, lens_blur_probability=0.0, motion_blur_probability=0.0,
                                  sharpen_probability=0.0) for name, config in (("close", CLOSE_CONFIG), ("single", SINGLE_CONFIG))}
PAPER_COLOURS = {  # rgb, and the channel that must dominate the other two
    "green": ((20, 220, 30), 1),
    "blue": ((30, 40, 230), 2),
    "red": ((230, 30, 20), 0),
}


@pytest.fixture(scope="module")
def background_root(tmp_path_factory):
    """The procedural backgrounds the golden hashes were taken on."""
    root = tmp_path_factory.mktemp("backgrounds")
    write_procedural_backgrounds(root / "photos", 9, seed=2)
    return root


@pytest.fixture
def forced_steep_corner_pushes(monkeypatch):
    """Single-regime policy with a steep camera, a corner push and no crop on every scene."""
    policy = replace(framing_regimes.FRAMING_POLICIES[FramingRegime.SINGLE], steep_angle_probability=1.0,
                     out_of_frame_probability=1.0, corner_push_probability=1.0, cropped_to_subject_probability=0.0)
    monkeypatch.setitem(framing_regimes.FRAMING_POLICIES, FramingRegime.SINGLE, policy)


def test_wide_regime_is_byte_identical_to_the_pre_regime_generator(background_root):
    config = SceneConfig(photo_long_side_range=(900, 1100))
    for (split, index), (photo_hash, label_hash) in PRE_REGIME_SCENE_HASHES.items():
        scene = compose_scene_on_demand(5, split, index, backgrounds=background_root, scene_config=config, framing_regime="wide")
        assert hashlib.sha256(scene.photo_rgb.tobytes()).hexdigest()[:16] == photo_hash, (split, index)
        assert hashlib.sha256(json.dumps(scene.label_record(), sort_keys=True).encode()).hexdigest()[:16] == label_hash
        assert "framing_regime" not in scene.label_record()["effects"]["framing"]


def coloured_check(colour: str) -> tuple[Image.Image, CheckLabel]:
    """The solid test check in another colour."""
    image, label = solid_green_check()
    return Image.new("RGB", image.size, PAPER_COLOURS[colour][0]), label


def colour_mask(photo: np.ndarray, colour: str) -> np.ndarray:
    """Pixels clearly of one test paper colour."""
    rgb = photo.astype(int)
    channel = PAPER_COLOURS[colour][1]
    others = [c for c in range(3) if c != channel]
    return (rgb[..., channel] - np.maximum(rgb[..., others[0]], rgb[..., others[1]])) > 60


def corner_polygon_iou(photo: np.ndarray, corners: list[list[float]]) -> float:
    """IoU of the green paper pixels and the filled corner polygon."""
    green = green_mask_of(photo)
    polygon = filled_polygon_mask(corners, *photo.shape[:2])
    return np.count_nonzero(green & polygon) / np.count_nonzero(green | polygon)


@pytest.mark.parametrize("regime", ["close", "single"])
@pytest.mark.parametrize("seed", range(6))
def test_corner_polygon_covers_pasted_pixels_close_up(regime, seed):
    config = CLOSE_CONFIG if regime == "close" else SINGLE_CONFIG
    photo, label = compose_scene("t", [solid_green_check()], GRAY_BACKGROUND, "gray", np.random.default_rng(seed), config)
    assert label.effects["framing"]["framing_regime"] == regime
    assert corner_polygon_iou(photo, label.checks[0].corners) > 0.95


@pytest.mark.parametrize("seed", range(6))
def test_corner_polygon_is_exact_under_steep_angle_and_corner_cut(seed, forced_steep_corner_pushes):
    photo, label = compose_scene("t", [solid_green_check()], GRAY_BACKGROUND, "gray", np.random.default_rng(seed), SINGLE_CONFIG)
    framing = label.effects["framing"]
    assert framing["steep_angle"] and framing["pushed_out_check_index"] == 0 and not label.checks[0].fully_in_frame
    assert corner_polygon_iou(photo, label.checks[0].corners) > 0.95
    corners = np.array(label.checks[0].corners)
    edges = [np.linalg.norm(corners[(i + 1) % 4] - corners[i]) for i in range(4)]
    opposite_ratios = [min(edges[i], edges[i + 2]) / max(edges[i], edges[i + 2]) for i in (0, 1)]
    assert min(opposite_ratios) < 0.85, f"steep tilt should make the far edge clearly shorter: {opposite_ratios}"


@pytest.mark.parametrize("regime", ["close", "single"])
@pytest.mark.parametrize("seed", range(6))
def test_outline_covers_and_hugs_deformed_paper_close_up(regime, seed):
    photo, label = compose_scene("t", [solid_green_check()], GRAY_BACKGROUND, "gray", np.random.default_rng(seed),
                                 CLOSEUP_DEFORMED[regime])
    check = label.checks[0]
    green = green_mask_of(photo)
    outline = filled_polygon_mask(check.outline, *photo.shape[:2])
    covered = cv2.dilate(outline.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    assert np.count_nonzero(green & ~covered) / np.count_nonzero(green) < 0.002, "paper outside the outline"
    assert np.count_nonzero(green & outline) / np.count_nonzero(green | outline) > 0.96


@pytest.mark.parametrize("seed", range(6))
def test_each_close_check_stays_inside_its_own_outline(seed):
    checks = [coloured_check(colour) for colour in PAPER_COLOURS]
    photo, label = compose_scene("t", checks, GRAY_BACKGROUND, "gray", np.random.default_rng(seed), CLOSEUP_DEFORMED["close"])
    for colour, check in zip(PAPER_COLOURS, label.checks):
        pixels = colour_mask(photo, colour)
        covered = cv2.dilate(filled_polygon_mask(check.outline, *photo.shape[:2]).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        assert np.count_nonzero(pixels) > 1000 or not check.fully_in_frame
        assert np.count_nonzero(pixels & ~covered) / max(1, np.count_nonzero(pixels)) < 0.003, colour


@pytest.mark.parametrize("seed", range(10))
def test_close_up_groups_fill_the_frame(seed):
    rng = np.random.default_rng(seed)
    for config, low in ((CLOSE_CONFIG, 0.83), (SINGLE_CONFIG, 0.68)):
        count = 1 if config is SINGLE_CONFIG else int(rng.integers(2, 7))
        photo, label = compose_scene("t", [solid_green_check() for _ in range(count)], GRAY_BACKGROUND, "g", rng, config)
        framing = label.effects["framing"]
        if framing["pushed_out_check_index"] is not None:
            continue
        points = np.vstack([np.array(check.outline) for check in label.checks])
        fill = max(np.ptp(points[:, 0]) / label.image_width, np.ptp(points[:, 1]) / label.image_height)
        assert low <= fill <= 1.04, f"{config.framing_regime} fill {fill:.2f}"


def test_check_counts_per_regime():
    rng = np.random.default_rng(0)
    close_counts = Counter(sample_check_count(rng, "close") for _ in range(2000))
    assert set(close_counts) == {2, 3, 4, 5, 6}
    assert close_counts[5] + close_counts[6] > 0.6 * 2000
    assert sample_check_count(rng, "single") == 1
    assert all(1 <= sample_check_count(rng) <= 12 for _ in range(50))


def test_single_layout_takes_any_rotation_and_exactly_one_check():
    rng = np.random.default_rng(1)
    rotations = [plan_single_layout([(6.0, 2.75)], rng)[1][0].rotation_degrees % 360 for _ in range(400)]
    off_axis = [r for r in rotations if min(abs(r - a) for a in (0, 90, 180, 270, 360)) > 15]
    assert 0.1 < len(off_axis) / len(rotations) < 0.35
    with pytest.raises(ValueError, match="exactly one check"):
        plan_single_layout([(6.0, 2.75)] * 2, rng)


def test_regime_mix_is_exact_per_block_and_deterministic():
    mix = parse_framing_regime_mix("close=0.75,single=0.25")
    regimes = [framing_regime_for_scene(11, "eval", index, mix) for index in range(600)]
    assert Counter(regimes) == {"close": 450, "single": 150}
    assert regimes == [framing_regime_for_scene(11, "eval", index, mix) for index in range(600)]
    assert regimes != [framing_regime_for_scene(12, "eval", index, mix) for index in range(600)]
    default = parse_framing_regime_mix("wide=0.4,close=0.45,single=0.15")
    assert regime_counts_per_block(default) == {"wide": 8, "close": 9, "single": 3}
    assert sum(regime_counts_per_block({"wide": 1 / 3, "close": 1 / 3, "single": 1 / 3}).values()) == REGIME_BLOCK_SIZE
    with pytest.raises(ValueError):
        parse_framing_regime_mix("close=0.5,single=0.4")
    with pytest.raises(ValueError, match="unknown framing regime"):
        parse_framing_regime_mix("closeup=1")


def test_labels_record_the_regime_only_off_wide():
    assert framing_regime_of_label({"effects": {"framing": {}}}) == "wide"
    _, label = compose_scene("t", [solid_green_check()], GRAY_BACKGROUND, "g", np.random.default_rng(0), SINGLE_CONFIG)
    assert framing_regime_of_label(label.to_dict()) == "single" and label.layout_mode == "single"
