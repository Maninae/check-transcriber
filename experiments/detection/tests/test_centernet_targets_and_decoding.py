"""CenterNet target encoding <-> numpy decoding round trip, and augmentation corner-order invariants."""

import numpy as np
import pytest

from experiments.detection.learned.centernet.centernet_config import OUTPUT_STRIDE
from experiments.detection.learned.centernet.centernet_decoding import (
    decode_checks_in_input_pixels,
    map_input_corners_to_source,
)
from experiments.detection.learned.centernet.check_center_targets import encode_check_targets
from experiments.detection.learned.centernet.letterbox_geometry import letterbox_affine_matrix
from experiments.detection.learned.centernet.scene_augmentation import (
    augment_scene_for_training,
    letterbox_scene_for_evaluation,
)

INPUT_SIZE = 256
ROUND_TRIP_TOLERANCE_PIXELS = 1e-3  # encoding is exact; well under the 1 px requirement


def rectangle_corners(center_x: float, center_y: float, width: float, height: float, angle_degrees: float) -> np.ndarray:
    """TL,TR,BR,BL of a check rotated clockwise (image coords, y down) by `angle_degrees`."""
    half_extents = np.array([[-width, -height], [width, -height], [width, height], [-width, height]]) / 2
    angle = np.deg2rad(angle_degrees)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return half_extents @ rotation.T + np.array([center_x, center_y])


def encode_then_decode(corner_sets: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Treat the targets as a perfect network output and decode them."""
    targets = encode_check_targets(corner_sets, INPUT_SIZE)
    return decode_checks_in_input_pixels(targets["center_heatmap_target"][0], targets["corner_offset_target"], score_threshold=0.5)


def match_decoded_to_truth(decoded_corners: np.ndarray, truth_corner_sets: list[np.ndarray]) -> np.ndarray:
    """Max corner error per truth check against the decoded quad with the nearest center."""
    errors = []
    for truth in truth_corner_sets:
        nearest = np.argmin(np.linalg.norm(decoded_corners.mean(axis=1) - truth.mean(axis=0), axis=1))
        errors.append(np.linalg.norm(decoded_corners[nearest] - truth, axis=1).max())
    return np.array(errors)


@pytest.mark.parametrize("angle_degrees", [0.0, 37.0, 90.0, 180.0, 271.5])
def test_single_check_round_trip_is_exact_and_keeps_order(angle_degrees):
    truth = rectangle_corners(101.3, 130.7, 120, 55, angle_degrees)
    decoded_corners, scores = encode_then_decode([truth])
    assert len(decoded_corners) == 1
    assert scores[0] == pytest.approx(1.0)
    np.testing.assert_allclose(decoded_corners[0], truth, atol=ROUND_TRIP_TOLERANCE_PIXELS)


def test_several_checks_each_decode_once():
    truths = [
        rectangle_corners(60, 60, 90, 40, 10),
        rectangle_corners(190, 70, 80, 36, 100),
        rectangle_corners(128, 190, 150, 64, 200),
    ]
    decoded_corners, _ = encode_then_decode(truths)
    assert len(decoded_corners) == len(truths)
    assert match_decoded_to_truth(decoded_corners, truths).max() < ROUND_TRIP_TOLERANCE_PIXELS


def test_partly_out_of_frame_check_keeps_its_off_canvas_corners():
    truth = rectangle_corners(-10, 120, 140, 60, 0)  # center left of the canvas
    targets = encode_check_targets([truth], INPUT_SIZE)
    anchor_column, anchor_row = targets["anchor_cells"][0]
    assert 0 <= anchor_column * OUTPUT_STRIDE < INPUT_SIZE
    decoded_corners, _ = encode_then_decode([truth])
    np.testing.assert_allclose(decoded_corners[0], truth, atol=ROUND_TRIP_TOLERANCE_PIXELS)
    assert decoded_corners[0][:, 0].min() < 0


def test_barely_visible_check_is_not_a_target():
    truth = rectangle_corners(-55, 120, 120, 60, 0)  # only a 5 px sliver on canvas
    assert len(encode_check_targets([truth], INPUT_SIZE)["anchor_cells"]) == 0


def test_letterbox_inverse_returns_source_pixels():
    source_width, source_height = 300, 500
    affine = letterbox_affine_matrix(source_width, source_height, INPUT_SIZE)
    source_corners = rectangle_corners(150, 260, 200, 90, 15)[None]
    input_corners = source_corners @ affine[:, :2].T + affine[:, 2]
    np.testing.assert_allclose(map_input_corners_to_source(input_corners, affine), source_corners, atol=1e-9)


def synthetic_scene(corner_sets: list[np.ndarray], width: int = 400, height: int = 300) -> dict:
    """Plain dark source image carrying the given check corners."""
    image = np.full((height, width, 3), 40, dtype=np.uint8)
    return {"source_image_rgb": image, "check_corner_sets": corner_sets}


def signed_area(corners: np.ndarray) -> float:
    """Shoelace area; positive = clockwise in image coords (y down)."""
    x, y = corners[:, 0], corners[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


@pytest.mark.parametrize("seed", range(20))
def test_augmentation_moves_corners_rigidly_and_keeps_order(seed):
    truth = rectangle_corners(200, 150, 160, 70, 25)
    data_dict = augment_scene_for_training(synthetic_scene([truth]), INPUT_SIZE, np.random.default_rng(seed))
    augmented = data_dict["input_check_corner_sets"][0]
    affine = data_dict["source_to_input_affine"]
    np.testing.assert_allclose(augmented, truth @ affine[:, :2].T + affine[:, 2], atol=1e-9)
    assert signed_area(augmented) > 0  # still clockwise: no mirroring
    top_edge_source = truth[1] - truth[0]
    top_edge_augmented = augmented[1] - augmented[0]
    scale = np.sqrt(abs(np.linalg.det(affine[:, :2])))
    np.testing.assert_allclose(np.linalg.norm(top_edge_augmented), scale * np.linalg.norm(top_edge_source), rtol=1e-9)
    assert data_dict["input_image_rgb"].shape == (INPUT_SIZE, INPUT_SIZE, 3)


def test_augmented_image_content_follows_the_corners():
    """A bright TL marker drawn in the source must land at the augmented TL corner."""
    truth = rectangle_corners(200, 150, 160, 70, 25)
    data_dict = synthetic_scene([truth])
    marker_x, marker_y = np.round(truth[0]).astype(int)
    data_dict["source_image_rgb"][marker_y - 3 : marker_y + 4, marker_x - 3 : marker_x + 4] = (255, 0, 0)
    checked_seeds = 0
    for seed in range(8):
        augmented = augment_scene_for_training(dict(data_dict), INPUT_SIZE, np.random.default_rng(seed))
        tl_x, tl_y = augmented["input_check_corner_sets"][0][0]
        if not (3 <= tl_x < INPUT_SIZE - 3 and 3 <= tl_y < INPUT_SIZE - 3):
            continue
        red_minus_green = augmented["input_image_rgb"][..., 0].astype(int) - augmented["input_image_rgb"][..., 1].astype(int)
        brightest_row, brightest_column = np.unravel_index(np.argmax(red_minus_green), red_minus_green.shape)
        assert np.hypot(brightest_column - tl_x, brightest_row - tl_y) < 4.0
        checked_seeds += 1
    assert checked_seeds >= 3


def test_evaluation_letterbox_is_deterministic():
    truth = rectangle_corners(200, 150, 160, 70, 0)
    first = letterbox_scene_for_evaluation(synthetic_scene([truth]), INPUT_SIZE)
    second = letterbox_scene_for_evaluation(synthetic_scene([truth]), INPUT_SIZE)
    np.testing.assert_array_equal(first["input_image_rgb"], second["input_image_rgb"])
