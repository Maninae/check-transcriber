"""End-to-end tests of the classical detector on drawn scenes (counts and corner accuracy)."""

import numpy as np
from shapely.geometry import Polygon, box

from experiments.detection.classical.classical_detector_config import ClassicalDetectorConfig
from experiments.detection.classical.detect_checks_classical import detect_checks_classical
from experiments.detection.tests.classical_synthetic_scenes import build_scene, rotated_rectangle_corners

CORNER_TOLERANCE_PIXELS = 3.0


def minimum_cyclic_corner_error(predicted_corners: np.ndarray, true_corners: np.ndarray) -> float:
    """Mean corner distance at the best cyclic alignment (clockwise order is assumed on both)."""
    return min(
        float(np.linalg.norm(np.roll(predicted_corners, shift, axis=0) - true_corners, axis=1).mean())
        for shift in range(4)
    )


def match_detection_to(true_corners: np.ndarray, detections: list) -> np.ndarray:
    """Corners of the detection whose centroid is nearest the true quad's centroid."""
    true_centroid = true_corners.mean(axis=0)
    return min(detections, key=lambda item: np.linalg.norm(item.corners.mean(axis=0) - true_centroid)).corners




def test_single_rotated_check_on_dark_background():
    true_corners = rotated_rectangle_corners((1000, 750), 900, 17.0)
    detections = detect_checks_classical(build_scene([true_corners]), ClassicalDetectorConfig())
    assert len(detections) == 1
    assert minimum_cyclic_corner_error(detections[0].corners, true_corners) < CORNER_TOLERANCE_PIXELS
    assert detections[0].orientation_known is False


def test_three_checks_on_textured_background():
    true_corner_list = [
        rotated_rectangle_corners((520, 380), 700, 0.0),
        rotated_rectangle_corners((1450, 420), 650, -8.0),
        rotated_rectangle_corners((1000, 1100), 800, 90.0 + 4.0),
    ]
    detections = detect_checks_classical(build_scene(true_corner_list, textured=True), ClassicalDetectorConfig())
    assert len(detections) == 3
    for true_corners in true_corner_list:
        assert minimum_cyclic_corner_error(match_detection_to(true_corners, detections), true_corners) < CORNER_TOLERANCE_PIXELS


def test_two_touching_checks_are_separated():
    width = 800
    height = width / 2.4
    upper_corners = rotated_rectangle_corners((1000, 750 - height / 2), width, 0.0)
    lower_corners = rotated_rectangle_corners((1000 + 60, 750 + height / 2 + 2), width, 0.0)
    detections = detect_checks_classical(build_scene([upper_corners, lower_corners]), ClassicalDetectorConfig())
    assert len(detections) == 2
    for true_corners in (upper_corners, lower_corners):
        assert minimum_cyclic_corner_error(match_detection_to(true_corners, detections), true_corners) < CORNER_TOLERANCE_PIXELS


def test_check_partly_out_of_frame_is_found_as_its_visible_part():
    true_corners = rotated_rectangle_corners((1800, 600), 900, 3.0)  # right third leaves the 2000 px frame
    detections = detect_checks_classical(build_scene([true_corners]), ClassicalDetectorConfig())
    assert len(detections) == 1
    visible_truth = Polygon(true_corners).intersection(box(0, 0, 2000, 1500))
    predicted = Polygon(detections[0].corners).intersection(box(0, 0, 2000, 1500))
    assert predicted.intersection(visible_truth).area / predicted.union(visible_truth).area > 0.9


def test_empty_background_yields_nothing():
    assert detect_checks_classical(build_scene([], textured=True), ClassicalDetectorConfig()) == []
