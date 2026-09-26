"""Corner refinement on synthetic checks with exactly known corners.

Each test starts from what a detector would give (the oriented box of the true corners,
or jittered corners) and asserts the refined corners land within about a pixel.
"""

import cv2
import numpy as np
import pytest

from experiments.detection.predictions.detected_check import DetectedCheck
from experiments.detection.refinement.detector_input_simulation import (
    compute_corner_errors,
    simulate_jittered_corners,
    simulate_oriented_box_corners,
)
from experiments.detection.refinement.quadrilateral_refinement import (
    refine_check_quadrilateral,
    refine_detected_checks,
)
from experiments.detection.tests.refinement_test_scenes import (
    PERSPECTIVE_CHECK_CORNERS,
    SCENE_WIDTH,
    fill_quad_supersampled,
    make_textured_background,
    draw_check_with_distractors,
    inset_quad_by_pixels,
    render_scene,
)

SUB_PIXEL_TOLERANCE_PIXELS = 0.5
LOW_CONTRAST_TOLERANCE_PIXELS = 0.75


def test_bright_check_on_dark_texture_from_oriented_box():
    image = render_scene(paper_bgr=(225, 232, 236), background_level=90, texture_amplitude=18)
    approximate = simulate_oriented_box_corners(PERSPECTIVE_CHECK_CORNERS)
    assert compute_corner_errors(approximate, PERSPECTIVE_CHECK_CORNERS).max() > 5  # keystone the box cannot model
    refined, diagnostics = refine_check_quadrilateral(image, approximate)
    assert compute_corner_errors(refined, PERSPECTIVE_CHECK_CORNERS).max() < SUB_PIXEL_TOLERANCE_PIXELS
    assert not diagnostics["quad_reverted"]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_dark_check_on_light_background_from_jitter(seed):
    image = render_scene(paper_bgr=(120, 135, 150), background_level=215, texture_amplitude=8, seed=seed, ink_bgr=(30, 30, 35))
    approximate = simulate_jittered_corners(PERSPECTIVE_CHECK_CORNERS, 8.0, np.random.default_rng(seed))
    refined, _ = refine_check_quadrilateral(image, approximate)
    assert compute_corner_errors(refined, PERSPECTIVE_CHECK_CORNERS).max() < SUB_PIXEL_TOLERANCE_PIXELS


def test_low_contrast_white_on_white():
    image = render_scene(paper_bgr=(238, 240, 242), background_level=222, texture_amplitude=5)
    refined, _ = refine_check_quadrilateral(image, simulate_oriented_box_corners(PERSPECTIVE_CHECK_CORNERS))
    assert compute_corner_errors(refined, PERSPECTIVE_CHECK_CORNERS).max() < LOW_CONTRAST_TOLERANCE_PIXELS


def test_partly_occluded_side():
    canvas = make_textured_background(SCENE_WIDTH, 480, 90, 18, seed=3)
    canvas = draw_check_with_distractors(canvas, PERSPECTIVE_CHECK_CORNERS, (225, 232, 236), (60, 60, 70))
    # A second, differently coloured check lies across the middle 35% of the bottom side.
    occluder = np.array([[300.0, 300.0], [480.0, 290.0], [490.0, 460.0], [310.0, 470.0]])
    canvas = fill_quad_supersampled(canvas, occluder, (200, 225, 215))
    image = np.clip(np.round(canvas), 0, 255).astype(np.uint8)
    refined, _ = refine_check_quadrilateral(image, simulate_oriented_box_corners(PERSPECTIVE_CHECK_CORNERS))
    assert compute_corner_errors(refined, PERSPECTIVE_CHECK_CORNERS).max() < LOW_CONTRAST_TOLERANCE_PIXELS


def test_side_off_the_image_edge_keeps_its_line():
    image = render_scene(paper_bgr=(225, 232, 236), background_level=90, texture_amplitude=18)
    cut_width = 600  # the check's right side (x ~ 632-649) now lies outside the frame
    cropped_image = np.ascontiguousarray(image[:, :cut_width])
    approximate = simulate_jittered_corners(PERSPECTIVE_CHECK_CORNERS, 6.0, np.random.default_rng(4))
    refined, diagnostics = refine_check_quadrilateral(cropped_image, approximate)
    errors = compute_corner_errors(refined, PERSPECTIVE_CHECK_CORNERS)
    # Left corners are fully observed.
    assert errors[[0, 3]].max() < SUB_PIXEL_TOLERANCE_PIXELS
    # Right corners stay on the input's right side line (nothing to see there).
    right_direction = approximate[2] - approximate[1]
    right_normal = np.array([-right_direction[1], right_direction[0]]) / np.linalg.norm(right_direction)
    for corner_index in (1, 2):
        assert abs(np.dot(refined[corner_index] - approximate[1], right_normal)) < 0.5
    assert diagnostics["passes"][0]["sides_without_support"][1]


def test_corner_order_and_winding_are_preserved():
    image = render_scene(paper_bgr=(225, 232, 236), background_level=90, texture_amplitude=18)
    approximate = simulate_oriented_box_corners(PERSPECTIVE_CHECK_CORNERS)
    for permutation in ([1, 2, 3, 0], [3, 2, 1, 0]):
        refined, _ = refine_check_quadrilateral(image, approximate[permutation])
        assert compute_corner_errors(refined, PERSPECTIVE_CHECK_CORNERS[permutation]).max() < SUB_PIXEL_TOLERANCE_PIXELS


def test_refine_detected_checks_keeps_score_and_orientation():
    image = render_scene(paper_bgr=(225, 232, 236), background_level=90, texture_amplitude=18)
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detection = DetectedCheck(corners=simulate_oriented_box_corners(PERSPECTIVE_CHECK_CORNERS), score=0.87, orientation_known=True)
    for input_image in (image, grey_image):
        (refined_detection,) = refine_detected_checks(input_image, [detection])
        assert refined_detection.score == 0.87 and refined_detection.orientation_known
        assert compute_corner_errors(refined_detection.corners, PERSPECTIVE_CHECK_CORNERS).max() < SUB_PIXEL_TOLERANCE_PIXELS
        assert "refinement" in refined_detection.extras


def test_other_detection_over_a_corner_does_not_capture_the_side():
    canvas = make_textured_background(SCENE_WIDTH, 480, 90, 18, seed=5)
    canvas = draw_check_with_distractors(canvas, PERSPECTIVE_CHECK_CORNERS, (225, 232, 236), (60, 60, 70))
    # A second check lies over the bottom-right corner; its detection is passed in.
    covering_check = np.array([[520.0, 300.0], [740.0, 290.0], [750.0, 470.0], [530.0, 475.0]])
    canvas = draw_check_with_distractors(canvas, covering_check, (215, 236, 222), (60, 60, 70))
    image = np.clip(np.round(canvas), 0, 255).astype(np.uint8)
    approximate = simulate_oriented_box_corners(PERSPECTIVE_CHECK_CORNERS)
    refined, _ = refine_check_quadrilateral(image, approximate, other_check_quads=[covering_check])
    # The hidden corner is extrapolated from the visible, straight stretches of its sides.
    assert compute_corner_errors(refined, PERSPECTIVE_CHECK_CORNERS).max() < 1.0


def test_white_on_white_with_only_a_contact_shadow():
    canvas = make_textured_background(SCENE_WIDTH, 480, 232, 3, seed=6)
    # The only cue: a ~2 px dark contact shadow straddling the paper edge.
    canvas = fill_quad_supersampled(canvas, inset_quad_by_pixels(PERSPECTIVE_CHECK_CORNERS, -1.0), (200, 200, 200))
    canvas = draw_check_with_distractors(canvas, inset_quad_by_pixels(PERSPECTIVE_CHECK_CORNERS, 1.0), (226, 228, 230), (60, 60, 70))
    image = np.clip(np.round(canvas), 0, 255).astype(np.uint8)
    refined, _ = refine_check_quadrilateral(image, simulate_jittered_corners(PERSPECTIVE_CHECK_CORNERS, 3.0, np.random.default_rng(6)))
    assert compute_corner_errors(refined, PERSPECTIVE_CHECK_CORNERS).max() < 1.0
