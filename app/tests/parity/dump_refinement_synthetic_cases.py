"""Render the synthetic refinement test scenes and their Python results, for the JS runner.

    PYTHONPATH=. <venv>/bin/python app/tests/parity/dump_refinement_synthetic_cases.py

Mirrors experiments/detection/tests/test_refinement_synthetic_checks.py case for case (bright
and dark checks, white on white, occlusion, a side off the image, permuted/reversed corner
order, greyscale input, another detection over a corner, contact shadow only): these reach
branches real scenes rarely hit. Writes `<output>/<case>.pixels` (uint8, H x W x C) and
`<output>/synthetic_cases.json` (shape, input corners, other quads, true corners, Python
refined corners and the test's tolerance).
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from experiments.detection.refinement.detector_input_simulation import simulate_jittered_corners, simulate_oriented_box_corners
from experiments.detection.refinement.quadrilateral_refinement import refine_check_quadrilateral
from experiments.detection.tests.refinement_test_scenes import (
    PERSPECTIVE_CHECK_CORNERS,
    SCENE_WIDTH,
    draw_check_with_distractors,
    fill_quad_supersampled,
    inset_quad_by_pixels,
    make_textured_background,
    render_scene,
)

DEFAULT_OUTPUT_DIRECTORY = Path("/tmp/cts-parity/refinement-synthetic")
SUB_PIXEL_TOLERANCE_PIXELS = 0.5
LOW_CONTRAST_TOLERANCE_PIXELS = 0.75
SCENE_HEIGHT = 480


def to_uint8(canvas: np.ndarray) -> np.ndarray:
    """Round a float canvas to the uint8 image the tests refine."""
    return np.clip(np.round(canvas), 0, 255).astype(np.uint8)


def build_cases() -> list[dict]:
    """One dict per test scenario: image, input corners, other quads, truth, tolerance."""
    oriented_box = simulate_oriented_box_corners(PERSPECTIVE_CHECK_CORNERS)
    bright = render_scene(paper_bgr=(225, 232, 236), background_level=90, texture_amplitude=18)
    cases = [{"name": "bright_on_dark_obb", "image": bright, "corners": oriented_box, "truth": PERSPECTIVE_CHECK_CORNERS, "tolerance": SUB_PIXEL_TOLERANCE_PIXELS}]
    for seed in (0, 1, 2):
        cases.append({
            "name": f"dark_on_light_jitter_seed{seed}",
            "image": render_scene(paper_bgr=(120, 135, 150), background_level=215, texture_amplitude=8, seed=seed, ink_bgr=(30, 30, 35)),
            "corners": simulate_jittered_corners(PERSPECTIVE_CHECK_CORNERS, 8.0, np.random.default_rng(seed)),
            "truth": PERSPECTIVE_CHECK_CORNERS, "tolerance": SUB_PIXEL_TOLERANCE_PIXELS,
        })
    cases.append({"name": "white_on_white", "image": render_scene(paper_bgr=(238, 240, 242), background_level=222, texture_amplitude=5),
                  "corners": oriented_box, "truth": PERSPECTIVE_CHECK_CORNERS, "tolerance": LOW_CONTRAST_TOLERANCE_PIXELS})
    canvas = draw_check_with_distractors(make_textured_background(SCENE_WIDTH, SCENE_HEIGHT, 90, 18, seed=3), PERSPECTIVE_CHECK_CORNERS, (225, 232, 236), (60, 60, 70))
    canvas = fill_quad_supersampled(canvas, np.array([[300.0, 300.0], [480.0, 290.0], [490.0, 460.0], [310.0, 470.0]]), (200, 225, 215))
    cases.append({"name": "partly_occluded_side", "image": to_uint8(canvas), "corners": oriented_box, "truth": PERSPECTIVE_CHECK_CORNERS, "tolerance": LOW_CONTRAST_TOLERANCE_PIXELS})
    cases.append({"name": "side_off_the_image", "image": np.ascontiguousarray(bright[:, :600]),
                  "corners": simulate_jittered_corners(PERSPECTIVE_CHECK_CORNERS, 6.0, np.random.default_rng(4)), "truth": None, "tolerance": None})
    for label, permutation in (("rotated_order", [1, 2, 3, 0]), ("reversed_winding", [3, 2, 1, 0])):
        cases.append({"name": label, "image": bright, "corners": oriented_box[permutation], "truth": PERSPECTIVE_CHECK_CORNERS[permutation], "tolerance": SUB_PIXEL_TOLERANCE_PIXELS})
    cases.append({"name": "greyscale_input", "image": cv2.cvtColor(bright, cv2.COLOR_BGR2GRAY), "corners": oriented_box, "truth": PERSPECTIVE_CHECK_CORNERS, "tolerance": SUB_PIXEL_TOLERANCE_PIXELS})
    covering_check = np.array([[520.0, 300.0], [740.0, 290.0], [750.0, 470.0], [530.0, 475.0]])
    canvas = draw_check_with_distractors(make_textured_background(SCENE_WIDTH, SCENE_HEIGHT, 90, 18, seed=5), PERSPECTIVE_CHECK_CORNERS, (225, 232, 236), (60, 60, 70))
    canvas = draw_check_with_distractors(canvas, covering_check, (215, 236, 222), (60, 60, 70))
    cases.append({"name": "other_detection_over_corner", "image": to_uint8(canvas), "corners": oriented_box, "other_quads": [covering_check], "truth": PERSPECTIVE_CHECK_CORNERS, "tolerance": 1.0})
    canvas = make_textured_background(SCENE_WIDTH, SCENE_HEIGHT, 232, 3, seed=6)
    canvas = fill_quad_supersampled(canvas, inset_quad_by_pixels(PERSPECTIVE_CHECK_CORNERS, -1.0), (200, 200, 200))
    canvas = draw_check_with_distractors(canvas, inset_quad_by_pixels(PERSPECTIVE_CHECK_CORNERS, 1.0), (226, 228, 230), (60, 60, 70))
    cases.append({"name": "contact_shadow_only", "image": to_uint8(canvas),
                  "corners": simulate_jittered_corners(PERSPECTIVE_CHECK_CORNERS, 3.0, np.random.default_rng(6)), "truth": PERSPECTIVE_CHECK_CORNERS, "tolerance": 1.0})
    return cases


def main() -> None:
    """Render, refine with Python, and write every case."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    records = []
    for case in build_cases():
        other_quads = case.get("other_quads", [])
        refined, diagnostics = refine_check_quadrilateral(case["image"], case["corners"], None, other_quads)
        (arguments.output / f"{case['name']}.pixels").write_bytes(case["image"].tobytes())
        records.append({
            "name": case["name"],
            "height": case["image"].shape[0], "width": case["image"].shape[1], "channels": 1 if case["image"].ndim == 2 else case["image"].shape[2],
            "corners": case["corners"].tolist(), "other_quads": [quad.tolist() for quad in other_quads],
            "truth": None if case["truth"] is None else case["truth"].tolist(), "tolerance": case["tolerance"],
            "python_refined": refined.tolist(), "python_sides_without_support": diagnostics["passes"][0]["sides_without_support"],
        })
        print(case["name"], "python max GT error:", None if case["truth"] is None else round(float(np.linalg.norm(refined - case["truth"], axis=1).max()), 3))
    (arguments.output / "synthetic_cases.json").write_text(json.dumps(records))


if __name__ == "__main__":
    main()
