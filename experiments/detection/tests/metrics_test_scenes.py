"""Hand-built scenes for the metrics tests (no dataset needed).

Image is 1000 x 800. Checks are axis-aligned 300 x 120 rectangles (short side 120 px),
corners TL/TR/BR/BL clockwise on screen; the outline is the corner quad densified with
edge midpoints, so outline IoU equals quad IoU for these undeformed checks.
"""

from pathlib import Path

import numpy as np

from experiments.detection.dataset.scene_annotations import CheckAnnotation, SceneAnnotation

TEST_IMAGE_WIDTH = 1000
TEST_IMAGE_HEIGHT = 800
TEST_CHECK_WIDTH = 300.0
TEST_CHECK_HEIGHT = 120.0


def rectangle_corners(left: float, top: float, width: float = TEST_CHECK_WIDTH, height: float = TEST_CHECK_HEIGHT) -> np.ndarray:
    """TL, TR, BR, BL corners of an axis-aligned rectangle."""
    return np.array([[left, top], [left + width, top], [left + width, top + height], [left, top + height]], dtype=np.float64)


def outline_from_corners(corners: np.ndarray) -> np.ndarray:
    """Corners with edge midpoints inserted, starting at corner 0 like the real labels."""
    outline_points = []
    for corner_index in range(4):
        outline_points.append(corners[corner_index])
        outline_points.append(0.5 * (corners[corner_index] + corners[(corner_index + 1) % 4]))
    return np.array(outline_points)


def make_check(check_index: int, corners: np.ndarray, deformation_kinds: tuple[str, ...] = ()) -> CheckAnnotation:
    """CheckAnnotation for a rectangle; in-frame flags derived from the corners."""
    inside = (
        (corners[:, 0] >= 0) & (corners[:, 0] <= TEST_IMAGE_WIDTH) & (corners[:, 1] >= 0) & (corners[:, 1] <= TEST_IMAGE_HEIGHT)
    )
    return CheckAnnotation(
        check_index=check_index,
        corners=corners,
        outline=outline_from_corners(corners),
        orientation_class=0,
        rotation_degrees_clockwise=0.0,
        fully_in_frame=bool(inside.all()),
        in_frame_fraction=1.0 if inside.all() else 0.5,
        visible_fraction=1.0,
        size_kind="personal",
        deformation_kinds=deformation_kinds,
    )


def make_scene(scene_id: str, checks: list[CheckAnnotation], layout_mode: str = "grid") -> SceneAnnotation:
    """SceneAnnotation on a fake wood background."""
    return SceneAnnotation(
        scene_id=scene_id,
        split_name="val",
        image_path=Path(f"/nonexistent/{scene_id}.jpg"),
        image_width=TEST_IMAGE_WIDTH,
        image_height=TEST_IMAGE_HEIGHT,
        background_id="web/test_wood.jpg",
        background_category="wood",
        background_source="web",
        layout_mode=layout_mode,
        cast_shadow_kind="none",
        checks=checks,
    )


def two_check_scene(scene_id: str = "scene_two") -> SceneAnnotation:
    """Two well-separated in-frame checks."""
    return make_scene(scene_id, [make_check(0, rectangle_corners(100, 100)), make_check(1, rectangle_corners(500, 500))])
