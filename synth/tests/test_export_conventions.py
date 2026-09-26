"""Export conventions: half-pixel shift, OBB for partly out-of-frame checks, OCR legibility gate."""

import numpy as np

from synth.compose.perspective import clip_polygon_to_rect
from synth.dataset.annotation_exports import passes_ultralytics_label_check, yolo_obb_lines
from synth.dataset.coco_export import build_coco_document
from synth.dataset.ocr_manifest import MIN_LEGIBLE_TEXT_HEIGHT_PHOTO_PX, field_status

IMAGE_WIDTH, IMAGE_HEIGHT = 400, 300


def scene_with_one_check(corners: list[list[float]]) -> dict:
    """A minimal scene label: one check with the given corners (no outline, no fields)."""
    check = {"corners": corners, "fields": [], "orientation_class": 0, "rotation_degrees_clockwise": 0.0,
             "template_id": "tpl_000", "canonical": {}, "visible_fraction": 1.0, "check_index": 0}
    return {"scene_id": "s", "image_file": "s.jpg", "image_width": IMAGE_WIDTH, "image_height": IMAGE_HEIGHT,
            "checks": [check]}


def rotated_rectangle(centre: tuple[float, float], size: tuple[float, float], degrees: float) -> list[list[float]]:
    """TL TR BR BL of a width x height rectangle rotated clockwise (image y down) about `centre`."""
    angle = np.deg2rad(degrees)
    along, down = np.array([np.cos(angle), np.sin(angle)]), np.array([-np.sin(angle), np.cos(angle)])
    half_width, half_height = size[0] / 2, size[1] / 2
    offsets = [(-half_width, -half_height), (half_width, -half_height), (half_width, half_height), (-half_width, half_height)]
    return [(np.array(centre) + a * along + d * down).tolist() for a, d in offsets]


def test_coco_points_are_shifted_from_pixel_centres_to_pixel_edges():
    coco = build_coco_document([scene_with_one_check([[10, 20], [110, 20], [110, 70], [10, 70]])])
    check = coco["annotations"][0]
    assert check["bbox"] == [10.5, 20.5, 100.0, 50.0]
    assert check["keypoints"][:3] == [10.5, 20.5, 2]


def test_partly_out_of_frame_obb_is_a_rectangle_in_check_order_not_a_trapezoid():
    corners = rotated_rectangle((40, 150), (240, 110), 20)   # left third hangs off the frame
    line = yolo_obb_lines(scene_with_one_check(corners))[0].split()
    normalized = np.array([float(v) for v in line[1:]]).reshape(4, 2)
    points = normalized * [IMAGE_WIDTH, IMAGE_HEIGHT]
    edges = [points[(i + 1) % 4] - points[i] for i in range(4)]
    for first, second in zip(edges, edges[1:] + edges[:1]):
        assert abs(np.dot(first, second)) / (np.linalg.norm(first) * np.linalg.norm(second)) < 1e-4, "not a right angle"
    top_direction = edges[0] / np.linalg.norm(edges[0])
    assert np.isclose(np.degrees(np.arctan2(top_direction[1], top_direction[0])), 20, atol=0.01), "lost the check's rotation"
    visible = clip_polygon_to_rect(np.array(corners) + 0.5, IMAGE_WIDTH, IMAGE_HEIGHT)
    along, down = top_direction, np.array([-top_direction[1], top_direction[0]])
    for axis in (along, down):   # the rectangle's extent on its own axes equals the visible part's
        assert np.isclose((points @ axis).min(), (visible @ axis).min(), atol=1e-3)
        assert np.isclose((points @ axis).max(), (visible @ axis).max(), atol=1e-3)
    assert passes_ultralytics_label_check(normalized)


def test_in_frame_obb_keeps_the_true_corners():
    corners = rotated_rectangle((200, 150), (200, 90), -15)
    line = yolo_obb_lines(scene_with_one_check(corners))[0].split()
    points = np.array([float(v) for v in line[1:]]).reshape(4, 2) * [IMAGE_WIDTH, IMAGE_HEIGHT]
    assert np.allclose(points, np.array(corners) + 0.5, atol=1e-3)


def field_with_height(height: float) -> dict:
    """A fully in-frame, uncovered field quad `height` px tall."""
    quad = [[100, 100], [220, 100], [220, 100 + height], [100, 100 + height]]
    return {"field_name": "payee", "quad": quad, "bbox_clipped": [100, 100, 220, 100 + height], "visible_fraction": 1.0}


def test_fields_below_the_legibility_floor_are_not_usable():
    assert field_status(field_with_height(MIN_LEGIBLE_TEXT_HEIGHT_PHOTO_PX - 2), IMAGE_WIDTH, IMAGE_HEIGHT) == "too_small"
    assert field_status(field_with_height(MIN_LEGIBLE_TEXT_HEIGHT_PHOTO_PX + 2), IMAGE_WIDTH, IMAGE_HEIGHT) == "ok"
