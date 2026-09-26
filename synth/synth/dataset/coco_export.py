"""COCO export: check instances with corner keypoints, plus field instances with their text.

One `annotations_coco.json` per split (image `file_name` is relative to the split directory).

- Category 1 `check`: segmentation = the paper outline (contract C3) clipped to the photo;
  keypoints = the 4 physical corners in the check's own order TL, TR, BR, BL (v=2 in frame,
  v=0 and (0, 0) outside); `attributes.orientation_class` (0/90/180/270) and exact rotation.
- Categories 2+ (one per field name, supercategory `field`): segmentation = the field quad
  clipped to the photo, `attributes.text` = ground truth, `attributes.handwritten`,
  `attributes.check_annotation_id` = the parent check. Fields fully out of frame are omitted.
- Coordinates: scene labels put pixel i's centre at i; COCO and YOLO put it at i + 0.5 (pixel i
  spans [i, i + 1]). `to_pixel_edge_coordinates` shifts every exported point by half a pixel.
- Amodal: a check covered by another keeps its full outline, and its in-frame corners are v=2
  even when hidden under a neighbour (v means in frame, not unoccluded).
"""

import json
from pathlib import Path

import numpy as np

from synth.compose.perspective import clip_polygon_to_rect, polygon_area
from synth.render.check_fields import FieldName

CHECK_CATEGORY_ID = 1
FIELD_CATEGORY_IDS = {field_name.value: index for index, field_name in enumerate(FieldName, start=2)}
CORNER_KEYPOINT_NAMES = ["top_left", "top_right", "bottom_right", "bottom_left"]
MIN_CLIPPED_AREA_PX = 16.0
COORDINATE_DECIMALS = 2
PIXEL_CENTRE_TO_EDGE_OFFSET = 0.5


def to_pixel_edge_coordinates(points: list[list[float]] | np.ndarray) -> np.ndarray:
    """Scene-label points (pixel centre at i) -> COCO/YOLO points (pixel i spans [i, i + 1])."""
    return np.asarray(points, np.float64) + PIXEL_CENTRE_TO_EDGE_OFFSET


def check_outline(check: dict) -> list[list[float]]:
    """The paper outline; labels written before contract C3 only have the 4 corners."""
    return check.get("outline") or check["corners"]


def polygon_instance(points: list[list[float]], width: int, height: int) -> dict | None:
    """COCO segmentation/area/bbox for a scene-label polygon clipped to the photo; None if (almost) nothing is inside."""
    polygon = clip_polygon_to_rect(to_pixel_edge_coordinates(points), width, height)
    area = polygon_area(polygon) if len(polygon) >= 3 else 0.0
    if area < MIN_CLIPPED_AREA_PX:
        return None
    x0, y0 = polygon.min(axis=0)
    x1, y1 = polygon.max(axis=0)
    return {"segmentation": [[round(float(v), COORDINATE_DECIMALS) for v in polygon.ravel()]],
            "area": round(area, COORDINATE_DECIMALS),
            "bbox": [round(float(v), COORDINATE_DECIMALS) for v in (x0, y0, x1 - x0, y1 - y0)], "iscrowd": 0}


def corner_keypoints(corners: list[list[float]], width: int, height: int) -> list[float]:
    """Flat COCO keypoint list for the 4 corners (v=2 in frame, even if covered; v=0 and (0, 0) outside)."""
    keypoints = []
    for x, y in to_pixel_edge_coordinates(corners).tolist():
        inside = 0 <= x <= width and 0 <= y <= height
        keypoints += [round(x, COORDINATE_DECIMALS), round(y, COORDINATE_DECIMALS), 2] if inside else [0, 0, 0]
    return keypoints


def coco_categories() -> list[dict]:
    """The check category (with keypoints) followed by one category per field name."""
    check_category = {"id": CHECK_CATEGORY_ID, "name": "check", "supercategory": "document",
                      "keypoints": CORNER_KEYPOINT_NAMES, "skeleton": [[1, 2], [2, 3], [3, 4], [4, 1]]}
    field_categories = [{"id": category_id, "name": field_name, "supercategory": "field"}
                        for field_name, category_id in FIELD_CATEGORY_IDS.items()]
    return [check_category, *field_categories]


def build_coco_document(scene_labels: list[dict]) -> dict:
    """COCO instances + keypoints document for a list of scene labels."""
    images, annotations = [], []
    for image_id, scene in enumerate(scene_labels, start=1):
        width, height = scene["image_width"], scene["image_height"]
        images.append({"id": image_id, "file_name": f"images/{scene['image_file']}", "width": width, "height": height,
                       "scene_id": scene["scene_id"]})
        for check in scene["checks"]:
            instance = polygon_instance(check_outline(check), width, height)
            if instance is None:
                continue
            keypoints = corner_keypoints(check["corners"], width, height)
            check_annotation_id = len(annotations) + 1
            annotations.append({
                "id": check_annotation_id, "image_id": image_id, "category_id": CHECK_CATEGORY_ID, **instance,
                "keypoints": keypoints, "num_keypoints": sum(1 for v in keypoints[2::3] if v > 0),
                "attributes": {"orientation_class": check["orientation_class"],
                               "rotation_degrees_clockwise": check["rotation_degrees_clockwise"],
                               "template_id": check["template_id"],
                               "layout_family": check["canonical"].get("layout_family"),
                               "visible_fraction": check["visible_fraction"], "check_index": check["check_index"]},
            })
            for field in check["fields"]:
                field_instance = polygon_instance(field["quad"], width, height)
                if field_instance is None or field["field_name"] not in FIELD_CATEGORY_IDS:
                    continue
                annotations.append({
                    "id": len(annotations) + 1, "image_id": image_id,
                    "category_id": FIELD_CATEGORY_IDS[field["field_name"]], **field_instance,
                    "attributes": {"text": field["text"], "handwritten": field["handwritten"],
                                   "visible_fraction": field["visible_fraction"],
                                   "check_annotation_id": check_annotation_id},
                })
    return {"info": {"description": "Check Transcriber synthetic scenes"}, "images": images,
            "annotations": annotations, "categories": coco_categories()}


def write_coco_for_split(split_directory: Path) -> Path:
    """Gather every scene label in a split and write `annotations_coco.json`."""
    scene_labels = [json.loads(path.read_text()) for path in sorted((split_directory / "annotations").glob("*.json"))]
    coco_path = split_directory / "annotations_coco.json"
    coco_path.write_text(json.dumps(build_coco_document(scene_labels)))
    return coco_path
