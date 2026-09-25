"""Export check polygons for off-the-shelf detectors: COCO (with corner keypoints) and YOLO.

- COCO: one `annotations_coco.json` per split. Each check is its outline (the deformed paper
  edge, contract C3) clipped to the photo, plus 4 keypoints (the check's own TL, TR, BR, BL) so corner/orientation models
  can train directly. Keypoints outside the photo get v=0 and (0, 0).
- YOLO segmentation (Ultralytics layout): `<split>/labels/<scene>.txt`, class 0, the clipped
  outline normalized to [0, 1].
- YOLO OBB: `<split>/labels_obb/<scene>.txt`, class 0, the 4 corners normalized and clamped to [0, 1].
"""

import json
from pathlib import Path

import numpy as np

from synth.compose.perspective import clip_polygon_to_rect, polygon_area

CHECK_CATEGORY_ID = 1
CORNER_KEYPOINT_NAMES = ["top_left", "top_right", "bottom_right", "bottom_left"]
MIN_CLIPPED_AREA_PX = 16.0


def clipped_check_polygon(points: list[list[float]], width: int, height: int) -> np.ndarray:
    """A check polygon (outline or corners) clipped to the photo."""
    return clip_polygon_to_rect(np.asarray(points, np.float64), width, height)


def check_outline(check: dict) -> list[list[float]]:
    """The paper outline; labels written before contract C3 only have the 4 corners."""
    return check.get("outline") or check["corners"]


def yolo_segmentation_lines(scene_label: dict) -> list[str]:
    """One Ultralytics segmentation line per visible check."""
    width, height = scene_label["image_width"], scene_label["image_height"]
    lines = []
    for check in scene_label["checks"]:
        polygon = clipped_check_polygon(check_outline(check), width, height)
        if len(polygon) < 3 or polygon_area(polygon) < MIN_CLIPPED_AREA_PX:
            continue
        normalized = (polygon / [width, height]).clip(0, 1)
        lines.append("0 " + " ".join(f"{value:.6f}" for value in normalized.ravel()))
    return lines


def yolo_obb_lines(scene_label: dict) -> list[str]:
    """One Ultralytics OBB line (4 corners) per visible check."""
    width, height = scene_label["image_width"], scene_label["image_height"]
    lines = []
    for check in scene_label["checks"]:
        polygon = clipped_check_polygon(check_outline(check), width, height)
        if len(polygon) < 3 or polygon_area(polygon) < MIN_CLIPPED_AREA_PX:
            continue
        normalized = (np.asarray(check["corners"]) / [width, height]).clip(0, 1)
        lines.append("0 " + " ".join(f"{value:.6f}" for value in normalized.ravel()))
    return lines


def write_yolo_labels(scene_label: dict, split_directory: Path) -> None:
    """Write both YOLO label files for one scene."""
    stem = Path(scene_label["image_file"]).stem
    (split_directory / "labels" / f"{stem}.txt").write_text("\n".join(yolo_segmentation_lines(scene_label)) + "\n")
    (split_directory / "labels_obb" / f"{stem}.txt").write_text("\n".join(yolo_obb_lines(scene_label)) + "\n")


def build_coco_document(scene_labels: list[dict]) -> dict:
    """COCO instances + keypoints document for a list of scene labels."""
    images, annotations = [], []
    for image_id, scene in enumerate(scene_labels, start=1):
        width, height = scene["image_width"], scene["image_height"]
        images.append({"id": image_id, "file_name": f"images/{scene['image_file']}", "width": width, "height": height})
        for check in scene["checks"]:
            polygon = clipped_check_polygon(check_outline(check), width, height)
            area = polygon_area(polygon)
            if len(polygon) < 3 or area < MIN_CLIPPED_AREA_PX:
                continue
            keypoints = []
            for x, y in check["corners"]:
                inside = 0 <= x <= width and 0 <= y <= height
                keypoints += [round(x, 2), round(y, 2), 2] if inside else [0, 0, 0]
            x0, y0 = polygon.min(axis=0)
            x1, y1 = polygon.max(axis=0)
            annotations.append({
                "id": len(annotations) + 1, "image_id": image_id, "category_id": CHECK_CATEGORY_ID,
                "segmentation": [[round(float(v), 2) for v in polygon.ravel()]],
                "area": round(area, 2), "bbox": [round(float(v), 2) for v in (x0, y0, x1 - x0, y1 - y0)], "iscrowd": 0,
                "keypoints": keypoints, "num_keypoints": sum(1 for v in keypoints[2::3] if v > 0),
                "orientation_class": check["orientation_class"], "template_id": check["template_id"],
                "visible_fraction": check["visible_fraction"],
            })
    return {
        "info": {"description": "Check Transcriber synthetic scenes"},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": CHECK_CATEGORY_ID, "name": "check", "supercategory": "document",
                        "keypoints": CORNER_KEYPOINT_NAMES, "skeleton": [[1, 2], [2, 3], [3, 4], [4, 1]]}],
    }


def write_coco_for_split(split_directory: Path) -> Path:
    """Gather every scene label in a split and write `annotations_coco.json`."""
    scene_labels = [json.loads(path.read_text()) for path in sorted((split_directory / "annotations").glob("*.json"))]
    coco_path = split_directory / "annotations_coco.json"
    coco_path.write_text(json.dumps(build_coco_document(scene_labels)))
    return coco_path


def write_yolo_data_yaml(output_directory: Path) -> Path:
    """Ultralytics dataset config pointing at the three splits."""
    yaml_path = output_directory / "data.yaml"
    yaml_path.write_text(f"path: {output_directory}\ntrain: train/images\nval: val/images\ntest: eval/images\nnames:\n  0: check\n")
    return yaml_path
