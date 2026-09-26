"""Cut the train split into the same OCR layout synth v1 ships for val/eval.

synth v1 only exported rectified check crops and per-field crops for val and eval. Training a
recognizer or a field localizer needs the same view of train, so this re-derives it from the
train scene photos and annotations, reproducing synth's `ocr_manifest.py` / `ocr_rectify.py`
arithmetic (ground-truth corners, 1600 px wide, physical aspect, 15% field margin). Those modules
are re-implemented here rather than imported because the synth package is being restructured.

Output under TRAIN_CROPS_ROOT:
    ocr/train/checks/<scene>__check=<i>.jpg
    ocr/train/fields/<scene>__check=<i>__field=<name>.png     target + auxiliary printed fields only
    ocr/train/rows/<scene>.jsonl                               per-scene resume cache
    ocr/ocr_fields__split=train.jsonl

Run: python -m experiments.field_reading.data_access.train_crop_export --workers 4
"""

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from experiments.field_reading.config import (AUXILIARY_PRINTED_FIELD_NAMES, SYNTH_V1_ROOT, TARGET_FIELD_NAMES,
                                              TRAIN_CROPS_ROOT)

logger = logging.getLogger(__name__)

RECTIFIED_CHECK_WIDTH_PX = 1600
CHECK_SIZE_INCHES = {"personal": (6.0, 2.75), "business": (8.5, 3.5), "money_order": (7.0, 3.125)}
FIELD_CROP_MARGIN_FRACTION = 0.15
FIELD_CROP_MIN_MARGIN_PX = 6
PIXEL_CENTRE_TO_EDGE_OFFSET = 0.5
CHECK_CROP_JPEG_QUALITY = 95
FIELD_IN_FRAME_MIN_FRACTION = 0.999
FIELD_UNOCCLUDED_MIN_VISIBLE_FRACTION = 0.99
MIN_LEGIBLE_TEXT_HEIGHT_PHOTO_PX = 14.0
EXPORTED_FIELD_NAMES = set(TARGET_FIELD_NAMES) | set(AUXILIARY_PRINTED_FIELD_NAMES)


def polygon_area(polygon: np.ndarray) -> float:
    """Shoelace area of a simple polygon."""
    if len(polygon) < 3:
        return 0.0
    x, y = polygon[:, 0], polygon[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def clip_polygon_to_rect(polygon: np.ndarray, width: float, height: float) -> np.ndarray:
    """Sutherland-Hodgman clip of a polygon to [0, width] x [0, height]."""
    points = [np.asarray(p, np.float64) for p in polygon]
    for axis, value, keep_greater in ((0, 0.0, True), (0, width, False), (1, 0.0, True), (1, height, False)):
        if not points:
            break
        inside = (lambda p, a=axis, v=value, g=keep_greater: p[a] >= v if g else p[a] <= v)
        output = []
        for index in range(len(points)):
            current, previous = points[index], points[index - 1]
            if inside(current):
                if not inside(previous):
                    t = (value - previous[axis]) / (current[axis] - previous[axis])
                    output.append(previous + t * (current - previous))
                output.append(current)
            elif inside(previous):
                t = (value - previous[axis]) / (current[axis] - previous[axis])
                output.append(previous + t * (current - previous))
        points = output
    return np.array(points).reshape(-1, 2)


def field_text_height_in_photo_px(quad: list[list[float]]) -> float:
    """Mean of the field quad's left and right edge lengths in the photo."""
    quad_array = np.asarray(quad, np.float64)
    return round(float((np.linalg.norm(quad_array[3] - quad_array[0]) + np.linalg.norm(quad_array[2] - quad_array[1])) / 2), 1)


def field_status(field: dict, photo_width: int, photo_height: int) -> str:
    """Same classification as synth's ocr_manifest.field_status."""
    if field["field_name"] == "micr":
        return "micr_blurred_in_app"
    if field["bbox_clipped"] is None:
        return "not_in_frame"
    quad = np.asarray(field["quad"], np.float64)
    full_area = polygon_area(quad)
    in_frame = polygon_area(clip_polygon_to_rect(quad, photo_width, photo_height)) / full_area if full_area else 0.0
    if in_frame < FIELD_IN_FRAME_MIN_FRACTION:
        return "partially_out_of_frame"
    if field["visible_fraction"] < FIELD_UNOCCLUDED_MIN_VISIBLE_FRACTION:
        return "occluded"
    if field_text_height_in_photo_px(field["quad"]) < MIN_LEGIBLE_TEXT_HEIGHT_PHOTO_PX:
        return "too_small"
    return "ok"


def rectify_check(photo_rgb: np.ndarray, corners: list[list[float]], size_kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Warp one check to its 1600 px upright crop; returns (crop, photo->crop homography)."""
    corner_array = np.asarray(corners, np.float64)
    width_inches, height_inches = CHECK_SIZE_INCHES[size_kind]
    size = (RECTIFIED_CHECK_WIDTH_PX, int(round(RECTIFIED_CHECK_WIDTH_PX * height_inches / width_inches)))
    target = np.array([[0, 0], [size[0], 0], [size[0], size[1]], [0, size[1]]], np.float32) - PIXEL_CENTRE_TO_EDGE_OFFSET
    homography = cv2.getPerspectiveTransform(corner_array.astype(np.float32), target).astype(np.float64)
    crop = cv2.warpPerspective(photo_rgb, homography, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return crop, homography


def field_box_in_crop(quad: list[list[float]], homography: np.ndarray, crop_size: tuple[int, int]) -> list[float] | None:
    """Axis-aligned pixel-edge box of a photo quad inside the rectified crop, clipped; None if empty."""
    width, height = crop_size
    points = np.asarray(quad, np.float64)
    homogeneous = np.hstack([points, np.ones((len(points), 1))]) @ homography.T
    mapped = homogeneous[:, :2] / homogeneous[:, 2:3] + PIXEL_CENTRE_TO_EDGE_OFFSET
    x0, y0 = np.clip(mapped.min(axis=0), 0, [width, height])
    x1, y1 = np.clip(mapped.max(axis=0), 0, [width, height])
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    return [round(float(v), 1) for v in (x0, y0, x1, y1)]


def field_crop_box(box: list[float], crop_size: tuple[int, int]) -> list[int]:
    """Integer crop window: the box plus a 15%-of-height margin, clipped to the check crop."""
    width, height = crop_size
    x0, y0, x1, y1 = box
    margin = max(FIELD_CROP_MIN_MARGIN_PX, FIELD_CROP_MARGIN_FRACTION * (y1 - y0))
    return [int(max(0, np.floor(x0 - margin))), int(max(0, np.floor(y0 - margin))),
            int(min(width, np.ceil(x1 + margin))), int(min(height, np.ceil(y1 + margin)))]


def export_scene(task: tuple[str, str]) -> dict:
    """Worker: cut one train scene; resumes from its per-scene rows file."""
    output_root, scene_id = Path(task[0]), task[1]
    split_ocr = output_root / "ocr" / "train"
    rows_path = split_ocr / "rows" / f"{scene_id}.jsonl"
    if rows_path.exists():
        return {"scene_id": scene_id, "status": "skipped", "rows": [json.loads(l) for l in rows_path.read_text().splitlines() if l]}
    try:
        scene_label = json.loads((SYNTH_V1_ROOT / "train" / "annotations" / f"{scene_id}.json").read_text())
        photo_bgr = cv2.imread(str(SYNTH_V1_ROOT / "train" / "images" / scene_label["image_file"]))
        if photo_bgr is None:
            raise FileNotFoundError(f"unreadable scene image for {scene_id}")
        photo_rgb = cv2.cvtColor(photo_bgr, cv2.COLOR_BGR2RGB)
        width, height = scene_label["image_width"], scene_label["image_height"]
        rows = []
        for check in scene_label["checks"]:
            stem = f"{scene_label['scene_id']}__check={check['check_index']}"
            crop, homography = rectify_check(photo_rgb, check["corners"], check["size_kind"])
            crop_size = (crop.shape[1], crop.shape[0])
            check_crop_path = split_ocr / "checks" / f"{stem}.jpg"
            cv2.imwrite(str(check_crop_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, CHECK_CROP_JPEG_QUALITY])
            canonical = check["canonical"]
            for field in check["fields"]:
                status = field_status(field, width, height)
                box = field_box_in_crop(field["quad"], homography, crop_size) if status != "not_in_frame" else None
                field_crop_path, crop_window = None, None
                if box is None and status != "micr_blurred_in_app":
                    status = "not_in_frame"
                if box is not None and field["field_name"] in EXPORTED_FIELD_NAMES:
                    crop_window = field_crop_box(box, crop_size)
                    x0, y0, x1, y1 = crop_window
                    field_crop_path = split_ocr / "fields" / f"{stem}__field={field['field_name']}.png"
                    cv2.imwrite(str(field_crop_path), cv2.cvtColor(crop[y0:y1, x0:x1], cv2.COLOR_RGB2BGR))
                is_handwritten = field["handwritten"]
                font_key = "signature_font_id" if field["field_name"] == "signature" else "handwriting_font_id"
                rows.append({
                    "scene_id": scene_label["scene_id"], "split": "train", "check_index": check["check_index"],
                    "field_name": field["field_name"], "text": field["text"], "handwritten": is_handwritten,
                    "status": status, "usable": status == "ok", "visible_fraction": field["visible_fraction"],
                    "check_visible_fraction": check["visible_fraction"],
                    "check_width_in_photo_px": round(float((np.linalg.norm(np.subtract(check["corners"][1], check["corners"][0]))
                                                            + np.linalg.norm(np.subtract(check["corners"][2], check["corners"][3]))) / 2), 1),
                    "text_height_in_photo_px": field_text_height_in_photo_px(field["quad"]),
                    "check_crop": str(check_crop_path.relative_to(output_root)),
                    "field_crop": str(field_crop_path.relative_to(output_root)) if field_crop_path else None,
                    "box_in_check_crop": box, "field_crop_window": crop_window, "check_crop_size": list(crop_size),
                    "template_id": check["template_id"], "layout_family": canonical.get("layout_family"),
                    "size_kind": check["size_kind"], "orientation_class": check["orientation_class"],
                    "pen_font_id": canonical.get(font_key) if is_handwritten else None,
                    "handwriting_font_id": canonical.get("handwriting_font_id"),
                    "signature_font_id": canonical.get("signature_font_id"),
                    "background_id": scene_label["background_id"], "deformed": bool(check.get("deformation")),
                })
    except Exception as error:  # noqa: BLE001 - per-scene boundary, reported by the coordinator
        return {"scene_id": scene_id, "status": "failed", "rows": [], "error": f"{type(error).__name__}: {error}"}
    partial_path = rows_path.with_suffix(".jsonl.partial")
    partial_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    partial_path.replace(rows_path)
    return {"scene_id": scene_id, "status": "built", "rows": rows}


def main() -> None:
    """Export every train scene with a small process pool, then write the split manifest."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="only the first N scenes (smoke test)")
    parser.add_argument("--output-root", type=Path, default=TRAIN_CROPS_ROOT)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for subdirectory in ("checks", "fields", "rows"):
        (arguments.output_root / "ocr" / "train" / subdirectory).mkdir(parents=True, exist_ok=True)
    scene_ids = sorted(p.stem for p in (SYNTH_V1_ROOT / "train" / "annotations").glob("*.json"))[: arguments.limit]
    all_rows, failures = [], []
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        for done_count, result in enumerate(pool.map(export_scene, [(str(arguments.output_root), s) for s in scene_ids], chunksize=4), 1):
            if result["status"] == "failed":
                failures.append(result)
                logger.warning("scene %s failed: %s", result["scene_id"], result["error"])
            all_rows.extend(result["rows"])
            if done_count % 250 == 0:
                logger.info("%d / %d scenes", done_count, len(scene_ids))
    all_rows.sort(key=lambda row: (row["scene_id"], row["check_index"], row["field_name"]))
    manifest_path = arguments.output_root / "ocr" / "ocr_fields__split=train.jsonl"
    manifest_path.write_text("".join(json.dumps(row) + "\n" for row in all_rows))
    logger.info("wrote %d rows from %d scenes (%d failed) to %s", len(all_rows), len(scene_ids), len(failures), manifest_path)


if __name__ == "__main__":
    main()
