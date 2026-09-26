"""Per-field OCR eval manifest: rectified check crops, field crops and one JSONL row per field.

This is what the experiments area scores OCR configurations on: for each labelled field, the
crop an OCR engine would see after the app's rectification (stage 3), with its ground truth.

Layout under the dataset root:
    ocr/<split>/checks/<scene>__check=<i>.jpg                 rectified upright check, 1600 px wide
    ocr/<split>/fields/<scene>__check=<i>__field=<name>.png   field crop (box + small margin)
    ocr/<split>/rows/<scene>.jsonl                            per-scene rows (resume cache)
    ocr/ocr_fields__split=<split>.jsonl                       every row of a split

Row `status` (only `ok` rows have `usable: true`):
- ok: fully in frame and not covered by another check.
- partially_out_of_frame / occluded: crop written, but some of the text may be missing.
- not_in_frame: no crop.
- too_small: in frame and uncovered, but its ink is under `MIN_LEGIBLE_TEXT_HEIGHT_PHOTO_PX` tall
  in the photo, so the 1600 px crop is an upscale of too few pixels to read.
- micr_blurred_in_app: the MICR band; the app blurs it and never reads it, kept for completeness.

`text_height_in_photo_px` is the field quad's mean left/right edge length in the photo: the
ink's full height (cap or ascender to baseline or descender), not its x-height.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from dataset_builder.ocr.ocr_rectify import field_box_in_crop, field_crop_box, rectify_check
from scene_composer.geometry.perspective import clip_polygon_to_rect, polygon_area

OCR_DIRECTORY_NAME = "ocr"
FIELD_IN_FRAME_MIN_FRACTION = 0.999
FIELD_UNOCCLUDED_MIN_VISIBLE_FRACTION = 0.99   # untouched fields measure 0.996-1.0; 0.975 already hid a leading '***'
MICR_FIELD_NAME = "micr"
SIGNATURE_FIELD_NAME = "signature"
CHECK_CROP_JPEG_QUALITY = 95
# Ink-box height floor. Tesseract's FAQ (tessdoc tess3/FAQ-Old): below a 10 px x-height there is
# "very little chance of accurate results". An ink box spans ~1.4x x-height on digit fields
# (amount, date) and ~2x on mixed text, so 14 px is ~10 px x-height on digits, ~7 px on words.
MIN_LEGIBLE_TEXT_HEIGHT_PHOTO_PX = 14.0


def field_text_height_in_photo_px(quad: list[list[float]]) -> float:
    """Mean of the field quad's left (TL->BL) and right (TR->BR) edges in the photo, in pixels."""
    quad_array = np.asarray(quad, np.float64)
    left = np.linalg.norm(quad_array[3] - quad_array[0])
    right = np.linalg.norm(quad_array[2] - quad_array[1])
    return round(float((left + right) / 2), 1)


def field_status(field: dict, photo_width: int, photo_height: int) -> str:
    """Classify one scene field label (see module docstring); occlusion and framing outrank size."""
    if field["field_name"] == MICR_FIELD_NAME:
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


def check_width_in_photo_px(corners: list[list[float]]) -> float:
    """Mean length of the check's top and bottom edges in the photo: the real resolution behind the 1600 px crop."""
    corner_array = np.asarray(corners, np.float64)
    top = np.linalg.norm(corner_array[1] - corner_array[0])
    bottom = np.linalg.norm(corner_array[2] - corner_array[3])
    return round(float((top + bottom) / 2), 1)


def pen_font_id(field: dict, canonical: dict) -> str | None:
    """The handwriting (or signature) font that drew this field, None for printed fields."""
    if not field["handwritten"]:
        return None
    key = "signature_font_id" if field["field_name"] == SIGNATURE_FIELD_NAME else "handwriting_font_id"
    return canonical.get(key)


def ocr_rows_for_scene(dataset_root: Path, split_name: str, scene_label: dict, photo_rgb: np.ndarray) -> list[dict]:
    """Write every crop of one scene and return its rows (paths relative to `dataset_root`)."""
    split_ocr = dataset_root / OCR_DIRECTORY_NAME / split_name
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
            if box is not None:
                crop_window = field_crop_box(box, crop_size)
                x0, y0, x1, y1 = crop_window
                field_crop_path = split_ocr / "fields" / f"{stem}__field={field['field_name']}.png"
                cv2.imwrite(str(field_crop_path), cv2.cvtColor(crop[y0:y1, x0:x1], cv2.COLOR_RGB2BGR))
            elif status != "micr_blurred_in_app":
                status = "not_in_frame"
            rows.append({
                "scene_id": scene_label["scene_id"], "split": split_name, "check_index": check["check_index"],
                "field_name": field["field_name"], "text": field["text"], "handwritten": field["handwritten"],
                "status": status, "usable": status == "ok", "visible_fraction": field["visible_fraction"],
                "check_visible_fraction": check["visible_fraction"],
                "check_width_in_photo_px": check_width_in_photo_px(check["corners"]),
                "text_height_in_photo_px": field_text_height_in_photo_px(field["quad"]),
                "check_crop": str(check_crop_path.relative_to(dataset_root)),
                "field_crop": str(field_crop_path.relative_to(dataset_root)) if field_crop_path else None,
                "box_in_check_crop": box, "field_crop_window": crop_window, "check_crop_size": list(crop_size),
                "template_id": check["template_id"], "layout_family": canonical.get("layout_family"),
                "size_kind": check["size_kind"], "orientation_class": check["orientation_class"],
                "pen_font_id": pen_font_id(field, canonical),
                "handwriting_font_id": canonical.get("handwriting_font_id"),
                "signature_font_id": canonical.get("signature_font_id"),
                "background_id": scene_label["background_id"], "deformed": bool(check.get("deformation")),
            })
    return rows


def write_ocr_rows_for_scene(task: tuple[str, str, str]) -> dict:
    """Worker entry: (dataset_root, split, scene_id) -> {status, rows}; resumes from the per-scene rows file."""
    dataset_root, split_name, scene_id = Path(task[0]), task[1], task[2]
    rows_path = dataset_root / OCR_DIRECTORY_NAME / split_name / "rows" / f"{scene_id}.jsonl"
    if rows_path.exists():
        rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line]
        return {"scene_id": scene_id, "status": "skipped", "rows": rows}
    try:
        scene_label = json.loads((dataset_root / split_name / "annotations" / f"{scene_id}.json").read_text())
        photo_bgr = cv2.imread(str(dataset_root / split_name / "images" / scene_label["image_file"]))
        if photo_bgr is None:
            raise FileNotFoundError(f"scene image unreadable for {scene_id}")
        rows = ocr_rows_for_scene(dataset_root, split_name, scene_label, cv2.cvtColor(photo_bgr, cv2.COLOR_BGR2RGB))
    except Exception as error:  # noqa: BLE001 - per-scene boundary, logged by the coordinator
        return {"scene_id": scene_id, "status": "failed", "rows": [], "error": f"{type(error).__name__}: {error}"}
    partial_path = rows_path.with_suffix(".jsonl.partial")
    partial_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    partial_path.replace(rows_path)
    return {"scene_id": scene_id, "status": "built", "rows": rows}


def prepare_ocr_directories(dataset_root: Path, split_names: list[str]) -> None:
    """Create the crop and row directories for every OCR split."""
    for split_name in split_names:
        for subdirectory in ("checks", "fields", "rows"):
            (dataset_root / OCR_DIRECTORY_NAME / split_name / subdirectory).mkdir(parents=True, exist_ok=True)


def write_split_ocr_manifest(dataset_root: Path, split_name: str, rows: list[dict]) -> Path:
    """Write every row of a split, sorted by scene, check and field, to one JSONL."""
    rows = sorted(rows, key=lambda row: (row["scene_id"], row["check_index"], row["field_name"]))
    manifest_path = dataset_root / OCR_DIRECTORY_NAME / f"ocr_fields__split={split_name}.jsonl"
    manifest_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return manifest_path
