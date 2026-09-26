"""Exports load and agree with the scene labels: COCO, YOLO seg, YOLO OBB, data yamls, OCR manifest.

`pycocotools` / `ultralytics` are not installed here, so formats are validated by hand against
their documented schemas (COCO keypoints; Ultralytics `class x y ...` normalized lines).
"""

import json

import numpy as np
import pytest
from PIL import Image

from dataset_builder.exports.annotation_exports import passes_ultralytics_label_check
from dataset_builder.exports.coco_export import CHECK_CATEGORY_ID, FIELD_CATEGORY_IDS
from dataset_builder.ocr.ocr_rectify import rectify_check, field_box_in_crop
from synthetic_checks.splits import SPLIT_NAMES


def read_yolo_lines(path):
    return [[float(value) for value in line.split()] for line in path.read_text().splitlines() if line.strip()]


def test_coco_documents_are_consistent(built_dataset):
    _, output_directory, manifest, _ = built_dataset
    for split_name in SPLIT_NAMES:
        coco = json.loads((output_directory / split_name / "annotations_coco.json").read_text())
        assert len(coco["images"]) == manifest.scene_counts[split_name]
        images = {image["id"]: image for image in coco["images"]}
        category_ids = {category["id"] for category in coco["categories"]}
        assert category_ids == {CHECK_CATEGORY_ID, *FIELD_CATEGORY_IDS.values()}
        annotation_ids = [annotation["id"] for annotation in coco["annotations"]]
        assert len(annotation_ids) == len(set(annotation_ids))
        checks = {a["id"]: a for a in coco["annotations"] if a["category_id"] == CHECK_CATEGORY_ID}
        assert len(checks) >= manifest.scene_counts[split_name]
        for annotation in coco["annotations"]:
            image = images[annotation["image_id"]]
            assert (output_directory / split_name / image["file_name"]).exists()
            polygon = np.array(annotation["segmentation"][0]).reshape(-1, 2)
            assert len(polygon) >= 3 and annotation["area"] > 0
            assert polygon.min() >= 0 and (polygon <= [image["width"], image["height"]]).all()
            x, y, w, h = annotation["bbox"]
            assert x >= 0 and y >= 0 and x + w <= image["width"] + 0.01 and y + h <= image["height"] + 0.01
            if annotation["category_id"] == CHECK_CATEGORY_ID:
                assert len(annotation["keypoints"]) == 12
                assert annotation["attributes"]["orientation_class"] in (0, 90, 180, 270)
            else:
                assert annotation["attributes"]["check_annotation_id"] in checks
                assert isinstance(annotation["attributes"]["text"], str)


def test_yolo_seg_and_obb_lines_match_annotations(built_dataset):
    _, output_directory, _, _ = built_dataset
    for split_name in SPLIT_NAMES:
        for label_path in (output_directory / split_name / "annotations").glob("*.json"):
            scene = json.loads(label_path.read_text())
            segmentation = read_yolo_lines(output_directory / split_name / "labels" / f"{label_path.stem}.txt")
            oriented = read_yolo_lines(output_directory / split_name / "labels_obb" / f"{label_path.stem}.txt")
            assert len(segmentation) == len(oriented) <= len(scene["checks"])
            assert len(segmentation) >= 1
            for line in segmentation:
                assert line[0] == 0 and len(line) % 2 == 1 and len(line) >= 7
                assert min(line[1:]) >= 0 and max(line[1:]) <= 1
            for line in oriented:
                assert line[0] == 0 and len(line) == 9
                assert passes_ultralytics_label_check(np.array(line[1:]).reshape(4, 2))


def test_data_yamls_and_obb_symlinks_resolve(built_dataset):
    _, output_directory, manifest, _ = built_dataset
    assert "test: eval/images" in (output_directory / "data.yaml").read_text()
    assert "yolo_obb" in (output_directory / "data_obb.yaml").read_text()
    assert "path:" not in (output_directory / "data.yaml").read_text().replace("`path:`", "")
    assert not (output_directory / "yolo_obb" / "train" / "images").is_symlink()
    for split_name in SPLIT_NAMES:
        obb_images = sorted((output_directory / "yolo_obb" / split_name / "images").glob("*.jpg"))
        assert len(obb_images) == manifest.scene_counts[split_name]
        assert (output_directory / "yolo_obb" / split_name / "labels" / f"{obb_images[0].stem}.txt").exists()


def test_ocr_manifest_rows_point_at_real_crops(built_dataset):
    _, output_directory, manifest, _ = built_dataset
    for split_name in ("val", "eval"):
        rows = [json.loads(line) for line in (output_directory / "ocr" / f"ocr_fields__split={split_name}.jsonl").read_text().splitlines()]
        assert len(rows) == manifest.ocr_row_counts[split_name]["rows"] > 0
        assert any(row["usable"] and row["handwritten"] for row in rows)
        assert any(row["status"] == "micr_blurred_in_app" and not row["usable"] for row in rows)
        assert all(row["usable"] == (row["status"] == "ok") for row in rows)
        assert all(row["text_height_in_photo_px"] >= 14 for row in rows if row["usable"])
        for row in rows:
            assert row["template_id"] in manifest.split_pools[split_name]["template_ids"]
            check_crop = Image.open(output_directory / row["check_crop"])
            assert check_crop.width == 1600 and tuple(row["check_crop_size"]) == check_crop.size
            if row["status"] == "not_in_frame":
                assert row["field_crop"] is None
                continue
            x0, y0, x1, y1 = row["box_in_check_crop"]
            wx0, wy0, wx1, wy1 = row["field_crop_window"]
            assert 0 <= wx0 <= x0 < x1 <= wx1 <= check_crop.width and 0 <= wy0 <= y0 < y1 <= wy1 <= check_crop.height
            assert Image.open(output_directory / row["field_crop"]).size == (wx1 - wx0, wy1 - wy0)
    assert not (output_directory / "ocr" / "train").exists()


def test_rectification_maps_corners_to_the_crop_corners():
    photo = np.zeros((600, 800, 3), np.uint8)
    corners = [[500.0, 100.0], [600.0, 400.0], [330.0, 480.0], [240.0, 180.0]]   # rotated ~70 degrees
    crop, homography = rectify_check(photo, corners, "personal")
    assert crop.shape == (733, 1600, 3)
    assert field_box_in_crop(corners, homography, (1600, 733)) == pytest.approx([0, 0, 1600, 733], abs=0.2)
