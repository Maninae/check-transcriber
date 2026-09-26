"""Presentation images for a built dataset: the five pictures that show the work at a glance.

Usage:
    python -m synth.dataset.showcase_images DATASET_DIR --out DIR [--print-page PNG] [--real-photo JPG]

Writes, each full-frame and under 5 MB (they are sent to a phone):
1. `showcase__1__contact_sheet.jpg`: six landscape scenes on different surfaces, paper outline in green,
   the check's own top-left corner as a dot (so orientation is visible).
2. `showcase__2__full_scene.jpg`: one scene at full resolution, untouched.
3. `showcase__3__crop_3x_vs_real.jpg`: a 3x crop of handwriting inside a scene above a 3x crop of a real
   phone photo of a cheque at a similar physical scale.
4. `showcase__4__flat_check.jpg`: one flat rendered check, handwritten, at print resolution.
5. `showcase__5__print_sheet.jpg`: a print-sheet page (pass `--print-page`).

- Scene choice is deterministic: largest readable groups first, one scene per background.
- Captions are small and gray; the pictures are the payload.
"""

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from synth.compose.perspective import polygon_area
from synth.render.check_layout import LayoutFamily
from synth.render.check_templates import build_template_catalog
from synth.render.fake_data import sample_check_content
from synth.render.fonts import load_font
from synth.render.render_check import render_check

logger = logging.getLogger(__name__)

CONTACT_TILE_WIDTH = 1200
CONTACT_COLUMNS = 3
CONTACT_SCENE_COUNT = 6
CONTACT_OUTLINE_BGR = (60, 200, 60)
CONTACT_OUTLINE_WIDTH = 4
TOP_LEFT_DOT_RADIUS = 11
FULL_SCENE_CHECK_RANGE = (4, 8)
CROP_HALF_SIZE_PX = (300, 130)          # half width, half height of the in-scene crop before upscaling
CROP_UPSCALE = 3
REAL_PHOTO_CROP_BOX = (800, 300, 2000, 820)  # handwriting line of the dz sample, similar ink scale
CAPTION_HEIGHT_PX = 56
CAPTION_FONT_ID = "source_sans_3"
CAPTION_EM_PX = 34
CAPTION_GRAY = (110, 110, 110)
JPEG_QUALITY = 88
MAX_OUTPUT_BYTES = 5_000_000
PRINT_PAGE_HEIGHT_PX = 2400
FLAT_CHECK_SEED = 21
BEDDING_ID_WORDS = ("bedsheet", "duvet", "comforter", "blanket", "quilt", "sheet")
DEFAULT_REAL_PHOTO = Path("/Volumes/vega/datasets/check-transcriber/samples/sarahhdd-cheque-dz/"
                          "cheques__Train__BDL__IMG_20240926_124907.jpg")


def load_scene_labels(dataset_directory: Path) -> list[tuple[Path, dict]]:
    """Every (split directory, scene label) pair in the dataset, in a stable order."""
    labels = []
    for split_name in ("train", "val", "eval"):
        for label_path in sorted((dataset_directory / split_name / "annotations").glob("*.json")):
            labels.append((dataset_directory / split_name, json.loads(label_path.read_text())))
    return labels


def readable_group_score(scene_label: dict) -> float:
    """Prefer landscape scenes with several fully visible checks that are large in the frame."""
    checks = scene_label["checks"]
    if scene_label["image_width"] <= scene_label["image_height"] or len(checks) < 3:
        return -1.0
    visible = [check for check in checks if check["visible_fraction"] > 0.97 and check["fully_in_frame"]]
    return len(visible) / len(checks) + min(len(checks), 8) / 8


def frame_fill_fraction(scene_label: dict) -> float:
    """Share of the photo covered by check outlines (overlaps counted twice; fine for ranking)."""
    area = sum(polygon_area(np.array(check["outline"])) for check in scene_label["checks"])
    return area / (scene_label["image_width"] * scene_label["image_height"])


def choose_full_scene(labels: list[tuple[Path, dict]]) -> tuple[Path, dict]:
    """A landscape, frame-filling scene of 4-8 fully visible checks, on bedding when possible."""
    candidates = [(split_directory, scene_label) for split_directory, scene_label in labels
                  if FULL_SCENE_CHECK_RANGE[0] <= len(scene_label["checks"]) <= FULL_SCENE_CHECK_RANGE[1]
                  and readable_group_score(scene_label) >= 1.0 + FULL_SCENE_CHECK_RANGE[0] / 8]
    bedding = [c for c in candidates if any(word in c[1]["background_id"] for word in BEDDING_ID_WORDS)]
    return max(bedding or candidates, key=lambda item: frame_fill_fraction(item[1]))


def choose_scenes_on_distinct_backgrounds(labels: list[tuple[Path, dict]], count: int) -> list[tuple[Path, dict]]:
    """The best-scoring scenes, at most one per background."""
    chosen, used_backgrounds = [], set()
    for split_directory, scene_label in sorted(labels, key=lambda item: -readable_group_score(item[1])):
        if readable_group_score(scene_label) < 0 or scene_label["background_id"] in used_backgrounds:
            continue
        chosen.append((split_directory, scene_label))
        used_backgrounds.add(scene_label["background_id"])
        if len(chosen) == count:
            break
    return chosen


def read_scene_bgr(split_directory: Path, scene_label: dict) -> np.ndarray:
    """The scene photo as BGR."""
    return cv2.imread(str(split_directory / "images" / scene_label["image_file"]))


def draw_outlines(image: np.ndarray, scene_label: dict, scale: float) -> np.ndarray:
    """Paper outline and the check's own top-left corner, nothing else."""
    for check in scene_label["checks"]:
        outline = np.round(np.array(check["outline"]) * scale).astype(np.int32)
        cv2.polylines(image, [outline], True, CONTACT_OUTLINE_BGR, CONTACT_OUTLINE_WIDTH, cv2.LINE_AA)
        top_left = np.round(np.array(check["corners"][0]) * scale).astype(int)
        cv2.circle(image, (int(top_left[0]), int(top_left[1])), TOP_LEFT_DOT_RADIUS, CONTACT_OUTLINE_BGR, -1, cv2.LINE_AA)
    return image


def write_contact_sheet(scenes: list[tuple[Path, dict]], output_path: Path) -> Path:
    """Tile landscape scenes edge to edge (all are 4:3, so there is no dead space)."""
    tile_height = CONTACT_TILE_WIDTH * 3 // 4
    tiles = []
    for split_directory, scene_label in scenes:
        image = read_scene_bgr(split_directory, scene_label)
        scale = CONTACT_TILE_WIDTH / image.shape[1]
        resized = cv2.resize(image, (CONTACT_TILE_WIDTH, tile_height), interpolation=cv2.INTER_AREA)
        tiles.append(draw_outlines(resized, scene_label, scale))
    rows = [np.hstack(tiles[i:i + CONTACT_COLUMNS]) for i in range(0, len(tiles), CONTACT_COLUMNS)]
    return save_jpeg_bgr(np.vstack(rows), output_path)


def save_jpeg_bgr(image: np.ndarray, output_path: Path) -> Path:
    """JPEG at the showcase quality, stepping quality down until it fits the size cap."""
    quality = JPEG_QUALITY
    while True:
        cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if output_path.stat().st_size <= MAX_OUTPUT_BYTES or quality <= 60:
            return output_path
        quality -= 6


def best_handwritten_payee(labels: list[tuple[Path, dict]]) -> tuple[Path, dict, dict]:
    """(split dir, scene, payee field) with the widest fully visible handwritten payee."""
    best, best_width = None, -1.0
    for split_directory, scene_label in labels:
        for check in scene_label["checks"]:
            if check["visible_fraction"] < 0.98:
                continue
            for field in check["fields"]:
                if field["field_name"] == "payee" and field["handwritten"] and field["bbox_clipped"]:
                    width = field["bbox_clipped"][2] - field["bbox_clipped"][0]
                    if width > best_width:
                        best, best_width = (split_directory, scene_label, field), width
    if best is None:
        raise ValueError("no fully visible handwritten payee in this dataset")
    return best


def caption_strip(width: int, text: str) -> Image.Image:
    """A white strip with a small gray caption."""
    strip = Image.new("RGB", (width, CAPTION_HEIGHT_PX), "white")
    ImageDraw.Draw(strip).text((16, CAPTION_HEIGHT_PX // 2), text, fill=CAPTION_GRAY,
                               font=load_font(CAPTION_FONT_ID, CAPTION_EM_PX), anchor="lm")
    return strip


def write_crop_comparison(labels: list[tuple[Path, dict]], real_photo_path: Path, output_path: Path) -> Path:
    """3x crop of in-scene handwriting above a 3x-scale crop of a real phone photo."""
    split_directory, scene_label, payee = best_handwritten_payee(labels)
    scene = Image.open(split_directory / "images" / scene_label["image_file"]).convert("RGB")
    x0, y0, x1, y1 = payee["bbox_clipped"]
    center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
    half_width, half_height = CROP_HALF_SIZE_PX
    output_size = (2 * half_width * CROP_UPSCALE, 2 * half_height * CROP_UPSCALE)
    synthetic = scene.crop((int(center_x - half_width), int(center_y - half_height),
                            int(center_x + half_width), int(center_y + half_height))).resize(output_size, Image.Resampling.LANCZOS)
    real = Image.open(real_photo_path).convert("RGB").crop(REAL_PHOTO_CROP_BOX).resize(output_size, Image.Resampling.LANCZOS)
    width = output_size[0]
    panels = [caption_strip(width, f"synthetic: {scene_label['scene_id']}, 3x crop inside the scene"), synthetic,
              caption_strip(width, "real phone photo of a cheque (public dz sample), same scale"), real]
    sheet = Image.new("RGB", (width, sum(panel.height for panel in panels)), "white")
    top = 0
    for panel in panels:
        sheet.paste(panel, (0, top))
        top += panel.height
    return save_jpeg_bgr(cv2.cvtColor(np.asarray(sheet), cv2.COLOR_RGB2BGR), output_path)


def write_flat_check(output_path: Path) -> Path:
    """One handwritten personal check, flat, textured, at print resolution."""
    template = next(t for t in build_template_catalog() if t.layout_family == LayoutFamily.PERSONAL_DATE_BOX)
    rng = np.random.default_rng(FLAT_CHECK_SEED)
    for _ in range(40):  # draw content until the whole check is filled by hand
        content = sample_check_content(template, rng)
        if "payee" in content.handwritten_fields and "amount_words" in content.handwritten_fields and content.memo_text:
            break
    image, _ = render_check(template, content, rng)
    flat = Image.new("RGB", image.size, "white")
    flat.paste(image, mask=image.split()[-1] if image.mode == "RGBA" else None)
    return save_jpeg_bgr(cv2.cvtColor(np.asarray(flat), cv2.COLOR_RGB2BGR), output_path)


def write_print_page(print_page_path: Path, output_path: Path) -> Path:
    """A print-sheet page scaled for a phone screen."""
    page = cv2.imread(str(print_page_path))
    scale = PRINT_PAGE_HEIGHT_PX / page.shape[0]
    page = cv2.resize(page, (int(page.shape[1] * scale), PRINT_PAGE_HEIGHT_PX), interpolation=cv2.INTER_AREA)
    return save_jpeg_bgr(page, output_path)


def write_showcase(dataset_directory: Path, output_directory: Path, print_page_path: Path | None,
                   real_photo_path: Path) -> list[Path]:
    """Write every showcase image that the inputs allow; returns their paths."""
    output_directory.mkdir(parents=True, exist_ok=True)
    labels = load_scene_labels(dataset_directory)
    scenes = choose_scenes_on_distinct_backgrounds(labels, CONTACT_SCENE_COUNT)
    written = [write_contact_sheet(scenes, output_directory / "showcase__1__contact_sheet.jpg")]
    split_directory, scene_label = choose_full_scene(labels)
    written.append(save_jpeg_bgr(read_scene_bgr(split_directory, scene_label), output_directory / "showcase__2__full_scene.jpg"))
    written.append(write_crop_comparison(labels, real_photo_path, output_directory / "showcase__3__crop_3x_vs_real.jpg"))
    written.append(write_flat_check(output_directory / "showcase__4__flat_check.jpg"))
    if print_page_path:
        written.append(write_print_page(print_page_path, output_directory / "showcase__5__print_sheet.jpg"))
    for path in written:
        logger.info("%s (%.1f MB)", path, path.stat().st_size / 1e6)
    return written


def main() -> None:
    """Entry point for `python -m synth.dataset.showcase_images`."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_directory", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--print-page", type=Path, default=None, help="a print_sheet__page=NN.png to include")
    parser.add_argument("--real-photo", type=Path, default=DEFAULT_REAL_PHOTO)
    arguments = parser.parse_args()
    write_showcase(arguments.dataset_directory, arguments.out, arguments.print_page, arguments.real_photo)


if __name__ == "__main__":
    main()
