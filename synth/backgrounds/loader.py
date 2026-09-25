"""List and load background images from a directory tree."""

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

BACKGROUND_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class BackgroundSource:
    """One background image; `background_id` is its path relative to the scanned root."""

    background_id: str
    file_path: Path


def list_background_sources(root_directory: Path) -> list[BackgroundSource]:
    """Every JPEG/PNG under `root_directory` (recursive), sorted by id for determinism."""
    if not root_directory.exists():
        raise FileNotFoundError(f"background directory missing: {root_directory}")
    sources = [
        BackgroundSource(file_path.relative_to(root_directory).as_posix(), file_path)
        for file_path in root_directory.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in BACKGROUND_SUFFIXES and not file_path.name.startswith(".")
    ]
    logger.info("found %d backgrounds under %s", len(sources), root_directory)
    return sorted(sources, key=lambda source: source.background_id)


def load_background_rgb(file_path: Path) -> np.ndarray:
    """Read an image as uint8 RGB (kept 8-bit so caches stay small; cover_crop converts)."""
    image_bgr = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"unreadable background image: {file_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def cover_crop(image: np.ndarray, width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    """Scale to cover (width, height), random crop, random flip / half-turn; returns float32 RGB in [0, 1].

    Accepts uint8 (from `load_background_rgb`) or float images in [0, 1].
    """
    if rng.random() < 0.5:
        image = image[:, ::-1]
    if rng.random() < 0.5:
        image = image[::-1, ::-1]
    source_height, source_width = image.shape[:2]
    scale = max(width / source_width, height / source_height) * rng.uniform(1.0, 1.25)
    resized = cv2.resize(np.ascontiguousarray(image), (int(np.ceil(source_width * scale)), int(np.ceil(source_height * scale))),
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    offset_x = int(rng.integers(0, resized.shape[1] - width + 1))
    offset_y = int(rng.integers(0, resized.shape[0] - height + 1))
    crop = resized[offset_y:offset_y + height, offset_x:offset_x + width]
    return crop.astype(np.float32) / 255.0 if crop.dtype == np.uint8 else np.ascontiguousarray(crop, dtype=np.float32)
