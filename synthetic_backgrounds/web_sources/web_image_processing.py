"""Decode, screen and normalize one downloaded web background.

Automatic rejections (cheap, before a human screens contact sheets):
- short side below `MIN_SHORT_SIDE_PX` (after scale normalization);
- aspect ratio beyond `MAX_ASPECT_RATIO` (the compositor needs a landscape or square crop);
- nearly one flat colour (grey-level std below `MIN_GREY_STD`);
- heavily saturated synthetic-looking renders (most pixels vivid).

Scale normalization for scanned textures with a known physical width: small seamless tiles are
repeated 2x2 and large scans are centre-cropped, so every texture spans roughly
`TARGET_PHYSICAL_EXTENT_MM` (about the table area a phone photo of a few checks covers).
"""

from dataclasses import dataclass

import cv2
import numpy as np

MIN_SHORT_SIDE_PX = 900
MAX_LONG_EDGE_PX = 2048
MAX_ASPECT_RATIO = 2.2
MIN_GREY_STD = 3.0
VIVID_SATURATION = 0.75          # HSV saturation (0..1) counted as vivid
VIVID_MIN_VALUE = 0.3            # ...when not near black
MAX_VIVID_PIXEL_FRACTION = 0.5
TARGET_PHYSICAL_EXTENT_MM = 600.0
TILE_WHEN_EXTENT_BELOW_FRACTION = 0.6   # tile 2x2 when the scan covers < 60% of the target
CROP_WHEN_EXTENT_ABOVE_FRACTION = 1.5   # crop when the scan covers > 150% of the target
JPEG_QUALITY = 92


@dataclass(frozen=True)
class ProcessedBackground:
    """A background ready to write, plus how it was normalized."""

    image_rgb: np.ndarray
    tile_count: int              # 1 = untouched, 2 = tiled 2x2
    crop_fraction: float         # side fraction kept by the centre crop (1.0 = none)


def decode_image_rgb(encoded_bytes: bytes) -> np.ndarray:
    """uint8 RGB from JPEG/PNG/WebP bytes; alpha is dropped. Raises ValueError when undecodable."""
    image = cv2.imdecode(np.frombuffer(encoded_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("undecodable image")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def normalize_physical_scale(image_rgb: np.ndarray, physical_extent_mm: float | None,
                             is_seamless_texture: bool) -> ProcessedBackground:
    """Tile or centre-crop a scanned texture towards the target extent; photos pass through."""
    if not physical_extent_mm or physical_extent_mm <= 0:
        return ProcessedBackground(image_rgb, 1, 1.0)
    extent_ratio = physical_extent_mm / TARGET_PHYSICAL_EXTENT_MM
    if extent_ratio < TILE_WHEN_EXTENT_BELOW_FRACTION and is_seamless_texture:
        return ProcessedBackground(np.tile(image_rgb, (2, 2, 1)), 2, 1.0)
    if extent_ratio > CROP_WHEN_EXTENT_ABOVE_FRACTION:
        height, width = image_rgb.shape[:2]
        # Never crop below the minimum resolution; a slightly coarser scale beats an upscaled blur.
        crop_fraction = min(1.0, max(1.0 / extent_ratio, MIN_SHORT_SIDE_PX * 1.2 / min(height, width)))
        crop_height, crop_width = int(round(height * crop_fraction)), int(round(width * crop_fraction))
        top, left = (height - crop_height) // 2, (width - crop_width) // 2
        return ProcessedBackground(image_rgb[top:top + crop_height, left:left + crop_width], 1, crop_fraction)
    return ProcessedBackground(image_rgb, 1, 1.0)


def rejection_reason(image_rgb: np.ndarray) -> str | None:
    """Why this image cannot be a background, or None if it passes the automatic filters."""
    height, width = image_rgb.shape[:2]
    if min(height, width) < MIN_SHORT_SIDE_PX:
        return f"too small ({width}x{height})"
    if max(height, width) / min(height, width) > MAX_ASPECT_RATIO:
        return f"extreme aspect ({width}x{height})"
    preview = cv2.resize(image_rgb, (256, int(round(256 * height / width))), interpolation=cv2.INTER_AREA)
    grey_std = float(cv2.cvtColor(preview, cv2.COLOR_RGB2GRAY).std())
    if grey_std < MIN_GREY_STD:
        return f"flat colour (grey std {grey_std:.1f})"
    hsv = cv2.cvtColor(preview, cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
    vivid_fraction = float(((hsv[..., 1] > VIVID_SATURATION) & (hsv[..., 2] > VIVID_MIN_VALUE)).mean())
    if vivid_fraction > MAX_VIVID_PIXEL_FRACTION:
        return f"oversaturated ({vivid_fraction:.0%} vivid pixels)"
    return None


def limit_long_edge(image_rgb: np.ndarray, max_long_edge: int = MAX_LONG_EDGE_PX) -> np.ndarray:
    """Downscale (area filter) so the long edge is at most `max_long_edge`; never upscales."""
    height, width = image_rgb.shape[:2]
    scale = max_long_edge / max(height, width)
    if scale >= 1:
        return image_rgb
    return cv2.resize(image_rgb, (int(round(width * scale)), int(round(height * scale))), interpolation=cv2.INTER_AREA)


def encode_jpeg(image_rgb: np.ndarray) -> bytes:
    """JPEG bytes at `JPEG_QUALITY`."""
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return encoded.tobytes()
