"""Render zoomed corner crops showing approximate (red), refined (green) and GT (blue) quads.

Each debug image is a 2x2 panel, one tile per corner in GT order (TL, TR, BR, BL), each a
square crop around the GT corner upscaled 3x so a 1 px error is visible by eye. Quads are
drawn with sub-pixel precision (cv2 shift bits) on the upscaled crop.
"""

from pathlib import Path

import cv2
import numpy as np

CORNER_TILE_HALF_SIZE_PIXELS = 40
ZOOM_FACTOR = 3
DRAWING_SHIFT_BITS = 4
APPROXIMATE_COLOUR_BGR = (0, 0, 255)
REFINED_COLOUR_BGR = (0, 220, 0)
GROUND_TRUTH_COLOUR_BGR = (255, 120, 0)


def draw_quad_on_tile(tile: np.ndarray, corners: np.ndarray, tile_origin: np.ndarray, colour: tuple[int, int, int]) -> None:
    """Draw a closed quad (full-image coords) onto an upscaled tile, with corner dots."""
    scale = ZOOM_FACTOR * (1 << DRAWING_SHIFT_BITS)
    # Pixel centres: full-image x maps to tile (x - origin + 0.5) * zoom.
    fixed_point = np.round((corners - tile_origin + 0.5) * scale).astype(np.int32)
    cv2.polylines(tile, [fixed_point.reshape(-1, 1, 2)], True, colour, 1, cv2.LINE_AA, DRAWING_SHIFT_BITS)
    for point in fixed_point:
        cv2.circle(tile, tuple(int(value) for value in point), 3 << DRAWING_SHIFT_BITS, colour, 1, cv2.LINE_AA, DRAWING_SHIFT_BITS)


def render_corner_debug_panel(
    image_bgr: np.ndarray, ground_truth_corners: np.ndarray, approximate_corners: np.ndarray, refined_corners: np.ndarray
) -> np.ndarray:
    """2x2 panel of 3x-zoomed crops around each GT corner with the three quads overlaid."""
    tiles = []
    tile_size = 2 * CORNER_TILE_HALF_SIZE_PIXELS
    for corner in ground_truth_corners:
        tile_origin = np.round(corner) - CORNER_TILE_HALF_SIZE_PIXELS
        # getRectSubPix centres a w-wide patch at c - (w-1)/2; this makes the origin integral.
        patch_centre = tile_origin + (tile_size - 1) / 2
        crop = cv2.getRectSubPix(image_bgr, (tile_size, tile_size), (float(patch_centre[0]), float(patch_centre[1])))
        tile = cv2.resize(crop, None, fx=ZOOM_FACTOR, fy=ZOOM_FACTOR, interpolation=cv2.INTER_NEAREST)
        draw_quad_on_tile(tile, approximate_corners, tile_origin, APPROXIMATE_COLOUR_BGR)
        draw_quad_on_tile(tile, ground_truth_corners, tile_origin, GROUND_TRUTH_COLOUR_BGR)
        draw_quad_on_tile(tile, refined_corners, tile_origin, REFINED_COLOUR_BGR)
        tiles.append(tile)
    return np.vstack([np.hstack(tiles[:2]), np.hstack([tiles[3], tiles[2]])])


def render_debug_panels_for_records(records: list[dict], image_paths_by_scene_id: dict[str, Path], output_directory: Path) -> list[Path]:
    """Write one JPEG panel per record; filename carries scene, check and errors."""
    output_directory.mkdir(parents=True, exist_ok=True)
    written_paths = []
    for record in records:
        image_bgr = cv2.imread(str(image_paths_by_scene_id[record["scene_id"]]), cv2.IMREAD_COLOR)
        panel = render_corner_debug_panel(
            image_bgr, np.array(record["ground_truth_corners"]), np.array(record["approximate_corners"]), np.array(record["refined_corners"])
        )
        name = (
            f"{record['scene_id']}__check={record['check_index']}"
            f"__before={max(record['errors_before']):.1f}__after={max(record['errors_after']):.1f}.jpg"
        )
        output_path = output_directory / name
        cv2.imwrite(str(output_path), panel, [cv2.IMWRITE_JPEG_QUALITY, 90])
        written_paths.append(output_path)
    return written_paths
