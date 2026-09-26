"""Paste one (possibly deformed) check onto the sheet-plane canvas, and record how light reaches it.

For the canvas region under the check:
1. Inverse-map every plane pixel to check coordinates (check_plane_map.py) and sample the
   check, pre-downscaled to plane resolution, with premultiplied alpha. Alpha is the paper
   coverage from an RGBA render (contract C2), or 1 everywhere for RGB.
2. Composite the paper's albedo (times the background's baked light transferred onto it)
   into the canvas and write the owner id (for occlusion measurements).
3. Update the scene's light buffers (scene_irradiance.py) from paper_shading.py: under the
   paper, its own Lambert factors for key and fill; around it, the drop shadow (key only) and
   contact occlusion (ambient). The light itself is applied once for the whole canvas later,
   so sheet, paper and shadows share one light.
"""

import cv2
import numpy as np

from synth.compose.check_plane_map import CheckPlaneMap
from synth.compose.paper_shading import PaperSurface, contact_occlusion, drop_shadow, paper_geometry_factors
from synth.compose.scene_irradiance import LightBuffers
from synth.compose.scene_light import SceneLight

EDGE_FEATHER_SIGMA_PX = 0.6
OUTLINE_SAMPLES_PER_EDGE = 24
SHADOW_PAD_MARGIN_PX = 12


def check_boundary_points(width_px: int, height_px: int, samples_per_edge: int) -> np.ndarray:
    """Points along the check's edge, starting at TL, clockwise in the check's own frame (TL, TR, BR, BL)."""
    fractions = np.linspace(0, 1, samples_per_edge, endpoint=False)
    top = np.stack([fractions * width_px, np.zeros_like(fractions)], axis=1)
    right = np.stack([np.full_like(fractions, width_px), fractions * height_px], axis=1)
    bottom = np.stack([(1 - fractions) * width_px, np.full_like(fractions, height_px)], axis=1)
    left = np.stack([np.zeros_like(fractions), (1 - fractions) * height_px], axis=1)
    return np.vstack([top, right, bottom, left])


def split_color_and_coverage(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """float RGB and coverage alpha (1 for an RGB image) from an HxWx3 or HxWx4 array in [0, 1]."""
    if image.shape[2] == 4:
        return image[..., :3], image[..., 3]
    return image, np.ones(image.shape[:2], np.float32)


def paste_deformed_check(canvas: np.ndarray, id_map: np.ndarray, buffers: LightBuffers, check_image: np.ndarray,
                         plane_map: CheckPlaneMap, light: SceneLight, surface: PaperSurface, color_multiplier: np.ndarray,
                         owner_id: int) -> tuple[int, int, np.ndarray] | None:
    """Warp one check (float HxWx3 or HxWx4, [0, 1], full render resolution) into the canvas and light buffers.

    `color_multiplier` (3,) carries the background's baked light and colour cast onto this check.

    Returns (x0, y0, alpha) of the canvas region written, or None if the check misses the canvas.
    """
    boundary_plane = plane_map.check_to_plane(check_boundary_points(plane_map.check_width_px, plane_map.check_height_px,
                                                                    OUTLINE_SAMPLES_PER_EDGE))
    max_lift = float(plane_map.deformation.height(np.linspace(0, plane_map.deformation.width_inches, 64)[:, None],
                                                  np.linspace(0, plane_map.deformation.height_inches, 32)[None]).max())
    ppi = plane_map.plane_pixels_per_inch
    pad = int((max_lift + surface.base_lift_inches) * ppi / np.tan(light.key_elevation_radians) * 1.5
              + 3 * surface.crease_ridge_half_width_inches * ppi + SHADOW_PAD_MARGIN_PX)
    canvas_height, canvas_width = canvas.shape[:2]
    x0 = max(0, int(np.floor(boundary_plane[:, 0].min())) - pad); x1 = min(canvas_width, int(np.ceil(boundary_plane[:, 0].max())) + pad)
    y0 = max(0, int(np.floor(boundary_plane[:, 1].min())) - pad); y1 = min(canvas_height, int(np.ceil(boundary_plane[:, 1].max())) + pad)
    if x1 <= x0 or y1 <= y0:
        return None

    color, coverage = split_color_and_coverage(check_image)
    scale = min(1.0, ppi / plane_map.dpi)  # sample from a copy at about plane resolution (antialiasing)
    small_size = (max(8, int(round(plane_map.check_width_px * scale))), max(4, int(round(plane_map.check_height_px * scale))))
    small_premultiplied = cv2.resize(color * coverage[..., None], small_size, interpolation=cv2.INTER_AREA)
    small_coverage = cv2.resize(coverage, small_size, interpolation=cv2.INTER_AREA)
    map_x, map_y = plane_map.plane_region_to_check_maps(x0, y0, x1 - x0, y1 - y0)
    scale_x, scale_y = small_size[0] / plane_map.check_width_px, small_size[1] / plane_map.check_height_px
    sample_x, sample_y = map_x * scale_x - 0.5, map_y * scale_y - 0.5  # continuous check coords -> pixel-center indices
    premultiplied = cv2.remap(small_premultiplied, sample_x, sample_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    sampled_alpha = cv2.remap(small_coverage, sample_x, sample_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    color = premultiplied / np.maximum(sampled_alpha, 1e-4)[..., None] * color_multiplier
    alpha = cv2.GaussianBlur(sampled_alpha, (0, 0), sigmaX=EDGE_FEATHER_SIGMA_PX) * sampled_alpha  # feather inward only

    height_px = (plane_map.deformation.height(map_x / plane_map.dpi, map_y / plane_map.dpi) * ppi).astype(np.float32) * (alpha > 0)
    key_geometry, fill_geometry = paper_geometry_factors(plane_map, map_x, map_y, light, surface)
    shadow = drop_shadow(alpha, height_px, light, surface, ppi)
    occlusion = contact_occlusion(alpha, height_px, surface, ppi)

    region = canvas[y0:y1, x0:x1]
    region[:] = region * (1 - alpha[..., None]) + np.clip(color, 0, 1) * alpha[..., None]
    rows, cols = slice(y0, y1), slice(x0, x1)
    outside = 1 - alpha
    buffers.key_factor[rows, cols] = (buffers.key_factor[rows, cols] * (1 - surface.drop_shadow_opacity * shadow) * outside
                                      + key_geometry * alpha)
    buffers.fill_factor[rows, cols] = buffers.fill_factor[rows, cols] * outside + fill_geometry * alpha
    buffers.ambient_factor[rows, cols] = buffers.ambient_factor[rows, cols] * (1 - occlusion) * outside + alpha
    owned = alpha > 0.5
    id_map[y0:y1, x0:x1][owned] = owner_id
    return x0, y0, alpha
