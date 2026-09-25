"""Paste one (possibly deformed) check onto the sheet-plane canvas.

For the canvas region under the check:
1. Inverse-map every plane pixel to check coordinates (check_plane_map.py) and sample the
   check, pre-downscaled to plane resolution, with premultiplied alpha. Alpha is the paper
   coverage from an RGBA render (contract C2), or 1 everywhere for RGB.
2. Surface maps: paper height (inches) and unit normal per plane pixel, from the height field.
3. Lambert shading from one light direction: flat paper keeps its brightness, panels turned
   toward the light brighten, panels turned away darken; creases get a thin dark line.
4. Drop shadow: the paper's silhouette, shifted away from the light by (base lift + height)
   / tan(elevation) and blurred more where the paper is higher, darkens what lies beneath.
5. Composite and write the owner id (for occlusion measurements).

The height and normal maps live per check region here; a scene-level lighting pass can rebuild
them from each CheckPlaneMap the same way.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from synth.compose.check_plane_map import CheckPlaneMap

SHADOW_WORK_DOWNSCALE = 4
SHADOW_SIGMA_PER_LIFT = 0.6       # extra blur (plane px) per plane px of height
EDGE_FEATHER_SIGMA_PX = 0.6
OUTLINE_SAMPLES_PER_EDGE = 24


@dataclass(frozen=True)
class PaperLight:
    """One directional light over the sheet and how paper and shadows respond to it."""

    azimuth_radians: float        # direction the light comes FROM, in plane x/y
    elevation_radians: float
    shading_strength: float       # 0 = ignore geometry, 1 = pure Lambert
    base_lift_inches: float       # how far flat paper sits above the sheet (thickness, bumps)
    shadow_strength: float

    def direction(self) -> np.ndarray:
        """Unit vector toward the light, (x, y, z) with z up out of the sheet."""
        cos_elevation = np.cos(self.elevation_radians)
        return np.array([np.cos(self.azimuth_radians) * cos_elevation, np.sin(self.azimuth_radians) * cos_elevation,
                         np.sin(self.elevation_radians)])

    def to_dict(self) -> dict:
        """Plain-JSON record."""
        return {"azimuth_degrees": round(float(np.degrees(self.azimuth_radians)), 1),
                "elevation_degrees": round(float(np.degrees(self.elevation_radians)), 1),
                "shading_strength": round(self.shading_strength, 3), "shadow_strength": round(self.shadow_strength, 3)}


def sample_paper_light(rng: np.random.Generator) -> PaperLight:
    """Room light from above at a moderate angle."""
    return PaperLight(rng.uniform(0, 2 * np.pi), np.deg2rad(rng.uniform(35, 70)), rng.uniform(0.5, 0.9),
                      rng.uniform(0.012, 0.04), rng.uniform(0.2, 0.45))


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


def surface_normals(plane_map: CheckPlaneMap, check_s: np.ndarray, check_t: np.ndarray) -> np.ndarray:
    """Unit paper normals in the plane frame at check inch coordinates (arrays of equal shape)."""
    slope_s, slope_t = plane_map.deformation.height_gradient(check_s, check_t)
    linear = plane_map.linear_part()
    direction = linear / np.linalg.norm(linear[:, 0])  # pure rotation: check axes -> plane axes
    plane_slope_x = direction[0, 0] * slope_s + direction[0, 1] * slope_t
    plane_slope_y = direction[1, 0] * slope_s + direction[1, 1] * slope_t
    normals = np.stack([-plane_slope_x, -plane_slope_y, np.ones_like(slope_s)], axis=-1)
    return normals / np.linalg.norm(normals, axis=-1, keepdims=True)


def shadow_mask(alpha: np.ndarray, height_px: np.ndarray, light: PaperLight, plane_pixels_per_inch: float) -> np.ndarray:
    """Soft shadow coverage for a region: each paper pixel casts along the light by its height."""
    small_size = (max(2, alpha.shape[1] // SHADOW_WORK_DOWNSCALE), max(2, alpha.shape[0] // SHADOW_WORK_DOWNSCALE))
    small_alpha = cv2.resize(alpha, small_size, interpolation=cv2.INTER_AREA)
    lift = (cv2.resize(height_px, small_size, interpolation=cv2.INTER_AREA) + light.base_lift_inches * plane_pixels_per_inch)
    lift = lift * (small_alpha > 0.01) / SHADOW_WORK_DOWNSCALE
    reach = int(np.ceil(lift.max() / np.tan(light.elevation_radians))) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1, 2 * reach + 1))
    lift_nearby = cv2.GaussianBlur(cv2.dilate(lift.astype(np.float32), kernel), (0, 0), sigmaX=max(1.0, reach / 2))
    run = lift_nearby / np.tan(light.elevation_radians)
    grid_y, grid_x = np.mgrid[0:small_size[1], 0:small_size[0]].astype(np.float32)
    # A shadow point p is dark if the paper at p - offset(lift) is there; the light comes from the azimuth.
    source_x = grid_x + np.cos(light.azimuth_radians) * run
    source_y = grid_y + np.sin(light.azimuth_radians) * run
    shifted = cv2.remap(small_alpha, source_x.astype(np.float32), source_y.astype(np.float32), cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT)
    sharp = cv2.GaussianBlur(shifted, (0, 0), sigmaX=1.0)
    soft = cv2.GaussianBlur(shifted, (0, 0), sigmaX=max(1.5, SHADOW_SIGMA_PER_LIFT * float(lift.max())))
    softness = np.clip(lift_nearby / max(1e-3, float(lift.max())), 0, 1)
    small_shadow = sharp * (1 - softness) + soft * softness
    return cv2.resize(small_shadow, (alpha.shape[1], alpha.shape[0]), interpolation=cv2.INTER_LINEAR)


def paste_deformed_check(canvas: np.ndarray, id_map: np.ndarray, check_image: np.ndarray,
                         plane_map: CheckPlaneMap, light: PaperLight, color_multiplier: np.ndarray,
                         owner_id: int) -> tuple[int, int, np.ndarray] | None:
    """Warp one check (float HxWx3 or HxWx4, [0, 1], full render resolution) into the canvas.

    `color_multiplier` (3,) carries the scene light and color cast that fall on this check.

    Returns (x0, y0, alpha) of the canvas region written, or None if the check misses the canvas.
    """
    boundary_plane = plane_map.check_to_plane(check_boundary_points(plane_map.check_width_px, plane_map.check_height_px,
                                                                    OUTLINE_SAMPLES_PER_EDGE))
    max_lift = float(plane_map.deformation.height(np.linspace(0, plane_map.deformation.width_inches, 64)[:, None],
                                                  np.linspace(0, plane_map.deformation.height_inches, 32)[None]).max())
    ppi = plane_map.plane_pixels_per_inch
    pad = int((max_lift + light.base_lift_inches) * ppi / np.tan(light.elevation_radians) * 1.5 + 12)
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

    check_s, check_t = map_x / plane_map.dpi, map_y / plane_map.dpi
    height_px = (plane_map.deformation.height(check_s, check_t) * ppi).astype(np.float32) * (alpha > 0)
    if not plane_map.deformation.is_flat:
        coarse_s, coarse_t = check_s[::4, ::4], check_t[::4, ::4]
        light_direction = light.direction()
        lambert = np.clip(surface_normals(plane_map, coarse_s, coarse_t) @ light_direction, 0, None) / light_direction[2]
        lambert_full = cv2.resize(lambert.astype(np.float32), (alpha.shape[1], alpha.shape[0]), interpolation=cv2.INTER_LINEAR)
        shading = (1 + light.shading_strength * (lambert_full - 1)) * plane_map.deformation.crease_line_shading(check_s, check_t)
        color = color * shading[..., None].astype(np.float32)

    shadow = shadow_mask(alpha, height_px, light, ppi)
    region = canvas[y0:y1, x0:x1]
    region *= (1 - light.shadow_strength * shadow * (1 - alpha))[..., None]
    region[:] = region * (1 - alpha[..., None]) + np.clip(color, 0, 1) * alpha[..., None]
    owned = alpha > 0.5
    id_map[y0:y1, x0:x1][owned] = owner_id
    return x0, y0, alpha
