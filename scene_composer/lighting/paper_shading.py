"""How the scene light falls on one pasted check and on the sheet around it.

Given a check's region on the canvas (its inverse-mapped check coordinates and coverage
alpha), this module produces the per-pixel factors that scene_irradiance.py multiplies into
the key, fill and ambient terms:
- geometry: Lambert of key and fill on the paper's normal from the height map, relative to
  flat sheet, so a fold panel tilted toward the key brightens and the other darkens.
- crease ridge: broken fibres leave a narrow ridge (mountain crease) or groove (valley) along
  each fold line; its two flanks tilt opposite ways, so one side catches light and the other
  darkens. It only changes shading, never the geometry or labels.
- drop shadow: the paper's silhouette shifted away from the key by (base lift + height) /
  tan(elevation), penumbra widening with height by the key's angular size; it removes key
  light only, so shadows keep the ambient colour.
- contact occlusion: the sheet right under and beside a lifted edge sees less of the room.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from scene_composer.geometry.check_plane_map import CheckPlaneMap
from scene_composer.lighting.scene_light import SceneLight

SHADOW_WORK_DOWNSCALE = 4
LAMBERT_WORK_STEP = 4
BASE_LIFT_INCHES_RANGE = (0.015, 0.045)
DROP_SHADOW_OPACITY_RANGE = (0.7, 0.95)
CREASE_RIDGE_SLOPE_RANGE = (0.25, 0.6)
CREASE_RIDGE_HALF_WIDTH_INCHES_RANGE = (0.015, 0.035)
CONTACT_OCCLUSION_RANGE = (0.25, 0.55)
CONTACT_REACH_INCHES = 0.12            # how far beside flat paper the room light is dimmed
LIFT_FOR_FULL_OCCLUSION_INCHES = 0.25
RIDGE_BUMP_TO_PEAK_SLOPE = 0.8578      # max |d/dx H exp(-(x/w)^2)| = 0.8578 H / w


@dataclass(frozen=True)
class PaperSurface:
    """How paper in this scene meets the light (shared by every check in the scene)."""

    base_lift_inches: float           # flat paper still stands this high (thickness, sheet texture)
    drop_shadow_opacity: float        # share of key light a fully shadowing sheet of paper blocks
    crease_ridge_slope: float         # peak slope of the ridge along a fold line
    crease_ridge_half_width_inches: float
    contact_occlusion: float

    def to_dict(self) -> dict:
        """Plain-JSON record."""
        return {key: round(float(value), 4) for key, value in self.__dict__.items()}


def sample_paper_surface(rng: np.random.Generator) -> PaperSurface:
    """Draw the scene's paper response."""
    return PaperSurface(rng.uniform(*BASE_LIFT_INCHES_RANGE), rng.uniform(*DROP_SHADOW_OPACITY_RANGE),
                        rng.uniform(*CREASE_RIDGE_SLOPE_RANGE), rng.uniform(*CREASE_RIDGE_HALF_WIDTH_INCHES_RANGE),
                        rng.uniform(*CONTACT_OCCLUSION_RANGE))


def check_axes_in_plane(plane_map: CheckPlaneMap) -> np.ndarray:
    """2x2 rotation whose columns are the check's s and t axes in the plane frame."""
    linear = plane_map.linear_part()
    return linear / np.linalg.norm(linear[:, 0])


def plane_slopes(plane_map: CheckPlaneMap, check_s: np.ndarray, check_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Height slopes (dz/dx, dz/dy) of the paper in the plane frame at check inch coordinates."""
    slope_s, slope_t = plane_map.deformation.height_gradient(check_s, check_t)
    axes = check_axes_in_plane(plane_map)
    return axes[0, 0] * slope_s + axes[0, 1] * slope_t, axes[1, 0] * slope_s + axes[1, 1] * slope_t


def lambert_relative_to_flat(slope_x: np.ndarray, slope_y: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """max(0, n . L) / L_z for the normal (-slope_x, -slope_y, 1) normalized."""
    dot = (-slope_x * direction[0] - slope_y * direction[1] + direction[2]) / np.sqrt(1 + slope_x**2 + slope_y**2)
    return np.clip(dot, 0, None) / direction[2]


def crease_ridge_slope(plane_map: CheckPlaneMap, check_s: np.ndarray, check_t: np.ndarray, surface: PaperSurface) -> np.ndarray:
    """Extra slope along the fold axis from the narrow ridge or groove at each crease (0 without a fold)."""
    fold = plane_map.deformation.fold
    if fold is None:
        return np.zeros_like(check_s)
    coordinate = check_s if fold["axis"] == 0 else check_t
    knots, heights = np.array(fold["knots"]), np.array(fold["heights"])
    slopes = np.diff(heights) / np.diff(knots)
    width = surface.crease_ridge_half_width_inches
    bump = surface.crease_ridge_slope * width / RIDGE_BUMP_TO_PEAK_SLOPE
    extra = np.zeros_like(coordinate)
    for knot, slope_change in zip(knots[1:-1], np.diff(slopes)):
        mountain = -1.0 if slope_change > 0 else 1.0   # slope rising through the knot is a valley
        distance = (coordinate - knot) / width
        extra += mountain * bump * (-2 * distance / width) * np.exp(-distance**2)
    return extra


def paper_geometry_factors(plane_map: CheckPlaneMap, map_x: np.ndarray, map_y: np.ndarray, light: SceneLight,
                           surface: PaperSurface) -> tuple[np.ndarray, np.ndarray]:
    """Key and fill Lambert factors (relative to flat sheet) over a region, from the height map and creases.

    The smooth part is evaluated on a coarse grid and upsampled; the narrow crease ridge at full resolution.
    """
    shape = map_x.shape
    if plane_map.deformation.is_flat:
        return np.ones(shape, np.float32), np.ones(shape, np.float32)
    check_s, check_t = map_x / plane_map.dpi, map_y / plane_map.dpi
    coarse_slope_x, coarse_slope_y = plane_slopes(plane_map, check_s[::LAMBERT_WORK_STEP, ::LAMBERT_WORK_STEP],
                                                  check_t[::LAMBERT_WORK_STEP, ::LAMBERT_WORK_STEP])
    slope_x = cv2.resize(coarse_slope_x.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    slope_y = cv2.resize(coarse_slope_y.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    fold = plane_map.deformation.fold
    if fold is not None:
        ridge = crease_ridge_slope(plane_map, check_s, check_t, surface).astype(np.float32)
        axis_in_plane = check_axes_in_plane(plane_map)[:, fold["axis"]]
        slope_x = slope_x + ridge * np.float32(axis_in_plane[0])
        slope_y = slope_y + ridge * np.float32(axis_in_plane[1])
    key = lambert_relative_to_flat(slope_x, slope_y, light.key_direction())
    fill = np.ones(shape, np.float32) if light.fill is None else lambert_relative_to_flat(slope_x, slope_y, light.fill.direction())
    creases = plane_map.deformation.crease_line_shading(check_s, check_t).astype(np.float32)
    return (key * creases).astype(np.float32), (fill * creases).astype(np.float32)


def drop_shadow(alpha: np.ndarray, height_px: np.ndarray, light: SceneLight, surface: PaperSurface,
                plane_pixels_per_inch: float) -> np.ndarray:
    """Soft shadow coverage [0, 1] over a region: each paper pixel casts away from the key by its height."""
    small_size = (max(2, alpha.shape[1] // SHADOW_WORK_DOWNSCALE), max(2, alpha.shape[0] // SHADOW_WORK_DOWNSCALE))
    small_alpha = cv2.resize(alpha, small_size, interpolation=cv2.INTER_AREA)
    lift = cv2.resize(height_px, small_size, interpolation=cv2.INTER_AREA) + surface.base_lift_inches * plane_pixels_per_inch
    lift = lift * (small_alpha > 0.01) / SHADOW_WORK_DOWNSCALE
    tan_elevation = np.tan(light.key_elevation_radians)
    reach = int(np.ceil(lift.max() / tan_elevation)) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1, 2 * reach + 1))
    lift_nearby = cv2.GaussianBlur(cv2.dilate(lift.astype(np.float32), kernel), (0, 0), sigmaX=max(1.0, reach / 2))
    run = lift_nearby / tan_elevation
    grid_y, grid_x = np.mgrid[0:small_size[1], 0:small_size[0]].astype(np.float32)
    # A sheet point is shadowed if the paper sits between it and the key: sample alpha toward the key.
    source_x = grid_x + np.cos(light.key_azimuth_radians) * run
    source_y = grid_y + np.sin(light.key_azimuth_radians) * run
    shifted = cv2.remap(small_alpha, source_x.astype(np.float32), source_y.astype(np.float32), cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT)
    # Penumbra grows with the occluder's distance from the sheet along the ray, times the source's angular size.
    penumbra_sigma_near = max(0.7, float(np.median(lift[lift > 0])) / np.sin(light.key_elevation_radians)
                              * light.key_angular_radius) if (lift > 0).any() else 0.7
    penumbra_sigma_far = max(penumbra_sigma_near, float(lift.max()) / np.sin(light.key_elevation_radians) * light.key_angular_radius)
    sharp = cv2.GaussianBlur(shifted, (0, 0), sigmaX=penumbra_sigma_near)
    soft = cv2.GaussianBlur(shifted, (0, 0), sigmaX=penumbra_sigma_far)
    softness = np.clip(lift_nearby / max(1e-3, float(lift.max())), 0, 1)
    small_shadow = sharp * (1 - softness) + soft * softness
    return cv2.resize(small_shadow, (alpha.shape[1], alpha.shape[0]), interpolation=cv2.INTER_LINEAR)


def contact_occlusion(alpha: np.ndarray, height_px: np.ndarray, surface: PaperSurface, plane_pixels_per_inch: float) -> np.ndarray:
    """Share of ambient light lost on the sheet beside and under the paper, highest under lifted parts."""
    lift_weight = np.clip(height_px / (LIFT_FOR_FULL_OCCLUSION_INCHES * plane_pixels_per_inch), 0, 1)
    occluder = alpha * (0.35 + 0.65 * lift_weight)
    small_size = (max(2, alpha.shape[1] // SHADOW_WORK_DOWNSCALE), max(2, alpha.shape[0] // SHADOW_WORK_DOWNSCALE))
    small = cv2.resize(occluder.astype(np.float32), small_size, interpolation=cv2.INTER_AREA)
    sigma = max(1.0, CONTACT_REACH_INCHES * plane_pixels_per_inch / SHADOW_WORK_DOWNSCALE)
    spread = cv2.GaussianBlur(small, (0, 0), sigmaX=sigma)
    return surface.contact_occlusion * cv2.resize(spread, (alpha.shape[1], alpha.shape[0]), interpolation=cv2.INTER_LINEAR)
