"""Stage 2: composite several rendered checks onto a background as a phone photo would show them.

Pipeline for one scene (every geometric step has a matching point transform for labels):
1. Camera: pick photo size and a tilted view; the sheet-plane canvas is what the camera sees.
2. Background: cover-crop onto the canvas; read its light field and color cast.
3. Placement: grid or pile layout (placement.py), maybe one check pushed out of view.
4. Paste each check: curl shading, scene light and color cast, soft drop shadow,
   feathered edge. An id map records which check owns each pixel (for occlusion).
5. Optional network harmonization (harmonize.py) of each visible check.
6. Camera warp (homography), lens distortion, then photometric effects and JPEG.
7. Labels: corners, orientation, field quads, clipped boxes, visibility, in photo pixels.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from synth.compose import lighting
from synth.compose.perspective import (
    apply_affine,
    clip_polygon_to_rect,
    inscribed_rect_of_quad,
    plane_points_to_photo,
    polygon_area,
    radial_distortion_maps,
    sample_camera_view,
)
from synth.compose.placement import (
    OUT_OF_FRAME_PROBABILITY,
    plan_grid_layout,
    plan_pile_layout,
    push_one_check_out_of_frame,
)
from synth.compose.scene_label import SceneCheckLabel, SceneFieldLabel, SceneLabel
from synth.backgrounds.loader import cover_crop
from synth.render.check_fields import CheckLabel


@dataclass(frozen=True)
class SceneConfig:
    """Knobs for scene composition; defaults aim at typical phone photos of a sheet."""

    photo_long_side_range: tuple[int, int] = (2000, 3200)
    photo_aspect: float = 4 / 3
    portrait_probability: float = 0.35
    pile_probability: float = 0.3
    curl_probability: float = 0.4
    glare_probability: float = 0.5
    motion_blur_probability: float = 0.25
    lens_blur_probability: float = 0.7
    sharpen_probability: float = 0.7
    jpeg_quality_range: tuple[int, int] = (68, 92)
    harmonize: bool = False
    harmonize_blend: float = 0.5


def sample_check_count(rng: np.random.Generator) -> int:
    """1 to 8 checks, occasionally 12."""
    return 12 if rng.random() < 0.08 else int(rng.integers(1, 9))


def sample_photo_size(config: SceneConfig, rng: np.random.Generator) -> tuple[int, int]:
    """Photo width and height in pixels."""
    long_side = int(rng.integers(*config.photo_long_side_range))
    short_side = int(round(long_side / config.photo_aspect))
    return (short_side, long_side) if rng.random() < config.portrait_probability else (long_side, short_side)


def check_to_plane_affine(check_width: int, check_height: int, small_width: int, small_height: int,
                          center_x: float, center_y: float, rotation_degrees: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (affine for the downscaled check image, affine for full-resolution check points)."""
    rotation = cv2.getRotationMatrix2D((small_width / 2, small_height / 2), rotation_degrees, 1.0)
    rotation[:, 2] += (center_x - small_width / 2, center_y - small_height / 2)
    scale = np.diag([small_width / check_width, small_height / check_height])
    full_resolution = np.hstack([rotation[:, :2] @ scale, rotation[:, 2:3]])
    return rotation, full_resolution


def box_corners(box: tuple[int, int, int, int]) -> np.ndarray:
    """(x0, y0, x1, y1) -> 4 corners TL, TR, BR, BL."""
    x0, y0, x1, y1 = box
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float64)


def owned_fraction_of_polygon(id_map: np.ndarray, polygon: np.ndarray, owner_id: int) -> float:
    """Share of `polygon`'s full area whose photo pixels belong to `owner_id`."""
    full_area = polygon_area(polygon)
    if full_area < 1:
        return 0.0
    height, width = id_map.shape
    x0, y0 = np.floor(np.clip(polygon.min(axis=0), 0, [width, height])).astype(int)
    x1, y1 = np.ceil(np.clip(polygon.max(axis=0), 0, [width, height])).astype(int)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(mask, [np.round(polygon - [x0, y0]).astype(np.int32)], 1)
    owned = np.count_nonzero(mask & (id_map[y0:y1, x0:x1] == owner_id))
    return float(min(1.0, owned / full_area))


def paste_check(canvas: np.ndarray, id_map: np.ndarray, check_rgb: np.ndarray, rotation: np.ndarray,
                small_size: tuple[int, int], shadow: dict, owner_id: int) -> tuple[int, int, np.ndarray] | None:
    """Warp one (already scaled) check into the canvas with shadow and feathering.

    Returns (x0, y0, alpha) for the canvas region touched, or None if it misses the canvas.
    """
    small_width, small_height = small_size
    corners = apply_affine(box_corners((0, 0, small_width, small_height)), rotation)
    pad = int(shadow["blur_sigma"] * 3 + np.hypot(*shadow["offset"]) + 2)
    canvas_height, canvas_width = canvas.shape[:2]
    x0 = max(0, int(np.floor(corners[:, 0].min())) - pad); x1 = min(canvas_width, int(np.ceil(corners[:, 0].max())) + pad)
    y0 = max(0, int(np.floor(corners[:, 1].min())) - pad); y1 = min(canvas_height, int(np.ceil(corners[:, 1].max())) + pad)
    if x1 <= x0 or y1 <= y0:
        return None
    local = rotation.copy()
    local[:, 2] -= (x0, y0)
    region_size = (x1 - x0, y1 - y0)
    warped = cv2.warpAffine(check_rgb, local, region_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    alpha = cv2.warpAffine(np.ones((small_height, small_width), np.float32), local, region_size, flags=cv2.INTER_LINEAR)
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=0.7) * alpha  # feather inward only, never grow past the edge

    shift = np.float32([[1, 0, shadow["offset"][0]], [0, 1, shadow["offset"][1]]])
    shadow_mask = cv2.warpAffine(cv2.GaussianBlur(alpha, (0, 0), sigmaX=shadow["blur_sigma"]), shift, region_size)
    region = canvas[y0:y1, x0:x1]
    region *= (1 - shadow["strength"] * shadow_mask * (1 - alpha))[..., None]
    region[:] = region * (1 - alpha[..., None]) + warped * alpha[..., None]
    id_map[y0:y1, x0:x1][alpha > 0.5] = owner_id
    return x0, y0, alpha


def compose_scene(
    scene_id: str,
    rendered_checks: list[tuple[Image.Image, CheckLabel]],
    background: np.ndarray,
    background_id: str,
    rng: np.random.Generator,
    config: SceneConfig = SceneConfig(),
) -> tuple[np.ndarray, SceneLabel]:
    """Build one scene. Returns (uint8 RGB photo, label). Deterministic given `rng`'s state."""
    photo_width, photo_height = sample_photo_size(config, rng)
    view = sample_camera_view(photo_width, photo_height, rng)
    canvas = cover_crop(background, view.canvas_width, view.canvas_height, rng).astype(np.float32)
    light_field = lighting.background_light_field(canvas)
    light_gamma = rng.uniform(0.5, 1.0)
    color_gains = lighting.background_color_gains(canvas, rng.uniform(0.25, 0.6))
    paper_exposure = rng.uniform(0.86, 1.0)

    safe_rect = inscribed_rect_of_quad(view.visible_quad_plane)
    sizes_inches = [(label.width_px / label.dpi, label.height_px / label.dpi) for _, label in rendered_checks]
    layout_mode = "pile" if rng.random() < config.pile_probability else "grid"
    planner = plan_pile_layout if layout_mode == "pile" else plan_grid_layout
    placements = planner(sizes_inches, safe_rect, rng)
    pushed_out_index = push_one_check_out_of_frame(placements, sizes_inches, safe_rect, rng) if rng.random() < OUT_OF_FRAME_PROBABILITY else None

    light_angle = rng.uniform(0, 2 * np.pi)
    id_map = np.zeros(canvas.shape[:2], np.uint8)
    check_affines, check_masks = [], []
    for index, ((image, label), placement) in enumerate(zip(rendered_checks, placements)):
        check_rgb = np.asarray(image, np.float32) / 255.0
        if rng.random() < config.curl_probability:
            check_rgb = lighting.curl_shading(check_rgb, rng)
        scale = placement.pixels_per_inch / label.dpi
        small_size = (max(8, int(round(label.width_px * scale))), max(4, int(round(label.height_px * scale))))
        small = cv2.resize(check_rgb, small_size, interpolation=cv2.INTER_AREA)
        rotation, full_affine = check_to_plane_affine(label.width_px, label.height_px, *small_size,
                                                      placement.center_x, placement.center_y, placement.rotation_degrees)
        check_affines.append(full_affine)
        # Scene light: the sheet's own low-frequency shading and color cast fall on the paper too.
        center_x = int(np.clip(placement.center_x, 0, canvas.shape[1] - 1)); center_y = int(np.clip(placement.center_y, 0, canvas.shape[0] - 1))
        small = np.clip(small * color_gains * paper_exposure * light_field[center_y, center_x] ** light_gamma, 0, 1)
        lift_px = rng.uniform(1.5, 6.0) * placement.pixels_per_inch / 100
        shadow = {"blur_sigma": max(1.0, lift_px * 1.5),
                  "offset": (np.cos(light_angle) * lift_px, np.sin(light_angle) * lift_px),
                  "strength": rng.uniform(0.2, 0.45)}
        pasted = paste_check(canvas, id_map, small, rotation, small_size, shadow, index + 1)
        if pasted is not None:
            check_masks.append((index + 1, *pasted))

    if config.harmonize:
        from synth.compose.harmonize import harmonize_pasted_checks  # optional torch backend; --no-harmonize avoids it
        visible_masks = [(x0, y0, alpha * (id_map[y0:y0 + alpha.shape[0], x0:x0 + alpha.shape[1]] == owner_id))
                         for owner_id, x0, y0, alpha in check_masks]
        canvas = harmonize_pasted_checks(canvas, visible_masks, config.harmonize_blend)

    photo = cv2.warpPerspective(canvas, view.plane_to_photo, (photo_width, photo_height), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT)
    photo_ids = cv2.warpPerspective(id_map, view.plane_to_photo, (photo_width, photo_height), flags=cv2.INTER_NEAREST)
    if view.radial_k1 != 0:
        map_x, map_y = radial_distortion_maps(photo_width, photo_height, view.radial_k1)
        photo = cv2.remap(photo, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        photo_ids = cv2.remap(photo_ids, map_x, map_y, cv2.INTER_NEAREST)

    check_labels = build_check_labels(rendered_checks, check_affines, view, photo_ids)
    effects = apply_camera_effects_in_place(photo, check_labels, config, rng)
    photo = np.clip(photo, 0, 1)
    jpeg_quality = int(rng.integers(*config.jpeg_quality_range))
    photo_uint8 = lighting.jpeg_roundtrip(photo, jpeg_quality)
    effects.update({"camera_tilt_quad_plane": view.visible_quad_plane.round(1).tolist(), "radial_k1": round(view.radial_k1, 4),
                    "light_gamma": round(float(light_gamma), 3), "paper_exposure": round(float(paper_exposure), 3),
                    "pushed_out_check_index": pushed_out_index})
    label = SceneLabel(scene_id=scene_id, image_file=f"{scene_id}.jpg", image_width=photo_width, image_height=photo_height,
                       background_id=background_id, layout_mode=layout_mode, harmonized=config.harmonize,
                       jpeg_quality=jpeg_quality, effects=effects, checks=check_labels)
    return photo_uint8, label


def build_check_labels(rendered_checks, check_affines, view, photo_ids) -> list[SceneCheckLabel]:
    """Project every check and field through plane -> photo and measure framing and occlusion."""
    labels = []
    for index, ((_, check_label), affine) in enumerate(zip(rendered_checks, check_affines)):
        corners = plane_points_to_photo(apply_affine(box_corners((0, 0, check_label.width_px, check_label.height_px)), affine), view)
        full_area = polygon_area(corners)
        clipped = clip_polygon_to_rect(corners, view.photo_width, view.photo_height)
        in_frame_fraction = polygon_area(clipped) / full_area if full_area else 0.0
        top_edge = corners[1] - corners[0]
        rotation_clockwise = float(np.degrees(np.arctan2(top_edge[1], top_edge[0])) % 360)
        fields = []
        for field_label in check_label.fields:
            quad = plane_points_to_photo(apply_affine(box_corners(field_label.box), affine), view)
            clipped_quad = clip_polygon_to_rect(quad, view.photo_width, view.photo_height)
            bbox = None
            if len(clipped_quad) >= 3 and polygon_area(clipped_quad) > 0:
                bbox = [round(float(v), 2) for v in (*clipped_quad.min(axis=0), *clipped_quad.max(axis=0))]
            fields.append(SceneFieldLabel(field_label.field_name, field_label.text, field_label.handwritten,
                                          quad.round(2).tolist(), bbox,
                                          round(owned_fraction_of_polygon(photo_ids, quad, index + 1), 4)))
        labels.append(SceneCheckLabel(
            check_index=index, template_id=check_label.template_id, size_kind=check_label.size_kind,
            corners=corners.round(2).tolist(), rotation_degrees_clockwise=round(rotation_clockwise, 2),
            orientation_class=int(round(rotation_clockwise / 90) % 4 * 90),
            fully_in_frame=bool(in_frame_fraction > 0.999), in_frame_fraction=round(float(in_frame_fraction), 4),
            visible_fraction=round(owned_fraction_of_polygon(photo_ids, corners, index + 1), 4),
            fields=fields, canonical=check_label.canonical))
    return labels


def apply_camera_effects_in_place(photo: np.ndarray, check_labels: list[SceneCheckLabel], config: SceneConfig,
                                  rng: np.random.Generator) -> dict:
    """Illumination, glare, white balance, blur, sharpening, noise. Mutates `photo`; returns what ran."""
    effects = {}
    photo[:] = lighting.illumination_gradient(photo, rng)
    effects["illumination_gradient"] = True
    if rng.random() < config.glare_probability:
        visible_checks = [c for c in check_labels if c.visible_fraction > 0.3]
        if visible_checks and rng.random() < 0.6:
            corners = np.array(visible_checks[int(rng.integers(len(visible_checks)))].corners)
            center = corners.mean(axis=0) + rng.uniform(-0.3, 0.3, 2) * (corners.max(axis=0) - corners.min(axis=0))
        else:
            center = rng.uniform(0.1, 0.9, 2) * [photo.shape[1], photo.shape[0]]
        photo[:] = lighting.glare_spot(photo, (float(center[0]), float(center[1])), rng)
        effects["glare_center"] = [round(float(v), 1) for v in center]
    photo[:] = lighting.white_balance_jitter(photo, rng)
    if rng.random() < config.lens_blur_probability:
        photo[:] = lighting.lens_blur(photo, rng)
        effects["lens_blur"] = True
    if rng.random() < config.motion_blur_probability:
        photo[:] = lighting.motion_blur(photo, rng)
        effects["motion_blur"] = True
    if rng.random() < config.sharpen_probability:
        photo[:] = lighting.phone_sharpening(photo, rng)
        effects["sharpening"] = True
    photo[:] = lighting.sensor_noise(photo, rng)
    return effects
