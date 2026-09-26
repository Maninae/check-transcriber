"""Stage 2: composite several rendered checks onto a background as a phone photo would show them.

Pipeline for one scene (every geometric step has a matching point transform for labels):
1. Layout: lay the checks out on the sheet in inches (placement.py for the wide regime,
   closeup_placement.py for close and single; `config.framing_regime` picks, framing_regimes.py).
2. Framing: choose the photo, the camera, and the sheet scale so the group fills the frame as the
   regime asks; maybe push one check partly out (scene_framing.py).
3. Background: cover-crop onto the sheet-plane canvas; read its baked light field and colour
   cast; sample the one scene light (scene_light.py), aimed from the background's bright side.
4. Paper: sample each check's deformation (paper_deformation.py) and build its exact
   check -> plane map (check_plane_map.py).
5. Paste each check's albedo: inverse-warped, coverage alpha; its Lambert factors, drop
   shadow and contact occlusion go into the light buffers (check_paste.py, paper_shading.py).
   An id map records which check owns each pixel (occlusion).
6. Optional network harmonization (harmonize.py) of each visible check, on albedo.
7. Maybe a phone/hand shadow (cast_shadows.py), then light the whole canvas at once
   (scene_irradiance.py): sheet, paper and every shadow share the same light.
8. Camera warp (homography), lens distortion, then the phone camera pass and JPEG.
9. Labels: corners, outline, orientation, field quads, visibility (scene_check_labels.py).

Determinism: the scene depends only on `rng` (and its inputs).
"""

import cv2
import numpy as np
from PIL import Image

from scene_composer.camera.camera_effects import apply_camera_effects
from scene_composer.camera.camera_pipeline import jpeg_roundtrip
from scene_composer.geometry.check_paste import paste_deformed_check
from scene_composer.geometry.check_plane_map import CheckPlaneMap
from scene_composer.geometry.closeup_placement import plan_close_layout, plan_single_layout
from scene_composer.geometry.framing_regimes import FramingRegime, parse_framing_regime, sample_close_check_count
from scene_composer.geometry.paper_deformation import sample_paper_deformation
from scene_composer.geometry.perspective import radial_distortion_maps
from scene_composer.geometry.placement import plan_layout
from scene_composer.geometry.scene_check_labels import build_check_labels
from scene_composer.geometry.scene_framing import frame_scene
from scene_composer.lighting import baked_background_light
from scene_composer.lighting.cast_shadows import apply_cast_shadow, cast_occluder_shadow
from scene_composer.lighting.paper_shading import sample_paper_surface
from scene_composer.lighting.scene_irradiance import LightBuffers, apply_scene_light
from scene_composer.lighting.scene_light import sample_scene_light
from scene_composer.scene_config import SceneConfig
from scene_composer.scene_label import SceneLabel
from synthetic_backgrounds.loader import cover_crop
from synthetic_checks.check_fields import CheckLabel

__all__ = ["SceneConfig", "compose_scene", "sample_check_count", "check_image_as_float"]

MAX_PLANE_PIXELS_PER_RENDER_PIXEL = 1.0  # never upsample the rendered paper
TWELVE_CHECK_PROBABILITY = 0.08


def sample_check_count(rng: np.random.Generator, framing_regime: str = FramingRegime.WIDE) -> int:
    """wide: 1 to 8 checks, occasionally 12. close: 2 to 6, mostly 5-6. single: 1 (no draw)."""
    regime = parse_framing_regime(framing_regime)
    if regime == FramingRegime.SINGLE:
        return 1
    if regime == FramingRegime.CLOSE:
        return sample_close_check_count(rng)
    return 12 if rng.random() < TWELVE_CHECK_PROBABILITY else int(rng.integers(1, 9))


def plan_layout_for_regime(sizes_inches: list[tuple[float, float]], framing_regime: FramingRegime, config: SceneConfig,
                           rng: np.random.Generator) -> tuple[str, list]:
    """The regime's layout planner; wide is v1's `plan_layout`, untouched."""
    if framing_regime == FramingRegime.SINGLE:
        return plan_single_layout(sizes_inches, rng)
    if framing_regime == FramingRegime.CLOSE:
        return plan_close_layout(sizes_inches, rng)
    return plan_layout(sizes_inches, config.loose_probability, rng)


def check_image_as_float(image: Image.Image) -> np.ndarray:
    """HxWx3 (RGB, opaque) or HxWx4 (RGBA, alpha = paper coverage) float32 in [0, 1] (contract C2)."""
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")
    return np.asarray(image, np.float32) / 255.0


def compose_scene(
    scene_id: str,
    rendered_checks: list[tuple[Image.Image, CheckLabel]],
    background: np.ndarray,
    background_id: str,
    rng: np.random.Generator,
    config: SceneConfig = SceneConfig(),
) -> tuple[np.ndarray, SceneLabel]:
    """Build one scene. Returns (uint8 RGB photo, label). Deterministic given `rng`'s state."""
    check_labels = [label for _, label in rendered_checks]
    sizes_inches = [(label.width_px / label.dpi, label.height_px / label.dpi) for label in check_labels]
    framing_regime = parse_framing_regime(config.framing_regime)
    layout_mode, placements = plan_layout_for_regime(sizes_inches, framing_regime, config, rng)
    max_ppi = min(label.dpi for label in check_labels) * MAX_PLANE_PIXELS_PER_RENDER_PIXEL
    long_side_range = config.photo_long_side_range if framing_regime == FramingRegime.WIDE else config.closeup_photo_long_side_range
    framing = frame_scene(sizes_inches, placements, long_side_range, config.photo_aspect, max_ppi, rng, framing_regime)
    view = framing.view

    canvas = cover_crop(background, view.canvas_width, view.canvas_height, rng).astype(np.float32)
    light_field = baked_background_light.background_light_field(canvas)
    light_gamma = rng.uniform(0.5, 1.0)
    color_gains = baked_background_light.background_color_gains(canvas, rng.uniform(0.25, 0.6))
    paper_exposure = rng.uniform(0.86, 1.0)
    scene_light = sample_scene_light(light_field, rng)
    paper_surface = sample_paper_surface(rng)

    id_map = np.zeros(canvas.shape[:2], np.uint8)
    light_buffers = LightBuffers.open_sheet(*canvas.shape[:2])
    plane_maps, check_masks = [], []
    for index, ((image, label), placement, size) in enumerate(zip(rendered_checks, placements, sizes_inches)):
        deformation = sample_paper_deformation(*size, rng, config.deformation_strength)
        center_plane = framing.inches_to_plane(np.array([placement.center_x_inches, placement.center_y_inches]))
        plane_map = CheckPlaneMap(label.width_px, label.height_px, label.dpi, (float(center_plane[0]), float(center_plane[1])),
                                  placement.rotation_degrees, framing.plane_pixels_per_inch, deformation,
                                  framing.nadir_plane, framing.camera_height_plane_px)
        plane_maps.append(plane_map)
        # The background's baked light and colour cast fall on the paper too (the scene light comes later).
        center_x = int(np.clip(center_plane[0], 0, canvas.shape[1] - 1)); center_y = int(np.clip(center_plane[1], 0, canvas.shape[0] - 1))
        color_multiplier = color_gains * paper_exposure * light_field[center_y, center_x] ** light_gamma
        pasted = paste_deformed_check(canvas, id_map, light_buffers, check_image_as_float(image), plane_map, scene_light,
                                      paper_surface, color_multiplier.astype(np.float32), index + 1)
        if pasted is not None:
            check_masks.append((index + 1, *pasted))

    if config.harmonize:
        from scene_composer.harmonization.harmonize import harmonize_pasted_checks  # optional torch backend; --no-harmonize avoids it
        visible_masks = [(x0, y0, alpha * (id_map[y0:y0 + alpha.shape[0], x0:x0 + alpha.shape[1]] == owner_id))
                         for owner_id, x0, y0, alpha in check_masks]
        canvas = harmonize_pasted_checks(canvas, visible_masks, config.harmonize_blend)

    cast_shadow_record = {}
    if rng.random() < config.cast_shadow_probability:
        camera_height_inches = framing.camera_height_plane_px / framing.plane_pixels_per_inch
        coarse_shadow, cast_shadow_record = cast_occluder_shadow(canvas.shape[:2], view.visible_quad_plane, framing.nadir_plane,
                                                                 camera_height_inches, framing.plane_pixels_per_inch, id_map,
                                                                 scene_light, rng)
        if coarse_shadow is not None:
            apply_cast_shadow(light_buffers.key_factor, light_buffers.ambient_factor, coarse_shadow)
    canvas = apply_scene_light(canvas, light_buffers, scene_light, view.visible_quad_plane)
    del light_buffers

    photo_width, photo_height = view.photo_width, view.photo_height
    photo = cv2.warpPerspective(canvas, view.plane_to_photo, (photo_width, photo_height), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT)
    photo_ids = cv2.warpPerspective(id_map, view.plane_to_photo, (photo_width, photo_height), flags=cv2.INTER_NEAREST)
    del canvas, id_map
    if view.radial_k1 != 0:
        map_x, map_y = radial_distortion_maps(photo_width, photo_height, view.radial_k1)
        photo = cv2.remap(photo, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        photo_ids = cv2.remap(photo_ids, map_x, map_y, cv2.INTER_NEAREST)

    scene_check_labels = build_check_labels(check_labels, plane_maps, view, photo_ids)
    photo, effects = apply_camera_effects(photo, photo_ids, scene_check_labels, scene_light, config, rng)
    jpeg_quality = int(rng.integers(*config.jpeg_quality_range))
    photo_uint8 = jpeg_roundtrip(photo, jpeg_quality)
    effects.update({"camera_tilt_quad_plane": view.visible_quad_plane.round(1).tolist(), "radial_k1": round(view.radial_k1, 4),
                    "light_gamma": round(float(light_gamma), 3), "paper_exposure": round(float(paper_exposure), 3),
                    "pushed_out_check_index": framing.pushed_out_check_index, "framing": framing.to_dict(),
                    "scene_light": scene_light.to_dict(), "paper_surface": paper_surface.to_dict(),
                    "cast_shadow": cast_shadow_record})
    label = SceneLabel(scene_id=scene_id, image_file=f"{scene_id}.jpg", image_width=photo_width, image_height=photo_height,
                       background_id=background_id, layout_mode=layout_mode, harmonized=config.harmonize,
                       jpeg_quality=jpeg_quality, effects=effects, checks=scene_check_labels)
    return photo_uint8, label
