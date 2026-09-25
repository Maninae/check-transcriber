"""Stage 2: composite several rendered checks onto a background as a phone photo would show them.

Pipeline for one scene (every geometric step has a matching point transform for labels):
1. Layout: lay the checks out on the sheet in inches (placement.py).
2. Framing: choose the photo, the camera, and the sheet scale so the group fills the frame;
   maybe push one check partly out (scene_framing.py).
3. Background: cover-crop onto the sheet-plane canvas; read its light field and color cast.
4. Paper: sample each check's deformation (paper_deformation.py) and build its exact
   check -> plane map (check_plane_map.py).
5. Paste each check: inverse-warped, coverage alpha, geometric shading, height-aware drop
   shadow (check_paste.py). An id map records which check owns each pixel (occlusion).
6. Optional network harmonization (harmonize.py) of each visible check.
7. Camera warp (homography), lens distortion, then photometric effects and JPEG.
8. Labels: corners, outline, orientation, field quads, visibility (scene_check_labels.py).

Determinism: the scene depends only on `rng` (and its inputs).
"""

import cv2
import numpy as np
from PIL import Image

from synth.backgrounds.loader import cover_crop
from synth.compose import lighting
from synth.compose.camera_effects import apply_camera_effects_in_place
from synth.compose.check_paste import paste_deformed_check, sample_paper_light
from synth.compose.check_plane_map import CheckPlaneMap
from synth.compose.paper_deformation import sample_paper_deformation
from synth.compose.perspective import radial_distortion_maps
from synth.compose.placement import plan_layout
from synth.compose.scene_check_labels import build_check_labels
from synth.compose.scene_config import SceneConfig
from synth.compose.scene_framing import frame_scene
from synth.compose.scene_label import SceneLabel
from synth.render.check_fields import CheckLabel

__all__ = ["SceneConfig", "compose_scene", "sample_check_count", "check_image_as_float"]

MAX_PLANE_PIXELS_PER_RENDER_PIXEL = 1.0  # never upsample the rendered paper
TWELVE_CHECK_PROBABILITY = 0.08


def sample_check_count(rng: np.random.Generator) -> int:
    """1 to 8 checks, occasionally 12."""
    return 12 if rng.random() < TWELVE_CHECK_PROBABILITY else int(rng.integers(1, 9))


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
    layout_mode, placements = plan_layout(sizes_inches, config.loose_probability, rng)
    max_ppi = min(label.dpi for label in check_labels) * MAX_PLANE_PIXELS_PER_RENDER_PIXEL
    framing = frame_scene(sizes_inches, placements, config.photo_long_side_range, config.photo_aspect, max_ppi, rng)
    view = framing.view

    canvas = cover_crop(background, view.canvas_width, view.canvas_height, rng).astype(np.float32)
    light_field = lighting.background_light_field(canvas)
    light_gamma = rng.uniform(0.5, 1.0)
    color_gains = lighting.background_color_gains(canvas, rng.uniform(0.25, 0.6))
    paper_exposure = rng.uniform(0.86, 1.0)
    paper_light = sample_paper_light(rng)

    id_map = np.zeros(canvas.shape[:2], np.uint8)
    plane_maps, check_masks = [], []
    for index, ((image, label), placement, size) in enumerate(zip(rendered_checks, placements, sizes_inches)):
        deformation = sample_paper_deformation(*size, rng, config.deformation_strength)
        center_plane = framing.inches_to_plane(np.array([placement.center_x_inches, placement.center_y_inches]))
        plane_map = CheckPlaneMap(label.width_px, label.height_px, label.dpi, (float(center_plane[0]), float(center_plane[1])),
                                  placement.rotation_degrees, framing.plane_pixels_per_inch, deformation,
                                  framing.nadir_plane, framing.camera_height_plane_px)
        plane_maps.append(plane_map)
        # Scene light: the sheet's own low-frequency shading and color cast fall on the paper too.
        center_x = int(np.clip(center_plane[0], 0, canvas.shape[1] - 1)); center_y = int(np.clip(center_plane[1], 0, canvas.shape[0] - 1))
        color_multiplier = color_gains * paper_exposure * light_field[center_y, center_x] ** light_gamma
        pasted = paste_deformed_check(canvas, id_map, check_image_as_float(image), plane_map, paper_light,
                                      color_multiplier.astype(np.float32), index + 1)
        if pasted is not None:
            check_masks.append((index + 1, *pasted))

    if config.harmonize:
        from synth.compose.harmonize import harmonize_pasted_checks  # optional torch backend; --no-harmonize avoids it
        visible_masks = [(x0, y0, alpha * (id_map[y0:y0 + alpha.shape[0], x0:x0 + alpha.shape[1]] == owner_id))
                         for owner_id, x0, y0, alpha in check_masks]
        canvas = harmonize_pasted_checks(canvas, visible_masks, config.harmonize_blend)

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
    effects = apply_camera_effects_in_place(photo, scene_check_labels, config, rng)
    photo = np.clip(photo, 0, 1)
    jpeg_quality = int(rng.integers(*config.jpeg_quality_range))
    photo_uint8 = lighting.jpeg_roundtrip(photo, jpeg_quality)
    effects.update({"camera_tilt_quad_plane": view.visible_quad_plane.round(1).tolist(), "radial_k1": round(view.radial_k1, 4),
                    "light_gamma": round(float(light_gamma), 3), "paper_exposure": round(float(paper_exposure), 3),
                    "pushed_out_check_index": framing.pushed_out_check_index, "framing": framing.to_dict(),
                    "paper_light": paper_light.to_dict()})
    label = SceneLabel(scene_id=scene_id, image_file=f"{scene_id}.jpg", image_width=photo_width, image_height=photo_height,
                       background_id=background_id, layout_mode=layout_mode, harmonized=config.harmonize,
                       jpeg_quality=jpeg_quality, effects=effects, checks=scene_check_labels)
    return photo_uint8, label
