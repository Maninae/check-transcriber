"""Camera-side photometric pass over the finished photo: light falloff, glare, white balance, blur, noise.

Split out of compose_scene.py so the coordinator stays small; the effects themselves live in
lighting.py. Geometry is untouched, so labels are unaffected.
"""

import numpy as np

from synth.compose import lighting
from synth.compose.scene_config import SceneConfig
from synth.compose.scene_label import SceneCheckLabel

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
