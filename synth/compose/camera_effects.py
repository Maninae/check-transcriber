"""Sample and run the phone camera pass over the finished photo (ops live in camera_pipeline.py).

Most scenes come out sharp and processed-looking, as phone photos do; some get defocus or
hand-shake blur, a few a broad sheen on the paper. Geometry is untouched, so labels are
unaffected. Returns the photo and a record of what ran for the scene label's `effects`.
"""

import numpy as np

from synth.compose import camera_pipeline
from synth.compose.scene_config import SceneConfig
from synth.compose.scene_label import SceneCheckLabel
from synth.compose.scene_light import SceneLight

VIGNETTE_RANGE = (0.05, 0.22)
SHEEN_RADIUS_FRACTION_RANGE = (0.12, 0.3)
SHEEN_INTENSITY_RANGE = (0.03, 0.12)
DEFOCUS_SIGMA_RANGE = (0.7, 1.8)
MOTION_LENGTH_RANGE = (3, 7)
SHOT_NOISE_RANGE = (0.0004, 0.004)       # linear variance per unit signal (ISO)
READ_NOISE_RANGE = (0.0006, 0.003)
CHROMA_NOISE_RANGE = (0.002, 0.01)
BASE_COMPRESSION_RANGE = (0.6, 0.9)
DETAIL_BOOST_RANGE = (1.0, 1.25)
CONTRAST_RANGE = (0.1, 0.4)
CHROMA_DENOISE_SIGMA_RANGE = (0.8, 2.0)
SHARPEN_SIGMA_RANGE = (0.8, 1.5)
SHARPEN_AMOUNT_RANGE = (0.35, 1.0)


def sheen_center(check_labels: list[SceneCheckLabel], photo_shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """Usually over a visible check, otherwise anywhere."""
    visible_checks = [c for c in check_labels if c.visible_fraction > 0.3]
    if visible_checks and rng.random() < 0.8:
        corners = np.array(visible_checks[int(rng.integers(len(visible_checks)))].corners)
        return corners.mean(axis=0) + rng.uniform(-0.3, 0.3, 2) * (corners.max(axis=0) - corners.min(axis=0))
    return rng.uniform(0.1, 0.9, 2) * [photo_shape[1], photo_shape[0]]


def apply_camera_effects(photo: np.ndarray, photo_ids: np.ndarray, check_labels: list[SceneCheckLabel], light: SceneLight,
                         config: SceneConfig, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Optics, sensor, ISP on a float32 sRGB photo. Returns (float32 sRGB photo, effects record)."""
    effects = {}
    linear = camera_pipeline.decode_srgb(photo)
    vignette = rng.uniform(*VIGNETTE_RANGE)
    linear = camera_pipeline.lens_vignette(linear, vignette)
    effects["vignette"] = round(float(vignette), 3)
    if rng.random() < config.glare_probability:
        center = sheen_center(check_labels, photo.shape[:2], rng)
        radius = rng.uniform(*SHEEN_RADIUS_FRACTION_RANGE) * max(photo.shape[:2])
        intensity = rng.uniform(*SHEEN_INTENSITY_RANGE)
        light_rgb = light.key_rgb() * np.asarray(light.white_balance_gains)
        linear = camera_pipeline.paper_sheen(linear, photo_ids > 0, (float(center[0]), float(center[1])), radius, intensity, light_rgb)
        effects["sheen"] = {"center": [round(float(v), 1) for v in center], "radius": round(float(radius), 1),
                            "intensity": round(float(intensity), 3)}
    blur_draw = rng.random()
    if blur_draw < config.lens_blur_probability:
        sigma = rng.uniform(*DEFOCUS_SIGMA_RANGE)
        linear = camera_pipeline.defocus_blur(linear, sigma, rng)
        effects["defocus_sigma"] = round(float(sigma), 2)
    elif blur_draw < config.lens_blur_probability + config.motion_blur_probability:
        length, angle = int(rng.integers(*MOTION_LENGTH_RANGE)), rng.uniform(0, 180)
        linear = camera_pipeline.motion_blur(linear, length, angle)
        effects["motion_blur"] = {"length_px": length, "angle_degrees": round(float(angle), 1)}
    noise = {"shot": rng.uniform(*SHOT_NOISE_RANGE), "read": rng.uniform(*READ_NOISE_RANGE), "chroma": rng.uniform(*CHROMA_NOISE_RANGE)}
    linear = camera_pipeline.sensor_noise(linear, noise["shot"], noise["read"], noise["chroma"], rng)
    tone = {"base_compression": rng.uniform(*BASE_COMPRESSION_RANGE), "detail_boost": rng.uniform(*DETAIL_BOOST_RANGE),
            "contrast": rng.uniform(*CONTRAST_RANGE)}
    linear = camera_pipeline.local_tone_map(linear, tone["base_compression"], tone["detail_boost"])
    srgb = camera_pipeline.contrast_curve(camera_pipeline.encode_srgb(linear), tone["contrast"])
    del linear
    sharpen_amount = rng.uniform(*SHARPEN_AMOUNT_RANGE) if rng.random() < config.sharpen_probability else 0.0
    finish = {"chroma_denoise_sigma": rng.uniform(*CHROMA_DENOISE_SIGMA_RANGE), "sharpen_sigma": rng.uniform(*SHARPEN_SIGMA_RANGE),
              "sharpen_amount": sharpen_amount}
    srgb = camera_pipeline.chroma_denoise_and_sharpen(srgb, finish["chroma_denoise_sigma"], finish["sharpen_sigma"], sharpen_amount)
    effects.update({"noise": noise, "tone_map": tone, "isp_finish": finish})
    effects = {key: ({k: round(float(v), 4) for k, v in value.items()} if key in ("noise", "tone_map", "isp_finish") else value)
               for key, value in effects.items()}
    return srgb, effects
