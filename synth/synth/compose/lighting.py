"""Transfer the background photo's own baked light onto the pasted paper.

A FLUX or real background already carries a lighting of its own (a bright patch, a darker
corner, a colour cast). The paper lying on it must share that: each check's albedo is
multiplied by the background's low-frequency brightness at its position and pushed toward
the sheet's colour cast (inter-reflection). The scene's explicit light model sits on top of
this (scene_light.py, scene_irradiance.py); camera effects live in camera_pipeline.py.
"""

import cv2
import numpy as np

LIGHT_FIELD_DOWNSAMPLE = 32
# Inter-reflection from a coloured surface tints paper a few percent at most; beyond this, a purple
# rug or an orange table dyed the checks (seen on CC0 material swatches, Sep 25).
MAX_INTERREFLECTION_TINT = 0.06


def background_light_field(background: np.ndarray) -> np.ndarray:
    """Low-frequency luminance of the background, normalized to mean 1 (same size as input)."""
    height, width = background.shape[:2]
    small = cv2.resize(background, (max(4, width // LIGHT_FIELD_DOWNSAMPLE), max(4, height // LIGHT_FIELD_DOWNSAMPLE)),
                       interpolation=cv2.INTER_AREA)
    luminance = small @ np.array([0.299, 0.587, 0.114], np.float32)
    luminance = cv2.GaussianBlur(luminance, (0, 0), sigmaX=max(small.shape) / 10)
    field = cv2.resize(luminance, (width, height), interpolation=cv2.INTER_CUBIC)
    return (field / max(1e-4, float(field.mean()))).astype(np.float32)


def background_color_gains(background: np.ndarray, strength: float) -> np.ndarray:
    """Per-channel gains that push white paper slightly toward the background's colour (capped inter-reflection)."""
    mean_color = background[::8, ::8].reshape(-1, 3).mean(axis=0)
    gains = mean_color / max(1e-4, float(mean_color.mean()))
    tint = np.clip((gains - 1) * strength, -MAX_INTERREFLECTION_TINT, MAX_INTERREFLECTION_TINT)
    return (1 + tint).astype(np.float32)
