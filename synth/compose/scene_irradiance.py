"""Light the sheet-plane canvas with the scene's `SceneLight`, in linear light.

The canvas holds albedo (background photo plus pasted paper) in sRGB. Three per-pixel
factors, built up while checks are pasted and occluder shadows are cast, say how much of each
light term reaches a pixel:
- key_factor: Lambert of the key on this surface relative to flat sheet, times its visibility
  (drop shadows under paper, cast shadows of a hand or phone). 1 on open flat sheet.
- fill_factor: the same for the fill (no shadows: a window is too broad to cast them).
- ambient_factor: ambient occlusion (the sheet right under a lifted edge sees less of the room).

Irradiance = key_rgb * key_field * key_factor + fill_rgb * fill_field * fill_factor
             + ambient_rgb * ambient_factor, normalized so open flat sheet averages 1 over the
frame, then times exposure and the phone's white-balance gains. Shadows therefore keep the
colour of the ambient and fill, never a neutral grey.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from synth.compose.scene_light import LUMINANCE_WEIGHTS, SceneLight, linear_to_srgb, srgb_to_linear

FIELD_DOWNSAMPLE = 16
MIN_KEY_FIELD = 0.35
APPLY_CHUNK_ROWS = 512
HIGHLIGHT_KNEE = 0.8   # linear level above which highlights roll off instead of clipping


@dataclass
class LightBuffers:
    """Per-pixel light factors over the canvas (float16 to keep memory low; smooth values)."""

    key_factor: np.ndarray
    fill_factor: np.ndarray
    ambient_factor: np.ndarray

    @classmethod
    def open_sheet(cls, height: int, width: int) -> "LightBuffers":
        """All ones: flat, unshadowed sheet everywhere."""
        return cls(*(np.ones((height, width), np.float16) for _ in range(3)))


def highlight_shoulder(linear: np.ndarray) -> np.ndarray:
    """Identity below the knee, then an exponential roll-off toward 1 (phones rarely clip white paper hard)."""
    headroom = 1 - HIGHLIGHT_KNEE
    return np.where(linear < HIGHLIGHT_KNEE, linear,
                    HIGHLIGHT_KNEE + headroom * (1 - np.exp(-(linear - HIGHLIGHT_KNEE) / headroom))).astype(np.float32)


def frame_coordinates(shape: tuple[int, int], visible_quad_plane: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coarse-grid plane coordinates, centred on the photo's view and in units of its width."""
    height, width = shape
    small_height, small_width = max(4, height // FIELD_DOWNSAMPLE), max(4, width // FIELD_DOWNSAMPLE)
    y_grid, x_grid = np.mgrid[0:small_height, 0:small_width].astype(np.float64)
    x_plane, y_plane = (x_grid + 0.5) * width / small_width, (y_grid + 0.5) * height / small_height
    center = visible_quad_plane.mean(axis=0)
    extent = float(np.ptp(visible_quad_plane[:, 0]))
    return (x_plane - center[0]) / extent, (y_plane - center[1]) / extent


def key_and_fill_fields(shape: tuple[int, int], visible_quad_plane: np.ndarray, light: SceneLight) -> tuple[np.ndarray, np.ndarray]:
    """Coarse fields: key strength (brighter toward the key's side), fill strength (fading away from its side)."""
    x_norm, y_norm = frame_coordinates(shape, visible_quad_plane)
    toward_key = np.cos(light.key_azimuth_radians) * x_norm + np.sin(light.key_azimuth_radians) * y_norm
    key_field = np.maximum(MIN_KEY_FIELD, 1 + light.key_falloff * toward_key)
    if light.fill is None:
        return key_field, np.zeros_like(key_field)
    toward_fill = np.cos(light.fill.azimuth_radians) * x_norm + np.sin(light.fill.azimuth_radians) * y_norm
    fill_field = np.clip(1 - light.fill.falloff * (0.5 - toward_fill), 0.05, 1.5)
    return key_field, fill_field


def open_sheet_normalizer(key_field: np.ndarray, fill_field: np.ndarray, light: SceneLight) -> float:
    """Mean luminance of flat unshadowed sheet over the frame (fields are over the canvas; close enough)."""
    irradiance = (key_field[..., None] * light.key_rgb() + fill_field[..., None] * light.fill_rgb() + light.ambient_rgb())
    return float((irradiance @ LUMINANCE_WEIGHTS).mean())


def apply_scene_light(canvas: np.ndarray, buffers: LightBuffers, light: SceneLight, visible_quad_plane: np.ndarray) -> np.ndarray:
    """Return the lit canvas (float32 sRGB). Works in row chunks so full-size temporaries stay small."""
    height, width = canvas.shape[:2]
    key_field, fill_field = key_and_fill_fields((height, width), visible_quad_plane, light)
    scale = light.exposure / open_sheet_normalizer(key_field, fill_field, light)
    gains = np.asarray(light.white_balance_gains) * scale
    key_rgb, fill_rgb, ambient_rgb = [(rgb * gains).astype(np.float32) for rgb in (light.key_rgb(), light.fill_rgb(), light.ambient_rgb())]
    key_full = cv2.resize(key_field.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    fill_full = cv2.resize(fill_field.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    lit = np.empty_like(canvas)
    for row0 in range(0, height, APPLY_CHUNK_ROWS):
        rows = slice(row0, min(height, row0 + APPLY_CHUNK_ROWS))
        key_term = (key_full[rows] * buffers.key_factor[rows])[..., None] * key_rgb
        fill_term = (fill_full[rows] * buffers.fill_factor[rows])[..., None] * fill_rgb
        ambient_term = buffers.ambient_factor[rows].astype(np.float32)[..., None] * ambient_rgb
        irradiance = key_term + fill_term + ambient_term
        lit[rows] = linear_to_srgb(highlight_shoulder(srgb_to_linear(canvas[rows]) * irradiance))
    return lit
