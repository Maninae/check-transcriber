"""The one light model a scene is lit by: a key source, an optional fill of another colour, ambient.

Everything photometric in a scene derives from `SceneLight`: the sheet's brightness falloff,
the paper's Lambert shading (paper_shading.py), the drop shadows under checks, the cast
shadows of a hand or phone (cast_shadows.py), and the camera's white balance.

- Directions live in the sheet plane frame (x right, y down, z up out of the sheet);
  azimuth is the direction the light comes FROM.
- Colours are linear RGB of a black body at a colour temperature, normalized to unit
  luminance, so a warm lamp and a cool window differ in hue but not in brightness.
- The key's azimuth follows the background's own baked brightness slope when it has one,
  so a FLUX sheet lit from the left gets paper and shadows lit from the left too.
"""

from dataclasses import dataclass

import cv2
import numpy as np

# Black-body colours (Mitchell Charity's table, sRGB-encoded 0-255), linearized at load time.
BLACK_BODY_KELVIN = np.array([2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 8000, 9000, 10000], np.float64)
BLACK_BODY_SRGB = np.array([
    [255, 137, 18],
    [255, 161, 72],
    [255, 180, 107],
    [255, 196, 137],
    [255, 209, 163],
    [255, 219, 186],
    [255, 228, 206],
    [255, 236, 224],
    [255, 243, 239],
    [255, 249, 253],
    [245, 243, 255],
    [227, 233, 255],
    [214, 225, 255],
    [204, 219, 255],
], np.float64) / 255.0
LUMINANCE_WEIGHTS = np.array([0.2126, 0.7152, 0.0722])

KEY_KELVIN_CHOICES = ((2700, 3300), (3800, 4800), (5000, 6500))   # tungsten lamp, LED/fluorescent, daylight
KEY_KELVIN_WEIGHTS = (0.35, 0.35, 0.30)
KEY_ELEVATION_DEGREES = (32.0, 78.0)
KEY_ANGULAR_RADIUS_RANGE = (0.03, 0.22)      # radians: bare bulb or sun .. big diffuser
AMBIENT_FRACTION_RANGE = (0.28, 0.7)          # ambient irradiance / key irradiance on flat sheet
KEY_FALLOFF_RANGE = (0.04, 0.28)              # brightness change across one frame width, toward the key
MIXED_LIGHT_PROBABILITY = 0.45
FILL_INTENSITY_RANGE = (0.2, 0.65)
FILL_KELVIN_OFFSET_RANGE = (1500, 3500)       # the fill differs from the key by this much, either way
FILL_ELEVATION_DEGREES = (15.0, 45.0)
FILL_FALLOFF_RANGE = (0.3, 0.9)               # a window's light fades across the frame
BACKGROUND_SLOPE_THRESHOLD = 0.06             # luminance change across the canvas that counts as a direction
BACKGROUND_AZIMUTH_JITTER = np.deg2rad(25)
WHITE_BALANCE_ADAPTATION_RANGE = (0.7, 1.0)   # 1 = the phone fully neutralizes the mean illuminant
WHITE_BALANCE_TINT_RANGE = (-0.025, 0.025)
EXPOSURE_RANGE = (0.9, 1.08)


def kelvin_to_linear_rgb(kelvin: float) -> np.ndarray:
    """Linear RGB of a black body at `kelvin`, scaled to unit luminance."""
    srgb = np.array([np.interp(kelvin, BLACK_BODY_KELVIN, BLACK_BODY_SRGB[:, channel]) for channel in range(3)])
    linear = srgb_to_linear(srgb)
    return linear / float(linear @ LUMINANCE_WEIGHTS)


def srgb_to_linear(image: np.ndarray) -> np.ndarray:
    """sRGB transfer curve, decoded."""
    image = np.clip(image, 0, 1)
    return np.where(image <= 0.04045, image / 12.92, ((image + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(image: np.ndarray) -> np.ndarray:
    """sRGB transfer curve, encoded."""
    image = np.clip(image, 0, 1)
    return np.where(image <= 0.0031308, image * 12.92, 1.055 * image ** (1 / 2.4) - 0.055)


def direction_from_angles(azimuth_radians: float, elevation_radians: float) -> np.ndarray:
    """Unit vector toward a light, (x, y, z) in the plane frame."""
    cos_elevation = np.cos(elevation_radians)
    return np.array([np.cos(azimuth_radians) * cos_elevation, np.sin(azimuth_radians) * cos_elevation,
                     np.sin(elevation_radians)])


@dataclass(frozen=True)
class FillLight:
    """A second source of another colour temperature (a window beside a lamp-lit room, or the reverse)."""

    azimuth_radians: float
    elevation_radians: float
    kelvin: float
    intensity: float          # relative to the key on flat sheet, where the fill is strongest
    falloff: float            # how much of its strength is lost across the frame, away from its side

    def direction(self) -> np.ndarray:
        """Unit vector toward the fill."""
        return direction_from_angles(self.azimuth_radians, self.elevation_radians)


@dataclass(frozen=True)
class SceneLight:
    """Key light, optional fill, ambient and the camera's white balance for one scene."""

    key_azimuth_radians: float
    key_elevation_radians: float
    key_kelvin: float
    key_angular_radius: float     # apparent radius of the source; sets every penumbra's width
    key_falloff: float
    ambient_kelvin: float
    ambient_fraction: float
    fill: FillLight | None
    white_balance_gains: tuple[float, float, float]   # linear gains the phone applied
    exposure: float
    azimuth_from_background: bool

    def key_direction(self) -> np.ndarray:
        """Unit vector toward the key."""
        return direction_from_angles(self.key_azimuth_radians, self.key_elevation_radians)

    def key_rgb(self) -> np.ndarray:
        """Key colour, linear, unit luminance."""
        return kelvin_to_linear_rgb(self.key_kelvin)

    def ambient_rgb(self) -> np.ndarray:
        """Ambient colour times its strength."""
        return kelvin_to_linear_rgb(self.ambient_kelvin) * self.ambient_fraction

    def fill_rgb(self) -> np.ndarray:
        """Fill colour times its peak strength (zeros when there is no fill)."""
        return np.zeros(3) if self.fill is None else kelvin_to_linear_rgb(self.fill.kelvin) * self.fill.intensity

    def to_dict(self) -> dict:
        """Plain-JSON record for the scene label's effects."""
        record = {"key_azimuth_degrees": round(float(np.degrees(self.key_azimuth_radians)), 1),
                  "key_elevation_degrees": round(float(np.degrees(self.key_elevation_radians)), 1),
                  "key_kelvin": int(self.key_kelvin), "key_angular_radius": round(self.key_angular_radius, 3),
                  "key_falloff": round(self.key_falloff, 3), "ambient_kelvin": int(self.ambient_kelvin),
                  "ambient_fraction": round(self.ambient_fraction, 3),
                  "white_balance_gains": [round(float(g), 3) for g in self.white_balance_gains],
                  "exposure": round(self.exposure, 3), "azimuth_from_background": self.azimuth_from_background}
        if self.fill is not None:
            record["fill"] = {"azimuth_degrees": round(float(np.degrees(self.fill.azimuth_radians)), 1),
                              "kelvin": int(self.fill.kelvin), "intensity": round(self.fill.intensity, 3),
                              "falloff": round(self.fill.falloff, 3)}
        return record


def background_brightness_azimuth(light_field: np.ndarray) -> float | None:
    """Direction the background looks lit from (toward its brighter side), or None if it is even.

    Fits a plane to the (mean-1) low-frequency luminance field on a coarse grid.
    """
    small = cv2.resize(light_field, (32, 24), interpolation=cv2.INTER_AREA).astype(np.float64)
    y_grid, x_grid = np.mgrid[0:small.shape[0], 0:small.shape[1]]
    x_norm, y_norm = x_grid / small.shape[1] - 0.5, y_grid / small.shape[0] - 0.5
    design = np.stack([np.ones(small.size), x_norm.ravel(), y_norm.ravel()], axis=1)
    _, slope_x, slope_y = np.linalg.lstsq(design, small.ravel(), rcond=None)[0]
    if np.hypot(slope_x, slope_y) < BACKGROUND_SLOPE_THRESHOLD:
        return None
    return float(np.arctan2(slope_y, slope_x))


def white_balance_for(illuminant_rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Gains an auto-white-balance would pick: mostly neutralize the mean illuminant, keep some warmth, add tint."""
    adaptation = rng.uniform(*WHITE_BALANCE_ADAPTATION_RANGE)
    gains = (illuminant_rgb / float(illuminant_rgb @ LUMINANCE_WEIGHTS)) ** -adaptation
    gains[1] *= 1 + rng.uniform(*WHITE_BALANCE_TINT_RANGE)
    return gains / float(gains @ LUMINANCE_WEIGHTS)


def sample_scene_light(light_field: np.ndarray, rng: np.random.Generator) -> SceneLight:
    """Draw the scene's light; `light_field` is the background's mean-1 low-frequency luminance."""
    background_azimuth = background_brightness_azimuth(light_field)
    if background_azimuth is not None:
        key_azimuth = background_azimuth + rng.uniform(-BACKGROUND_AZIMUTH_JITTER, BACKGROUND_AZIMUTH_JITTER)
    else:
        key_azimuth = rng.uniform(0, 2 * np.pi)
    kelvin_range = KEY_KELVIN_CHOICES[int(rng.choice(len(KEY_KELVIN_CHOICES), p=KEY_KELVIN_WEIGHTS))]
    key_kelvin = rng.uniform(*kelvin_range)
    fill = None
    if rng.random() < MIXED_LIGHT_PROBABILITY:
        offset = rng.uniform(*FILL_KELVIN_OFFSET_RANGE)
        fill_kelvin = key_kelvin + offset if key_kelvin < 4500 else key_kelvin - offset
        fill = FillLight(key_azimuth + rng.uniform(0.6, 1.0) * np.pi * rng.choice([-1, 1]),
                         np.deg2rad(rng.uniform(*FILL_ELEVATION_DEGREES)), float(np.clip(fill_kelvin, 2200, 9500)),
                         rng.uniform(*FILL_INTENSITY_RANGE), rng.uniform(*FILL_FALLOFF_RANGE))
    ambient_kelvin = (key_kelvin + (fill.kelvin if fill else key_kelvin)) / 2 + rng.uniform(-400, 400)
    ambient_fraction = rng.uniform(*AMBIENT_FRACTION_RANGE)
    key_rgb, ambient_rgb = kelvin_to_linear_rgb(key_kelvin), kelvin_to_linear_rgb(ambient_kelvin) * ambient_fraction
    mean_illuminant = key_rgb + ambient_rgb
    if fill is not None:
        mean_illuminant = mean_illuminant + kelvin_to_linear_rgb(fill.kelvin) * fill.intensity * (1 - fill.falloff / 2)
    return SceneLight(float(key_azimuth), float(np.deg2rad(rng.uniform(*KEY_ELEVATION_DEGREES))), float(key_kelvin),
                      rng.uniform(*KEY_ANGULAR_RADIUS_RANGE), rng.uniform(*KEY_FALLOFF_RANGE), float(ambient_kelvin),
                      float(ambient_fraction), fill, tuple(float(g) for g in white_balance_for(mean_illuminant, rng)),
                      rng.uniform(*EXPOSURE_RANGE), background_azimuth is not None)
