"""Classical photometric effects that make a flat composite read as a phone photo.

All functions take and return float32 RGB images in [0, 1] (except the JPEG round-trip,
which returns uint8). None of them moves pixels geometrically, so labels are unaffected,
with one exception: blur spreads ink by a few pixels, well inside label tolerance.

Two groups:
- scene-matching: derive the background's low-frequency light field and color cast and
  apply them to the pasted checks, so paper and sheet share one light.
- camera: illumination falloff, glare, white balance, blur, sharpening, noise, JPEG.
"""

import cv2
import numpy as np

LIGHT_FIELD_DOWNSAMPLE = 32


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
    """Per-channel gains that push white paper toward the background's color cast."""
    mean_color = background[::8, ::8].reshape(-1, 3).mean(axis=0)
    gains = mean_color / max(1e-4, float(mean_color.mean()))
    return (1 + (gains - 1) * strength).astype(np.float32)


def curl_shading(check: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Brighten or darken one edge band, as when a check's end lifts off the sheet."""
    height, width = check.shape[:2]
    axis_is_x = rng.random() < 0.7
    length = width if axis_is_x else height
    band = rng.uniform(0.15, 0.4) * length
    ramp = np.clip(1 - np.arange(length, dtype=np.float32) / band, 0, 1) ** 2
    if rng.random() < 0.5:
        ramp = ramp[::-1]
    shading = 1 + rng.choice([-1, 1]) * rng.uniform(0.04, 0.14) * ramp
    shading = shading[None, :, None] if axis_is_x else shading[:, None, None]
    return np.clip(check * shading, 0, 1)


def illumination_gradient(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Multiply by a smooth falloff: a linear ramp plus a broad bright blob, plus vignetting.

    The field is smooth, so it is computed on a small grid and upsampled.
    """
    height, width = image.shape[:2]
    small_height, small_width = max(8, height // 16), max(8, width // 16)
    y_grid, x_grid = np.mgrid[0:small_height, 0:small_width].astype(np.float32)
    x_norm, y_norm = x_grid / small_width - 0.5, y_grid / small_height - 0.5
    angle = rng.uniform(0, 2 * np.pi)
    ramp = 1 + rng.uniform(0.05, 0.3) * (np.cos(angle) * x_norm + np.sin(angle) * y_norm)
    blob_x, blob_y = rng.uniform(-0.4, 0.4, 2)
    blob = rng.uniform(0.0, 0.18) * np.exp(-((x_norm - blob_x) ** 2 + (y_norm - blob_y) ** 2) / (2 * rng.uniform(0.15, 0.35) ** 2))
    vignette = 1 - rng.uniform(0.05, 0.25) * (x_norm**2 + y_norm**2) * 2
    field = cv2.resize((ramp + blob) * vignette, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.clip(image * field[..., None], 0, 1)


def glare_spot(image: np.ndarray, center_xy: tuple[float, float], rng: np.random.Generator) -> np.ndarray:
    """Add a soft specular highlight: additive, desaturating, elliptical (computed in a local window)."""
    height, width = image.shape[:2]
    radius_x = rng.uniform(0.03, 0.09) * width
    radius_y = radius_x * rng.uniform(0.5, 1.2)
    intensity = rng.uniform(0.2, 0.6)
    x0, x1 = int(max(0, center_xy[0] - 3 * radius_x)), int(min(width, center_xy[0] + 3 * radius_x))
    y0, y1 = int(max(0, center_xy[1] - 3 * radius_y)), int(min(height, center_xy[1] + 3 * radius_y))
    if x1 <= x0 or y1 <= y0:
        return image
    y_grid, x_grid = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    falloff = np.exp(-(((x_grid - center_xy[0]) / radius_x) ** 2 + ((y_grid - center_xy[1]) / radius_y) ** 2))
    window = image[y0:y1, x0:x1]
    image[y0:y1, x0:x1] = np.clip(window + intensity * falloff[..., None] * (1.02 - window), 0, 1)
    return image


def white_balance_jitter(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Warm/cool color-temperature shift plus a small green/magenta tint and exposure change."""
    temperature = rng.uniform(-0.12, 0.12)
    tint = rng.uniform(-0.04, 0.04)
    gains = np.array([1 + temperature, 1 + tint, 1 - temperature], np.float32) * rng.uniform(0.9, 1.08)
    return np.clip(image * gains, 0, 1)


def lens_blur(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Mild gaussian defocus."""
    sigma = rng.uniform(0.3, 1.3)
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)


def motion_blur(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Short linear hand-shake blur."""
    length = int(rng.integers(3, 9))
    kernel = np.zeros((length, length), np.float32)
    kernel[length // 2, :] = 1.0 / length
    rotation = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), rng.uniform(0, 180), 1.0)
    kernel = cv2.warpAffine(kernel, rotation, (length, length))
    return cv2.filter2D(image, -1, kernel / max(1e-6, kernel.sum()))


def phone_sharpening(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Unsharp mask, as phone pipelines apply; produces the familiar light halos on edges."""
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=rng.uniform(1.0, 2.0))
    return np.clip(image + rng.uniform(0.2, 0.7) * (image - blurred), 0, 1)


def sensor_noise(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Signal-dependent shot noise, read noise, and slightly blotchy chroma noise."""
    shot_scale = rng.uniform(0.0005, 0.005)
    read_sigma = rng.uniform(0.002, 0.007)
    noise = rng.standard_normal(image.shape, dtype=np.float32) * np.sqrt(image * shot_scale + read_sigma**2)
    height, width = image.shape[:2]
    chroma_small = rng.standard_normal((height // 2, width // 2, 3), dtype=np.float32) * rng.uniform(0.0, 0.008)
    chroma = cv2.resize(chroma_small, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.clip(image + noise + chroma, 0, 1)


def jpeg_roundtrip(image: np.ndarray, quality: int) -> np.ndarray:
    """Encode and decode as JPEG at `quality`; returns uint8 RGB."""
    image_uint8 = (np.clip(image, 0, 1) * 255 + 0.5).astype(np.uint8)
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
