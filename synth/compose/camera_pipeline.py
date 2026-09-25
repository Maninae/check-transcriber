"""Phone camera pipeline over the finished photo, in the order a phone applies it.

Optics first (on linear light): lens vignetting, a broad specular sheen on paper, defocus or
hand-shake blur. Then the sensor: shot + read noise in linear light, so dark areas are
noisier once encoded. Then the ISP: local tone mapping (lifts shadows, tames highlights,
boosts local contrast), a global contrast curve, chroma denoise, luminance sharpening with
its halos, JPEG.

Every op is pure (float32 in, float32 out) and moves no pixel geometrically, so labels are
unaffected; blur spreads ink by a pixel or two, well inside label tolerance.
"""

import cv2
import numpy as np

from synth.compose.scene_light import LUMINANCE_WEIGHTS, linear_to_srgb, srgb_to_linear

TONE_MAP_DOWNSCALE = 8
TONE_MAP_BASE_SIGMA_FRACTION = 0.03    # of the photo's long side
LOG_EPSILON = 1e-3
SHEEN_ON_SHEET_SHARE = 0.25            # fabric reflects much less specular than paper
AUTO_EXPOSURE_PERCENTILE = 98
LUMA_DENOISE_SIGMA = 0.9
CHROMA_EDGE_THRESHOLD = 0.04           # luma gradient above which chroma is left alone (ink strokes keep their colour)


def auto_expose(linear: np.ndarray, highlight_target: float) -> tuple[np.ndarray, float]:
    """Scale so the 98th-percentile luminance lands on `highlight_target` (linear); returns (image, gain).

    Phones meter so white paper stays below clipping; this is what keeps scenes from looking washed out.
    """
    luminance = linear[::4, ::4] @ LUMINANCE_WEIGHTS.astype(np.float32)
    gain = highlight_target / max(1e-4, float(np.percentile(luminance, AUTO_EXPOSURE_PERCENTILE)))
    return linear * np.float32(gain), gain


def lens_vignette(linear: np.ndarray, strength: float) -> np.ndarray:
    """Radial cos^4-like falloff: 1 at the centre, 1 - strength at the corners."""
    height, width = linear.shape[:2]
    y_grid, x_grid = np.ogrid[0:height, 0:width]
    radius_squared = ((x_grid - width / 2) ** 2 + (y_grid - height / 2) ** 2) / ((width / 2) ** 2 + (height / 2) ** 2)
    return linear * (1 - strength * radius_squared.astype(np.float32))[..., None]


def paper_sheen(linear: np.ndarray, paper_mask: np.ndarray, center_xy: tuple[float, float], radius_px: float,
                intensity: float, light_rgb: np.ndarray) -> np.ndarray:
    """Broad soft specular lobe, strongest on paper; additive in the light's colour, so it desaturates."""
    height, width = linear.shape[:2]
    small_w, small_h = max(8, width // 16), max(8, height // 16)
    y_grid, x_grid = np.mgrid[0:small_h, 0:small_w].astype(np.float32)
    distance_squared = ((x_grid + 0.5) * width / small_w - center_xy[0]) ** 2 + ((y_grid + 0.5) * height / small_h - center_xy[1]) ** 2
    lobe = cv2.resize(np.exp(-distance_squared / (2 * radius_px**2)).astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    weight = SHEEN_ON_SHEET_SHARE + (1 - SHEEN_ON_SHEET_SHARE) * cv2.GaussianBlur(paper_mask.astype(np.float32), (0, 0), 3)
    return linear + (intensity * lobe * weight)[..., None] * light_rgb.astype(np.float32)


def defocus_blur(image: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Gaussian defocus that is stronger toward one side (the focal plane is tilted against the sheet)."""
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)
    height, width = image.shape[:2]
    angle = rng.uniform(0, 2 * np.pi)
    y_grid, x_grid = np.mgrid[0:8, 0:8].astype(np.float32) / 7 - 0.5
    ramp = np.clip(0.55 + (np.cos(angle) * x_grid + np.sin(angle) * y_grid) * rng.uniform(0.4, 0.9), 0.15, 1.0)
    weight = cv2.resize(ramp, (width, height), interpolation=cv2.INTER_LINEAR)[..., None]
    return image * (1 - weight) + blurred * weight


def motion_blur(image: np.ndarray, half_length_px: int, angle_degrees: float) -> np.ndarray:
    """Linear hand-shake blur over 2 * half_length_px + 1 pixels, centred, so nothing shifts (labels stay exact).

    The kernel side is odd and the line is rotated about the centre pixel (k, k), which is
    exactly where filter2D anchors it; an even side would shift the image by half a pixel or more.
    """
    size = 2 * half_length_px + 1
    kernel = np.zeros((size, size), np.float32)
    kernel[half_length_px, :] = 1.0
    rotation = cv2.getRotationMatrix2D((float(half_length_px), float(half_length_px)), angle_degrees, 1.0)
    kernel = cv2.warpAffine(kernel, rotation, (size, size), flags=cv2.INTER_LINEAR)
    return cv2.filter2D(image, -1, kernel / max(1e-6, float(kernel.sum())), anchor=(half_length_px, half_length_px),
                        borderType=cv2.BORDER_REFLECT)


def sensor_noise(linear: np.ndarray, shot_scale: float, read_sigma: float, chroma_sigma: float,
                 rng: np.random.Generator) -> np.ndarray:
    """Photon shot noise (variance ~ signal) plus read noise as one luminance grain, and blotchy low-frequency chroma noise.

    Per-channel fine noise is left out on purpose: the phone's chroma denoise removes it anyway,
    and one shared channel costs a third of the random draws.
    """
    linear = np.clip(linear, 0, None)
    luminance = linear @ LUMINANCE_WEIGHTS.astype(np.float32)
    grain = rng.standard_normal(luminance.shape, dtype=np.float32) * np.sqrt(luminance * shot_scale + read_sigma**2)
    height, width = linear.shape[:2]
    chroma = rng.standard_normal((height // 3 + 1, width // 3 + 1, 3), dtype=np.float32) * chroma_sigma
    chroma = cv2.resize(chroma - chroma.mean(axis=2, keepdims=True), (width, height), interpolation=cv2.INTER_LINEAR)
    return linear + grain[..., None] + chroma * np.sqrt(luminance + read_sigma)[..., None]


def local_tone_map(linear: np.ndarray, base_compression: float, detail_boost: float) -> np.ndarray:
    """Base/detail split of log luminance: compress the base (large-scale light), boost the detail."""
    height, width = linear.shape[:2]
    luminance = np.clip(linear @ LUMINANCE_WEIGHTS.astype(np.float32), 0, None)
    log_luminance = np.log(luminance + LOG_EPSILON)
    small = cv2.resize(log_luminance, (max(8, width // TONE_MAP_DOWNSCALE), max(8, height // TONE_MAP_DOWNSCALE)),
                       interpolation=cv2.INTER_AREA)
    sigma = TONE_MAP_BASE_SIGMA_FRACTION * max(small.shape)
    base = cv2.resize(cv2.GaussianBlur(small, (0, 0), sigmaX=sigma), (width, height), interpolation=cv2.INTER_LINEAR)
    anchor = float(np.percentile(small, 60))
    new_log = anchor + base_compression * (base - anchor) + detail_boost * (log_luminance - base)
    ratio = np.exp(new_log - log_luminance)
    return linear * ratio[..., None]


def contrast_curve(srgb: np.ndarray, strength: float) -> np.ndarray:
    """Blend toward smoothstep: deeper blacks, cleaner whites, same mid-grey."""
    srgb = np.clip(srgb, 0, 1)
    return srgb + strength * (srgb * srgb * (3 - 2 * srgb) - srgb)


def flat_region_weight(luma: np.ndarray) -> np.ndarray:
    """1 where luminance is locally flat, 0 at edges (ink, paper borders), from a dilated gradient."""
    smooth_luma = cv2.GaussianBlur(luma, (0, 0), sigmaX=0.8)
    gradient = np.abs(cv2.Sobel(smooth_luma, cv2.CV_32F, 1, 0, ksize=3)) + np.abs(cv2.Sobel(smooth_luma, cv2.CV_32F, 0, 1, ksize=3))
    return np.clip(1 - cv2.dilate(gradient, np.ones((3, 3), np.uint8)) / (8 * CHROMA_EDGE_THRESHOLD), 0, 1)


def denoise_and_sharpen(srgb: np.ndarray, chroma_sigma: float, luma_denoise: float, sharpen_sigma: float,
                        sharpen_amount: float) -> np.ndarray:
    """Phone ISP finish: edge-aware chroma and luma smoothing, then an unsharp mask on luminance (with its light halos).

    Smoothing is guided by luma edges, as phone noise reduction is: colour blotches and grain on
    paper and fabric fade, thin ink strokes keep their hue and edges.
    """
    ycrcb = cv2.cvtColor(np.clip(srgb, 0, 1).astype(np.float32), cv2.COLOR_RGB2YCrCb)
    flatness = flat_region_weight(ycrcb[..., 0])
    for channel in (1, 2):
        smoothed = cv2.GaussianBlur(ycrcb[..., channel], (0, 0), sigmaX=chroma_sigma)
        ycrcb[..., channel] = smoothed * flatness + ycrcb[..., channel] * (1 - flatness)
    luma_weight = luma_denoise * flatness
    luma = cv2.GaussianBlur(ycrcb[..., 0], (0, 0), sigmaX=LUMA_DENOISE_SIGMA) * luma_weight + ycrcb[..., 0] * (1 - luma_weight)
    if sharpen_amount > 0:
        luma = luma + sharpen_amount * (luma - cv2.GaussianBlur(luma, (0, 0), sigmaX=sharpen_sigma))
    ycrcb[..., 0] = luma
    return np.clip(cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB), 0, 1)


def encode_srgb(linear: np.ndarray) -> np.ndarray:
    """Linear -> sRGB float32."""
    return linear_to_srgb(linear).astype(np.float32)


def decode_srgb(srgb: np.ndarray) -> np.ndarray:
    """sRGB -> linear float32."""
    return srgb_to_linear(srgb).astype(np.float32)


def jpeg_roundtrip(image: np.ndarray, quality: int) -> np.ndarray:
    """Encode and decode as JPEG at `quality`; returns uint8 RGB."""
    image_uint8 = (np.clip(image, 0, 1) * 255 + 0.5).astype(np.uint8)
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
