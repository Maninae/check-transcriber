"""Geometry for the scene: where the camera looks, and how points move through every warp.

Coordinate frames, in the order a pixel travels:
1. check frame: pixels of the flat rendered check.
2. sheet plane: a top-down canvas of the bedsheet; each check lands here by its CheckPlaneMap
   (a rigid placement plus paper deformation, check_plane_map.py).
3. photo frame: the plane seen by a tilted camera (homography), then mild lens distortion.

Every image warp here has a matching point transform, so labels follow pixels exactly.
"""

from dataclasses import dataclass

import cv2
import numpy as np

MAX_TILT_FRACTION = 0.32
CORNER_JITTER_FRACTION = 0.03
MAX_CAMERA_ROLL_DEGREES = 4.0
CANVAS_MARGIN_FRACTION = 0.08
DISTORTION_ITERATIONS = 8


@dataclass
class CameraView:
    """The camera's view of the sheet plane."""

    canvas_width: int
    canvas_height: int
    visible_quad_plane: np.ndarray  # (4, 2) plane points that map to the photo's corners
    plane_to_photo: np.ndarray      # (3, 3) homography
    photo_width: int
    photo_height: int
    radial_k1: float


def apply_homography(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """Map (N, 2) points through a 3x3 homography."""
    homogeneous = np.hstack([points, np.ones((len(points), 1))]) @ homography.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def apply_affine(points: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Map (N, 2) points through a 2x3 affine matrix."""
    return np.hstack([points, np.ones((len(points), 1))]) @ affine.T


def rotate_points(points: np.ndarray, center: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate points about `center` (counter-clockwise on screen for positive degrees)."""
    radians = np.deg2rad(degrees)
    rotation = np.array([[np.cos(radians), np.sin(radians)], [-np.sin(radians), np.cos(radians)]])
    return (points - center) @ rotation.T + center


def sample_camera_view(photo_width: int, photo_height: int, rng: np.random.Generator,
                       tilt_range: tuple[float, float] = (0.0, MAX_TILT_FRACTION)) -> CameraView:
    """Pick a camera tilt: the far edge of the sheet looks narrower, so its plane quad is wider.

    The tilted edge is random (top most often). Corners are jittered and the quad is rolled
    slightly, as a hand-held phone would be. `tilt_range` is the far edge's plane-quad widening as a
    fraction of the photo side; 0.8 makes the far edge ~0.55x the near one (a steep, close shot).
    """
    rect = np.array([[0, 0], [photo_width, 0], [photo_width, photo_height], [0, photo_height]], np.float64)
    tilt = rng.uniform(*tilt_range)
    far_edge = rng.choice(["top", "top", "top", "bottom", "left", "right"])
    quad = rect.copy()
    widen_x, widen_y = tilt * photo_width / 2, tilt * photo_height / 2
    if far_edge == "top":
        quad[0, 0] -= widen_x; quad[1, 0] += widen_x
    elif far_edge == "bottom":
        quad[3, 0] -= widen_x; quad[2, 0] += widen_x
    elif far_edge == "left":
        quad[0, 1] -= widen_y; quad[3, 1] += widen_y
    else:
        quad[1, 1] -= widen_y; quad[2, 1] += widen_y
    quad += rng.uniform(-CORNER_JITTER_FRACTION, CORNER_JITTER_FRACTION, quad.shape) * [photo_width, photo_height]
    quad = rotate_points(quad, quad.mean(axis=0), rng.uniform(-MAX_CAMERA_ROLL_DEGREES, MAX_CAMERA_ROLL_DEGREES))

    margin = CANVAS_MARGIN_FRACTION * max(photo_width, photo_height)
    quad -= quad.min(axis=0) - margin
    canvas_width, canvas_height = (np.ceil(quad.max(axis=0) + margin)).astype(int)
    plane_to_photo = cv2.getPerspectiveTransform(quad.astype(np.float32), rect.astype(np.float32)).astype(np.float64)
    return CameraView(int(canvas_width), int(canvas_height), quad, plane_to_photo, photo_width, photo_height,
                      radial_k1=float(rng.uniform(-0.035, 0.015)))


def normalized_radius_squared(points: np.ndarray, width: int, height: int) -> np.ndarray:
    """r^2 about the image center, with the half-diagonal as unit length."""
    center = np.array([width / 2, height / 2])
    half_diagonal = np.hypot(width / 2, height / 2)
    return (((points - center) / half_diagonal) ** 2).sum(axis=1)


def distort_points(points: np.ndarray, width: int, height: int, k1: float) -> np.ndarray:
    """Where an undistorted photo point appears after radial lens distortion.

    The image remap samples undistorted(p) = center + (q - center)(1 + k1 r(q)^2) for each
    output pixel q, so the point transform inverts that by fixed-point iteration.
    """
    if k1 == 0:
        return points.copy()
    center = np.array([width / 2, height / 2])
    distorted = points.astype(np.float64).copy()
    for _ in range(DISTORTION_ITERATIONS):
        scale = 1 + k1 * normalized_radius_squared(distorted, width, height)
        distorted = center + (points - center) / scale[:, None]
    return distorted


def radial_distortion_maps(width: int, height: int, k1: float) -> tuple[np.ndarray, np.ndarray]:
    """cv2.remap maps that apply radial distortion of strength k1 to a whole image."""
    y_grid, x_grid = np.mgrid[0:height, 0:width].astype(np.float32)
    center_x, center_y = width / 2, height / 2
    half_diagonal = np.hypot(center_x, center_y)
    r_squared = ((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2) / half_diagonal**2
    scale = 1 + k1 * r_squared
    return (center_x + (x_grid - center_x) * scale).astype(np.float32), (center_y + (y_grid - center_y) * scale).astype(np.float32)


def plane_points_to_photo(points: np.ndarray, view: CameraView) -> np.ndarray:
    """Full plane -> photo transform for labels: homography, then lens distortion."""
    return distort_points(apply_homography(points, view.plane_to_photo), view.photo_width, view.photo_height, view.radial_k1)


def clip_polygon_to_rect(polygon: np.ndarray, width: float, height: float) -> np.ndarray:
    """Sutherland-Hodgman clip of a polygon to [0, width] x [0, height]; may return an empty array."""
    def clip_against(points, inside, intersect):
        output = []
        for index in range(len(points)):
            current, previous = points[index], points[index - 1]
            if inside(current):
                if not inside(previous):
                    output.append(intersect(previous, current))
                output.append(current)
            elif inside(previous):
                output.append(intersect(previous, current))
        return output

    def crossing(a, b, axis, value):
        t = (value - a[axis]) / (b[axis] - a[axis])
        return a + t * (b - a)

    points = [np.asarray(p, np.float64) for p in polygon]
    for axis, value, keep_greater in ((0, 0.0, True), (0, width, False), (1, 0.0, True), (1, height, False)):
        if not points:
            break
        points = clip_against(points,
                              (lambda p, a=axis, v=value, g=keep_greater: p[a] >= v if g else p[a] <= v),
                              (lambda p, q, a=axis, v=value: crossing(p, q, a, v)))
    return np.array(points).reshape(-1, 2)


def polygon_area(polygon: np.ndarray) -> float:
    """Shoelace area of a simple polygon."""
    if len(polygon) < 3:
        return 0.0
    x, y = polygon[:, 0], polygon[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)
