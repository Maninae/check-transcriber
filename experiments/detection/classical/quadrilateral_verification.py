"""Geometry gates and an evidence score for a candidate quadrilateral.

The score asks one question per side: is there a real border here? Points are sampled
along the middle of each side, and a sample counts as "on a border" when ANY of three
signals fires (each is the natural border cue on some background):

- gradient: lightness gradient along the side normal, maximized over +-2 px
  (dark rugs, wood, the drop shadow along a check on a white sheet);
- color: Lab distance between a point just inside and one just outside
  (pastel check on a same-lightness colored fabric);
- texture: local std just outside minus just inside (white check on white crochet or
  beige carpet: same color, but the fabric is textured and the paper is not).

Samples falling outside the image count as supported (the check runs out of frame).

    score = 0.6 * mean side support + 0.4 * second-weakest side support

The second-weakest term forgives exactly one weak side (a hand shadow, glare, an
overlapping check on top) but rejects a quad that only two real borders support, the
signature of a strip of background between two checks.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from experiments.detection.classical.classical_detector_config import (
    ClassicalDetectorConfig,
    odd_kernel_size,
)
from experiments.detection.classical.edge_line_snapping import side_is_on_image_border
from experiments.detection.classical.quadrilateral_geometry import (
    is_convex_quadrilateral,
    quadrilateral_area,
    quadrilateral_aspect_ratio,
    quadrilateral_interior_angles_degrees,
)
from experiments.detection.classical.working_image_channels import WorkingImageChannels

SIDE_END_EXCLUSION_FRACTION = 0.08
GRADIENT_SEARCH_OFFSETS_PIXELS = (-2.0, -1.0, 0.0, 1.0, 2.0)
COLOR_SAMPLE_OFFSET_PIXELS = 4.0
MEAN_SUPPORT_WEIGHT = 0.6
SECOND_WEAKEST_SUPPORT_WEIGHT = 0.4
MAXIMUM_STRENGTH_RATIO = 3.0


@dataclass
class QuadrilateralEvidence:
    """Per-side border support and the combined verification score."""

    side_supports: np.ndarray  # (4,) fraction of samples on a border
    side_strengths: np.ndarray  # (4,) mean border-signal ratio to its threshold, clipped at 3
    score: float


def quadrilateral_touches_image_border(corners: np.ndarray, image_size: tuple[int, int]) -> bool:
    """True when any side lies along the image border (the check runs out of frame)."""
    image_width, image_height = image_size
    return any(
        side_is_on_image_border(corners[index], corners[(index + 1) % 4], image_width, image_height) for index in range(4)
    )


def passes_geometry_gates(corners: np.ndarray, image_size: tuple[int, int], config: ClassicalDetectorConfig) -> bool:
    """Convex, check-sized, check-shaped (aspect in range, no needle corners).

    `image_size` is (width, height) of the working image. Quads cut by the image border
    get the looser `border_truncated_aspect_range`, since one dimension is truncated.
    """
    if not is_convex_quadrilateral(corners):
        return False
    area_fraction = quadrilateral_area(corners) / float(image_size[0] * image_size[1])
    if not config.minimum_area_fraction <= area_fraction <= config.maximum_area_fraction:
        return False
    aspect_ratio = quadrilateral_aspect_ratio(corners)
    if quadrilateral_touches_image_border(corners, image_size):
        minimum_aspect, maximum_aspect = config.border_truncated_aspect_range
    else:
        minimum_aspect, maximum_aspect = config.minimum_aspect_ratio, config.maximum_aspect_ratio
    if not minimum_aspect <= aspect_ratio <= maximum_aspect:
        return False
    return bool(np.min(quadrilateral_interior_angles_degrees(corners)) >= config.minimum_interior_angle_degrees)


def sample_map_at_points(signal_map: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Bilinear samples of a float32 map at (N, 2) points (cv.remap, edge-replicated)."""
    map_x = points[:, 0].astype(np.float32).reshape(1, -1)
    map_y = points[:, 1].astype(np.float32).reshape(1, -1)
    return cv2.remap(signal_map, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).reshape(-1)


def measure_side_border_support(
    side_start: np.ndarray,
    side_end: np.ndarray,
    inward_normal: np.ndarray,
    channels: WorkingImageChannels,
    config: ClassicalDetectorConfig,
) -> tuple[float, float]:
    """(support, strength) of one side.

    support: fraction of samples where at least one border signal fires.
    strength: mean over samples of the strongest signal's ratio to its threshold, clipped at
    `MAXIMUM_STRENGTH_RATIO`; out-of-frame samples count as exactly 1 (threshold level), so a
    side resting on the image border never outranks a real, strong border.
    """
    image_height, image_width = channels.lightness.shape
    fractions = np.linspace(SIDE_END_EXCLUSION_FRACTION, 1.0 - SIDE_END_EXCLUSION_FRACTION, config.boundary_samples_per_side)
    sample_points = side_start + fractions[:, None] * (side_end - side_start)
    out_of_frame = (
        (sample_points[:, 0] < 0) | (sample_points[:, 0] > image_width - 1)
        | (sample_points[:, 1] < 0) | (sample_points[:, 1] > image_height - 1)
    )

    normal_gradient = np.zeros(len(sample_points))
    for offset in GRADIENT_SEARCH_OFFSETS_PIXELS:
        shifted_points = sample_points + offset * inward_normal
        gradient_x = sample_map_at_points(channels.gradient_x, shifted_points)
        gradient_y = sample_map_at_points(channels.gradient_y, shifted_points)
        normal_gradient = np.maximum(normal_gradient, np.abs(gradient_x * inward_normal[0] + gradient_y * inward_normal[1]))

    inner_points = sample_points + COLOR_SAMPLE_OFFSET_PIXELS * inward_normal
    outer_points = sample_points - COLOR_SAMPLE_OFFSET_PIXELS * inward_normal
    color_distance = np.sqrt(
        sum(
            (sample_map_at_points(channel_map, inner_points) - sample_map_at_points(channel_map, outer_points)) ** 2
            for channel_map in (channels.lightness, channels.lab_a, channels.lab_b)
        )
    )

    texture_offset = float(odd_kernel_size(config.texture_window_fraction, channels.long_side_pixels))
    texture_contrast = sample_map_at_points(
        channels.texture_std, sample_points - texture_offset * inward_normal
    ) - sample_map_at_points(channels.texture_std, sample_points + texture_offset * inward_normal)

    signal_ratio = np.maximum.reduce(
        [
            normal_gradient / config.edge_support_gradient_threshold,
            color_distance / config.edge_support_color_threshold,
            texture_contrast / config.edge_support_texture_threshold,
        ]
    )
    signal_ratio = np.where(out_of_frame, 1.0, np.minimum(signal_ratio, MAXIMUM_STRENGTH_RATIO))
    on_border = (signal_ratio > 1.0) | out_of_frame
    return float(on_border.mean()), float(signal_ratio.mean())


def measure_quadrilateral_evidence(
    corners: np.ndarray, channels: WorkingImageChannels, config: ClassicalDetectorConfig
) -> QuadrilateralEvidence:
    """Border support of all four sides and the combined score."""
    centroid = corners.mean(axis=0)
    side_supports, side_strengths = [], []
    for side_index in range(4):
        side_start, side_end = corners[side_index], corners[(side_index + 1) % 4]
        side_vector = side_end - side_start
        normal = np.array([-side_vector[1], side_vector[0]]) / max(float(np.linalg.norm(side_vector)), 1e-9)
        side_midpoint = (side_start + side_end) / 2.0
        inward_normal = normal if float((centroid - side_midpoint) @ normal) > 0 else -normal
        support, strength = measure_side_border_support(side_start, side_end, inward_normal, channels, config)
        side_supports.append(support)
        side_strengths.append(strength)
    side_supports = np.array(side_supports)
    second_weakest_support = float(np.sort(side_supports)[1])
    score = MEAN_SUPPORT_WEIGHT * float(side_supports.mean()) + SECOND_WEAKEST_SUPPORT_WEIGHT * second_weakest_support
    return QuadrilateralEvidence(side_supports, np.array(side_strengths), score)
