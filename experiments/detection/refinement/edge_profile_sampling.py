"""Score every candidate edge offset along one side of an approximate quad.

For each sample position along the side we read a colour profile along the side's
outward normal with one `cv2.remap` call (bilinear, float32) and average a few profiles
offset along the tangent to suppress texture. Each normal offset is then scored as a
paper boundary: mean colour distance to the paper colour just OUTSIDE the offset minus
the same just INSIDE it (box windows, the centre pixel excluded). The score is high where
paper gives way to something else, low for a background stripe boundary (its inner side
is not paper-coloured) and reduced for thin printed lines (they fill only part of the
outer window).

- Offsets whose windows leave the image score -inf, so sides running off the frame
  produce no evidence and the caller keeps their approximate line.
- Only remap and elementwise math: portable to OpenCV.js.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SideScoreProfiles:
    """Edge scores for one side on a (sample position x normal offset) grid, plus its frame."""

    side_start: np.ndarray  # (2,) image pixels
    unit_tangent: np.ndarray  # (2,) start -> end
    unit_normal: np.ndarray  # (2,) pointing away from the quad centroid
    side_length: float
    positions_pixels: np.ndarray  # (S,) distance of each sample from side_start
    normal_offsets: np.ndarray  # (J,) integer offsets (px) along the outward normal
    scores: np.ndarray  # (S, J), -inf where the windows leave the image

    def points_at(self, positions_pixels: np.ndarray, offsets_pixels: np.ndarray) -> np.ndarray:
        """Image coordinates of (position along side, normal offset) pairs."""
        return (
            self.side_start[None, :]
            + positions_pixels[:, None] * self.unit_tangent[None, :]
            + offsets_pixels[:, None] * self.unit_normal[None, :]
        )


def compute_side_frame(side_start: np.ndarray, side_end: np.ndarray, quad_centroid: np.ndarray):
    """Return (unit tangent start->end, unit normal pointing away from the centroid, length)."""
    side_vector = side_end - side_start
    side_length = float(np.hypot(*side_vector))
    unit_tangent = side_vector / max(side_length, 1e-9)
    unit_normal = np.array([unit_tangent[1], -unit_tangent[0]])
    if np.dot(unit_normal, (side_start + side_end) / 2 - quad_centroid) < 0:
        unit_normal = -unit_normal
    return unit_tangent, unit_normal, side_length


def box_window_means(values: np.ndarray, centre_indices: np.ndarray, inner_window: int, outer_window: int):
    """Means of `values` over [c - inner, c) and (c, c + outer] for each centre c (axis 1)."""
    cumulative = np.concatenate([np.zeros_like(values[:, :1]), np.cumsum(values, axis=1)], axis=1)
    inner_means = (cumulative[:, centre_indices] - cumulative[:, centre_indices - inner_window]) / inner_window
    outer_means = (cumulative[:, centre_indices + outer_window + 1] - cumulative[:, centre_indices + 1]) / outer_window
    return inner_means, outer_means


def score_side_edge_profiles(
    image_crop_float: np.ndarray,
    crop_origin_xy: np.ndarray,
    image_width_height: tuple[int, int],
    side_start: np.ndarray,
    side_end: np.ndarray,
    quad_centroid: np.ndarray,
    inward_band_pixels: float,
    outward_band_pixels: float,
    number_of_samples: int,
    corner_margin_pixels: float,
    tangential_offsets_pixels: tuple[float, ...],
    inner_window_pixels: int,
    outer_window_pixels: int,
    paper_colour: np.ndarray,
) -> SideScoreProfiles:
    """Paper-boundary score for offsets in [-inward_band, +outward_band] at each sample.

    Args:
        image_crop_float: (H, W, C) float32 crop of the full-resolution image.
        crop_origin_xy: full-image (x, y) of the crop's pixel (0, 0).
        image_width_height: full image size, to reject profiles that leave the frame.
        corner_margin_pixels: samples start and end this far from the side's corners.
        paper_colour: (C,) median paper colour of the check.
    """
    unit_tangent, unit_normal, side_length = compute_side_frame(side_start, side_end, quad_centroid)
    margin = min(corner_margin_pixels, 0.3 * side_length)
    positions_pixels = np.linspace(margin, side_length - margin, number_of_samples)
    inward, outward = int(np.ceil(inward_band_pixels)), int(np.ceil(outward_band_pixels))
    sampled_offsets = np.arange(-inward - inner_window_pixels, outward + outer_window_pixels + 1, dtype=np.float64)
    tangential_offsets = np.asarray(tangential_offsets_pixels, dtype=np.float64)

    # Map grid: rows = sample x tangential offset, columns = normal offset.
    along = (positions_pixels[:, None] + tangential_offsets[None, :]).reshape(-1)
    base_points = side_start[None, :] + along[:, None] * unit_tangent[None, :]
    map_x = base_points[:, 0:1] + sampled_offsets[None, :] * unit_normal[0]
    map_y = base_points[:, 1:2] + sampled_offsets[None, :] * unit_normal[1]
    image_width, image_height = image_width_height
    inside_image = (map_x >= 0) & (map_x <= image_width - 1) & (map_y >= 0) & (map_y <= image_height - 1)
    profiles = cv2.remap(
        image_crop_float,
        (map_x - crop_origin_xy[0]).astype(np.float32),
        (map_y - crop_origin_xy[1]).astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    if profiles.ndim == 2:
        profiles = profiles[:, :, None]
    number_of_tangential = len(tangential_offsets)
    profiles = profiles.reshape(number_of_samples, number_of_tangential, len(sampled_offsets), -1).mean(axis=1)
    inside_image = inside_image.reshape(number_of_samples, number_of_tangential, -1).all(axis=1)

    distance_to_paper = np.linalg.norm(profiles - paper_colour, axis=2)
    centre_indices = np.arange(inner_window_pixels, len(sampled_offsets) - outer_window_pixels)
    inner_distance, outer_distance = box_window_means(distance_to_paper, centre_indices, inner_window_pixels, outer_window_pixels)
    scores = outer_distance - inner_distance
    window_valid = np.ones_like(scores, dtype=bool)
    for shift in range(-inner_window_pixels, outer_window_pixels + 1):
        window_valid &= inside_image[:, centre_indices + shift]
    return SideScoreProfiles(
        side_start=side_start.astype(np.float64),
        unit_tangent=unit_tangent,
        unit_normal=unit_normal,
        side_length=side_length,
        positions_pixels=positions_pixels,
        normal_offsets=sampled_offsets[centre_indices],
        scores=np.where(window_valid, scores, -np.inf),
    )
