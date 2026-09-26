"""Score every candidate edge offset along one side of an approximate quad.

For each sample position along the side we read a colour profile along the side's
outward normal with one `cv2.remap` call (bilinear, straight from the uint8 image) and average a few profiles
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


def compute_two_class_paperness(
    profiles: np.ndarray, paper_colour: np.ndarray, background_window_pixels: int, minimum_contrast: float, tent_beyond_paper: bool = True
) -> np.ndarray:
    """Per-sample paperness in [0, 1]: 0 = this sample's background colour, 1 = paper.

    The background colour B of each sample is the mean of the outermost
    `background_window_pixels` of its profile (outside the approximate side). Each pixel is
    projected onto the paper-minus-B direction and clipped, so a faint but real hue
    difference still spans the full 0-1 range while dark ink clips to 0 like background.
    With `tent_beyond_paper`, colours beyond the paper along that axis fall off again
    (1 - |t - 1|), so paperness peaks AT the paper colour.
    Samples whose B is within `minimum_contrast` of the paper colour get NaN (no evidence).
    """
    background_colours = profiles[:, -background_window_pixels:, :].mean(axis=1)  # (S, C)
    paper_minus_background = paper_colour[None, :] - background_colours
    contrast_squared = (paper_minus_background**2).sum(axis=1)
    paperness = ((profiles - background_colours[:, None, :]) * paper_minus_background[:, None, :]).sum(axis=2) / np.maximum(contrast_squared, 1e-9)[:, None]
    if tent_beyond_paper:
        # Past the paper colour (away from the background) is NOT more paper: a drop shadow
        # beside a check darker than its sheet, or ink, would otherwise count as paper.
        paperness = np.where(paperness > 1.0, 2.0 - paperness, paperness)
    paperness = np.clip(paperness, 0.0, 1.0)
    paperness[contrast_squared < minimum_contrast**2] = np.nan
    return paperness


def compute_contact_line_scores(profiles: np.ndarray, centre_indices: np.ndarray, half_width: int, contrast_unit: float) -> np.ndarray:
    """Valley depth in luminance at each centre: mean of the pixels `half_width` either side minus the centre.

    White paper on a white sheet has almost no colour step, but the paper casts a thin
    dark contact shadow exactly along its edge. Depth is divided by `contrast_unit` and
    clipped to [0, 1] so it lives on the same scale as the two-class paperness step.
    """
    luminance = profiles.mean(axis=2)
    last_index = luminance.shape[1] - 1
    left = luminance[:, np.clip(centre_indices - half_width, 0, last_index)]
    right = luminance[:, np.clip(centre_indices + half_width, 0, last_index)]
    depth = (left + right) / 2 - luminance[:, centre_indices]
    return np.clip(depth / contrast_unit, 0.0, 1.0)


def score_side_edge_profiles(
    image: np.ndarray,
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
    score_mode: str = "paper_distance",
    background_window_pixels: int = 6,
    minimum_paper_background_contrast: float = 6.0,
    tent_beyond_paper: bool = True,
    contact_line_half_width_pixels: int = 0,
    contact_line_contrast_unit: float = 20.0,
    contact_line_below_contrast: float = 0.0,
) -> SideScoreProfiles:
    """Paper-boundary score for offsets in [-inward_band, +outward_band] at each sample.

    Args:
        image: full-resolution (H, W) or (H, W, C) uint8 image; bilinear remap of uint8
            rounds to whole grey levels, negligible after the window averaging.
        corner_margin_pixels: samples start and end this far from the side's corners.
        paper_colour: (C,) median paper colour of the check.
        score_mode: "paper_distance" (rise in distance to paper colour) or "two_class"
            (step in a per-sample paper-vs-background feature, see
            `compute_two_class_paperness`).
        contact_line_half_width_pixels: when > 0 (two_class mode), samples whose
            background colour is too close to the paper for a colour step instead score the
            thin dark contact shadow a sheet of paper casts along its own edge (see
            `compute_contact_line_scores`). Samples whose paper/background contrast is
            below `contact_line_below_contrast` switch to it too: a soft contact-shadow
            halo brightens back THROUGH the paper colour and fakes an outer colour edge.
    """
    unit_tangent, unit_normal, side_length = compute_side_frame(side_start, side_end, quad_centroid)
    margin = min(corner_margin_pixels, 0.3 * side_length)
    positions_pixels = np.linspace(margin, side_length - margin, number_of_samples)
    inward, outward = int(np.ceil(inward_band_pixels)), int(np.ceil(outward_band_pixels))
    outside_reach = outer_window_pixels
    sampled_offsets = np.arange(-inward - inner_window_pixels, outward + outside_reach + 1, dtype=np.float64)
    tangential_offsets = np.asarray(tangential_offsets_pixels, dtype=np.float64)

    # Map grid: rows = sample x tangential offset, columns = normal offset.
    along = (positions_pixels[:, None] + tangential_offsets[None, :]).reshape(-1)
    base_points = side_start[None, :] + along[:, None] * unit_tangent[None, :]
    map_x = base_points[:, 0:1] + sampled_offsets[None, :] * unit_normal[0]
    map_y = base_points[:, 1:2] + sampled_offsets[None, :] * unit_normal[1]
    image_height, image_width = image.shape[:2]
    inside_image = (map_x >= 0) & (map_x <= image_width - 1) & (map_y >= 0) & (map_y <= image_height - 1)
    profiles = cv2.remap(
        image, map_x.astype(np.float32), map_y.astype(np.float32), interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    ).astype(np.float32)
    if profiles.ndim == 2:
        profiles = profiles[:, :, None]
    number_of_tangential = len(tangential_offsets)
    profiles = profiles.reshape(number_of_samples, number_of_tangential, len(sampled_offsets), -1).mean(axis=1)
    inside_image = inside_image.reshape(number_of_samples, number_of_tangential, -1).all(axis=1)

    if score_mode == "paper_distance":
        edge_feature = np.linalg.norm(profiles - paper_colour, axis=2)  # rises past the edge
    elif score_mode == "two_class":
        edge_feature = -compute_two_class_paperness(profiles, paper_colour, background_window_pixels, minimum_paper_background_contrast, tent_beyond_paper)
    else:
        raise ValueError(f"unknown score_mode {score_mode!r}")
    centre_indices = np.arange(inner_window_pixels, len(sampled_offsets) - outside_reach)
    inner_distance, outer_distance = box_window_means(edge_feature, centre_indices, inner_window_pixels, outer_window_pixels)
    scores = outer_distance - inner_distance
    window_valid = np.ones_like(scores, dtype=bool)
    for shift in range(-inner_window_pixels, outside_reach + 1):
        window_valid &= inside_image[:, centre_indices + shift]
    if score_mode == "two_class" and contact_line_half_width_pixels > 0:
        background_colours = profiles[:, -background_window_pixels:, :].mean(axis=1)
        low_contrast = np.linalg.norm(background_colours - paper_colour[None, :], axis=1) < contact_line_below_contrast
        no_colour_evidence = ~np.isfinite(scores).any(axis=1) | low_contrast
        if no_colour_evidence.any():
            contact_scores = compute_contact_line_scores(profiles[no_colour_evidence], centre_indices, contact_line_half_width_pixels, contact_line_contrast_unit)
            scores[no_colour_evidence] = contact_scores
    window_valid &= np.isfinite(scores)
    return SideScoreProfiles(
        side_start=side_start.astype(np.float64),
        unit_tangent=unit_tangent,
        unit_normal=unit_normal,
        side_length=side_length,
        positions_pixels=positions_pixels,
        normal_offsets=sampled_offsets[centre_indices],
        scores=np.where(window_valid, scores, -np.inf),
    )
