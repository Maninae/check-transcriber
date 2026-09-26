"""Tuning knobs for edge-based corner refinement (one dataclass, pixels or fractions).

Bands scale with the check's short side so a 400 px and a 1500 px check get the same
relative search. Pass 1 searches mostly INWARD: an oriented box (or any enclosing
detector box) contains the paper's corners, so the true side lies at or inside it; a
narrow outward band keeps neighbouring checks out of reach. Later passes are small and
symmetric around the previous pass's curves.
"""

from dataclasses import asdict, dataclass


@dataclass
class CornerRefinementConfig:
    """Parameters of `refine_check_quadrilateral`; defaults were tuned on val."""

    # Per pass: search band inward / outward of the current side, as a fraction of the
    # short side, clamped to [minimum_band_pixels, maximum_band_pixels].
    inward_band_fraction_per_pass: tuple[float, ...] = (0.14, 0.03)
    outward_band_fraction_per_pass: tuple[float, ...] = (0.03, 0.03)
    minimum_band_pixels: float = 6.0
    maximum_band_pixels: float = 140.0
    # Radon line search half-range (degrees) per pass, and its step.
    maximum_line_angle_degrees_per_pass: tuple[float, ...] = (6.0, 2.0)
    line_angle_step_degrees: float = 0.5
    # The outermost candidate line whose integrated score reaches this fraction of the
    # best wins (printed borders and rules lie inside the paper edge).
    outermost_line_ratio: float = 0.5
    # Per-sample scores are capped at this contrast (colour units) before integration.
    line_score_clip: float = 20.0
    # Curve degree per pass (1 straight, 2 captures curl), and grow iterations.
    curve_degree_per_pass: tuple[int, ...] = (1, 2)
    curve_grow_iterations: int = 3
    point_tolerance_pixels: float = 3.0

    # Samples along each side: one every `sample_spacing_pixels`, clamped to this range.
    sample_spacing_pixels: float = 6.0
    minimum_samples_per_side: int = 16
    maximum_samples_per_side: int = 128
    # Samples closer to a corner than this fraction of the side are skipped.
    corner_margin_fraction: float = 0.03
    minimum_corner_margin_pixels: float = 4.0

    # Profile filtering: tangential averaging and the inside / outside box windows.
    tangential_offsets_pixels: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0)
    inner_window_pixels: int = 4
    outer_window_pixels: int = 4
    # Paper colour = median over a grid spanning this central part of the input quad.
    paper_sample_inset_fraction: float = 0.15

    # Robust fit: RANSAC inlier distance, minimum support, minimum edge score.
    ransac_iterations: int = 48
    ransac_inlier_distance_pixels: float = 1.5
    minimum_inlier_fraction: float = 0.25
    minimum_inlier_count: int = 6
    minimum_edge_score: float = 4.0

    # Guard rails vs the INPUT quad: max corner move as a multiple of the pass-1 inward
    # band, and max relative change of any side's length.
    maximum_corner_move_band_multiple: float = 1.5
    maximum_side_length_change_fraction: float = 0.35

    random_seed: int = 0

    def to_json_dict(self) -> dict:
        """Plain dict for results files."""
        return asdict(self)
