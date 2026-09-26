"""Every threshold of the classical (no model) check detector, in one dataclass.

Sizes that depend on image scale are stored as fractions of the working image's long
side (or of its area), so one config works for 2 MP and 12 MP photos alike. Pixel
kernel sizes are derived from those fractions at runtime by `odd_kernel_size`.
"""

from dataclasses import asdict, dataclass, field


@dataclass
class ClassicalDetectorConfig:
    """Thresholds for preprocessing, candidate generation, verification and refinement."""

    # Working copy: the long side is resized to this many pixels before detection.
    working_long_side_pixels: int = 1600

    # Channels. Text strokes are removed from lightness by a grayscale closing of this size.
    text_suppression_kernel_fraction: float = 0.0045
    texture_window_fraction: float = 0.009  # local std window for the smoothness map
    chroma_weight_in_paper_score: float = 1.0  # paper score = lightness - w * chroma

    # Smooth-paper masks: pixels whose local lightness std is below each threshold.
    smooth_texture_std_thresholds: tuple[float, ...] = (3.0, 5.0)
    # Edge-bounded cells: gradient magnitude thresholds (on text-suppressed lightness).
    edge_gradient_thresholds: tuple[float, ...] = (12.0, 24.0)
    edge_dilation_pixels: int = 1
    # Canny edge-bounded cells: (low, high) hysteresis pairs on text-suppressed lightness.
    canny_threshold_pairs: tuple[tuple[float, float], ...] = ((20.0, 50.0), (40.0, 100.0))
    canny_dilation_pixels: int = 2
    use_chroma_edges: bool = True  # OR Canny of amplified Lab a/b into every Canny edge map
    chroma_edge_gain: float = 4.0
    use_hole_filled_edges: bool = True  # also offer each Canny map with enclosed holes filled
    # Textured masks (for plain white sheets, where the check's security print is the texture).
    textured_std_thresholds: tuple[float, ...] = (3.0,)
    # Unions of adjacent edge-bounded cells (a check cut in two by a shadow edge or a fold).
    merge_adjacent_cells: bool = True
    minimum_cell_piece_fraction: float = 0.0015  # of the image area, smaller pieces are ignored
    minimum_shared_boundary_fraction: float = 0.03  # of the long side, in edge-band pixels
    # Otsu threshold on the paper score is always tried; this offsets it (in score units).
    paper_score_otsu_offsets: tuple[float, ...] = (0.0,)
    mask_opening_fraction: float = 0.006  # cuts thin bridges between touching blobs

    # Line-based hypotheses (Hough segments -> parallel pairs + end caps -> rectangles).
    use_line_hypotheses: bool = True
    line_canny_low: float = 15.0
    line_canny_high: float = 40.0
    line_texture_suppression_std: float = 6.0  # edges whose smoothest neighbourhood is rougher are dropped
    line_minimum_length_fraction: float = 0.04  # of the working long side
    line_maximum_gap_fraction: float = 0.006  # HoughLinesP gap bridging
    line_merge_gap_fraction: float = 0.03  # collinear pieces closer than this are fused
    line_maximum_segments: int = 120
    line_maximum_pairs: int = 1500
    line_maximum_hypotheses: int = 400
    line_parallel_tolerance_degrees: float = 7.0
    line_perpendicular_tolerance_degrees: float = 10.0
    line_minimum_separation_fraction: float = 0.03
    line_hypothesis_rectangularity: float = 0.9  # stand-in rank value (lines have no region)

    # Geometry gates on a candidate quadrilateral (working-image pixels / fractions).
    minimum_area_fraction: float = 0.004
    maximum_area_fraction: float = 0.8
    minimum_aspect_ratio: float = 1.5
    maximum_aspect_ratio: float = 3.4
    # A quad with a side on the image border is truncated, so its aspect only needs this range.
    border_truncated_aspect_range: tuple[float, float] = (1.0, 8.0)
    minimum_interior_angle_degrees: float = 55.0
    minimum_region_rectangularity: float = 0.80  # region area / fitted quad area
    approx_poly_epsilon_fraction: float = 0.02  # of the contour perimeter

    # Working-resolution snap of each fitted side to the nearest lightness step.
    working_snap_search_fraction: float = 0.006  # of the working long side, each way
    working_snap_minimum_step: float = 2.0

    # Verification: sample the quad boundary; a sample is on an edge when ANY signal fires.
    boundary_samples_per_side: int = 40
    edge_support_gradient_threshold: float = 10.0  # normal gradient (Sobel / 4 units)
    edge_support_color_threshold: float = 8.0  # Lab distance, inner vs outer band
    edge_support_texture_threshold: float = 4.0  # outer minus inner local std
    edge_support_seam_residue_threshold: float = 30.0  # thin dark line on the side (print residue)
    minimum_edge_support: float = 0.45
    minimum_verification_score: float = 0.65  # tuned (sweep_v1)

    # Interior appearance gate (a check has print on smooth, bright, near-neutral paper).
    print_residue_threshold: float = 20.0
    minimum_interior_print_fraction: float = 0.05
    maximum_interior_texture: float = 4.5
    minimum_interior_paper_score: float = 130.0
    maximum_interior_chroma: float = 35.0

    # Interior seam gate: a straight strong border across the interior means a spanning quad.
    seam_gradient_threshold: float = 16.0
    maximum_interior_seam_strength: float = 0.55  # tuned (sweep_v1)
    minimum_line_hypothesis_score: float = 0.9  # line-only quads need stronger borders, tuned (sweep_v1)

    # Selection rank = score + 0.25 * rectangularity + this * weakest side's border strength.
    weakest_side_strength_rank_weight: float = 0.0

    # Selection: duplicates above this IoU are suppressed; overlaps are resolved by score.
    duplicate_iou_threshold: float = 0.5
    maximum_contained_fraction: float = 0.75  # smaller quad mostly inside a larger kept one
    relative_area_floor: float = 0.3  # of the median kept quad area; smaller ones are fragments
    maximum_covered_fraction: float = 0.75  # candidate area already covered by the union of kept quads, tuned (sweep_v1)

    # Full-resolution sub-pixel side refinement.
    refine_corners_at_full_resolution: bool = True
    refinement_search_fraction: float = 0.006  # of the full-res long side, each way
    refinement_samples_per_side: int = 60
    refinement_minimum_step_strength: float = 3.0

    extra_notes: dict = field(default_factory=dict)

    def to_json_dict(self) -> dict:
        """Plain-dict form for the predictions file header."""
        return asdict(self)


def odd_kernel_size(fraction_of_long_side: float, long_side_pixels: int, minimum_size: int = 3) -> int:
    """Convert a fraction of the long side into an odd kernel size in pixels."""
    kernel_size = max(minimum_size, int(round(fraction_of_long_side * long_side_pixels)))
    return kernel_size if kernel_size % 2 == 1 else kernel_size + 1
