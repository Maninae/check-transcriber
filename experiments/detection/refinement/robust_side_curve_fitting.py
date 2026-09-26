"""Robust fit of a side's edge as offset = polynomial(position) in the side's own frame.

The frame is the current approximate side (start corner, unit tangent, outward normal),
so the paper edge is nearly horizontal in it and vertical residuals equal perpendicular
ones. Degree 1 is a straight side; degree 2 captures a curl's bow, so extrapolating it
to the corner lands on the physical corner instead of the chord's end.

Fitting: RANSAC over point pairs for the dominant straight line (an overlapping check or
a fold owns the outliers), then Tukey-biweight IRLS for the chosen degree. Positions are
normalised to [-1, 1] over the side for conditioning. Plain numpy, portable to JS.
"""

from dataclasses import dataclass

import numpy as np

TUKEY_REWEIGHTING_ROUNDS = 5
TUKEY_SCALE_INLIER_MULTIPLE = 2.0
NEWTON_ITERATIONS = 8
NEWTON_MAXIMUM_STEP_PIXELS = 1e4


@dataclass
class SideCurve:
    """Edge curve point(s) = start + s * tangent + poly((s - mid) / half) * normal."""

    side_start: np.ndarray
    unit_tangent: np.ndarray
    unit_normal: np.ndarray
    side_length: float
    coefficients: np.ndarray  # constant, linear, [quadratic] in normalised position
    inlier_count: int = 0

    def normalised(self, positions: np.ndarray) -> np.ndarray:
        """Map positions (px along the side) to [-1, 1]."""
        half_length = max(self.side_length / 2, 1e-9)
        return (positions - half_length) / half_length

    def offsets_at(self, positions: np.ndarray) -> np.ndarray:
        """Offsets (px) of the curve at many positions at once."""
        return (self.normalised(positions)[:, None] ** np.arange(len(self.coefficients))[None, :]) @ self.coefficients

    def offset_and_slope(self, position: float) -> tuple[float, float]:
        """Offset (px) and d offset / d position at one position."""
        normalised_position = float(self.normalised(np.array([position]))[0])
        powers = normalised_position ** np.arange(len(self.coefficients))
        offset = float(self.coefficients @ powers)
        derivative_terms = np.arange(1, len(self.coefficients)) * normalised_position ** np.arange(len(self.coefficients) - 1)
        slope = float(self.coefficients[1:] @ derivative_terms) / max(self.side_length / 2, 1e-9)
        return offset, slope

    def point_and_derivative(self, position: float) -> tuple[np.ndarray, np.ndarray]:
        """Image point on the curve and its derivative with respect to position."""
        offset, slope = self.offset_and_slope(position)
        point = self.side_start + position * self.unit_tangent + offset * self.unit_normal
        return point, self.unit_tangent + slope * self.unit_normal


def fit_weighted_polynomial(normalised_positions: np.ndarray, offsets: np.ndarray, weights: np.ndarray, degree: int) -> np.ndarray:
    """Weighted least squares for offset = sum_k c_k * x^k."""
    design = normalised_positions[:, None] ** np.arange(degree + 1)[None, :]
    root_weights = np.sqrt(np.clip(weights, 0, None))[:, None]
    coefficients, *_ = np.linalg.lstsq(design * root_weights, offsets * root_weights[:, 0], rcond=None)
    return coefficients


def fit_robust_side_curve(
    curve_frame: SideCurve,
    positions: np.ndarray,
    offsets: np.ndarray,
    strengths: np.ndarray,
    degree: int,
    inlier_distance_pixels: float,
    ransac_iterations: int,
    random_generator: np.random.Generator,
) -> SideCurve | None:
    """RANSAC straight line, then Tukey IRLS polynomial of `degree`; None if < 2 points."""
    if len(positions) < max(2, degree + 1):
        return None
    normalised_positions = curve_frame.normalised(positions)
    weights = np.clip(strengths, 1e-6, None)
    # All RANSAC hypotheses at once: (iterations, points) residual matrix.
    pair_indices = random_generator.integers(0, len(positions), size=(ransac_iterations, 2))
    first_positions, second_positions = normalised_positions[pair_indices[:, 0]], normalised_positions[pair_indices[:, 1]]
    first_offsets, second_offsets = offsets[pair_indices[:, 0]], offsets[pair_indices[:, 1]]
    position_gaps = second_positions - first_positions
    usable = np.abs(position_gaps) > 1e-6
    slopes = np.where(usable, (second_offsets - first_offsets) / np.where(usable, position_gaps, 1.0), 0.0)
    predicted = first_offsets[:, None] + slopes[:, None] * (normalised_positions[None, :] - first_positions[:, None])
    inlier_matrix = (np.abs(offsets[None, :] - predicted) < inlier_distance_pixels) & usable[:, None]
    best_inliers = inlier_matrix[int(np.argmax(inlier_matrix @ weights))]
    if best_inliers.sum() < degree + 1:
        return None
    coefficients = fit_weighted_polynomial(normalised_positions[best_inliers], offsets[best_inliers], weights[best_inliers], 1)
    tukey_scale = TUKEY_SCALE_INLIER_MULTIPLE * inlier_distance_pixels
    for round_index in range(TUKEY_REWEIGHTING_ROUNDS):
        # Straight first; the curvature term joins once the inlier set has settled.
        round_degree = degree if round_index >= 1 else 1
        residuals = offsets - (normalised_positions[:, None] ** np.arange(len(coefficients))[None, :]) @ coefficients
        tukey_weights = np.where(np.abs(residuals) < tukey_scale, (1 - (residuals / tukey_scale) ** 2) ** 2, 0.0)
        if np.count_nonzero(tukey_weights) < round_degree + 1:
            break
        coefficients = fit_weighted_polynomial(normalised_positions, offsets, tukey_weights * weights, round_degree)
    residuals = offsets - (normalised_positions[:, None] ** np.arange(len(coefficients))[None, :]) @ coefficients
    return SideCurve(
        side_start=curve_frame.side_start,
        unit_tangent=curve_frame.unit_tangent,
        unit_normal=curve_frame.unit_normal,
        side_length=curve_frame.side_length,
        coefficients=coefficients,
        inlier_count=int((np.abs(residuals) < inlier_distance_pixels).sum()),
    )


def intersect_side_curves(incoming_curve: SideCurve, outgoing_curve: SideCurve) -> np.ndarray | None:
    """Corner where the incoming side's END meets the outgoing side's START (Newton).

    Returns None when the curves are near-parallel or Newton wanders off.
    """
    incoming_position, outgoing_position = incoming_curve.side_length, 0.0
    for _ in range(NEWTON_ITERATIONS):
        incoming_point, incoming_derivative = incoming_curve.point_and_derivative(incoming_position)
        outgoing_point, outgoing_derivative = outgoing_curve.point_and_derivative(outgoing_position)
        jacobian = np.column_stack([incoming_derivative, -outgoing_derivative])
        if abs(np.linalg.det(jacobian)) < 1e-6:
            return None
        step = np.linalg.solve(jacobian, outgoing_point - incoming_point)
        if not np.all(np.isfinite(step)) or np.abs(step).max() > NEWTON_MAXIMUM_STEP_PIXELS:
            return None
        incoming_position += step[0]
        outgoing_position += step[1]
        if np.abs(step).max() < 1e-4:
            break
    return incoming_curve.point_and_derivative(incoming_position)[0]
