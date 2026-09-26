"""Keep a check's sides from locking onto the edges of OTHER detected checks.

Checks overlap in loose layouts. Where check B lies over check A, B's edge crosses A's
search band and looks exactly like a paper edge (paper on one side, something else on the
other). We cannot tell from pixels which check is on top, so we make the conservative
choice: every score-grid cell that falls inside another detection's quad (dilated by a few
px so B's own edge is covered too) carries no evidence. A's side is then fitted on its
stretches that are clear of other checks, and a hidden corner is extrapolated from them.

- Cost: one convex point-in-quad test per (sample, offset) cell per nearby quad.
- Only quads whose bounding box comes near this check are tested.
"""

import numpy as np

from experiments.detection.refinement.edge_profile_sampling import SideScoreProfiles


def dilate_convex_quad(corners: np.ndarray, dilation_pixels: float) -> np.ndarray:
    """Move every side of a convex quad outward by `dilation_pixels` (either winding)."""
    centroid = corners.mean(axis=0)
    moved_lines = []
    for side_index in range(4):
        start, end = corners[side_index], corners[(side_index + 1) % 4]
        direction = (end - start) / max(np.linalg.norm(end - start), 1e-9)
        outward_normal = np.array([direction[1], -direction[0]])
        if np.dot(outward_normal, start - centroid) < 0:
            outward_normal = -outward_normal
        moved_lines.append((start + dilation_pixels * outward_normal, direction))
    dilated = []
    for corner_index in range(4):
        (first_point, first_direction), (second_point, second_direction) = moved_lines[corner_index - 1], moved_lines[corner_index]
        matrix = np.array([first_direction, -second_direction]).T
        if abs(np.linalg.det(matrix)) < 1e-9:
            dilated.append(corners[corner_index])
            continue
        parameters = np.linalg.solve(matrix, second_point - first_point)
        dilated.append(first_point + parameters[0] * first_direction)
    return np.array(dilated)


def points_inside_convex_quad(points: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """(N,) bool: point strictly on the inner side of all four edges (either winding)."""
    edge_starts = corners
    edge_vectors = np.roll(corners, -1, axis=0) - corners
    # Cross product of each edge with (point - edge start); all the same sign = inside.
    relative = points[:, None, :] - edge_starts[None, :, :]
    cross = edge_vectors[None, :, 0] * relative[..., 1] - edge_vectors[None, :, 1] * relative[..., 0]
    return np.all(cross > 0, axis=1) | np.all(cross < 0, axis=1)


def select_nearby_quads(own_corners: np.ndarray, other_quads: list[np.ndarray], reach_pixels: float) -> list[np.ndarray]:
    """Other quads whose bounding box comes within `reach_pixels` of this quad's bounding box."""
    own_minimum, own_maximum = own_corners.min(axis=0) - reach_pixels, own_corners.max(axis=0) + reach_pixels
    return [
        quad for quad in other_quads
        if np.all(quad.max(axis=0) >= own_minimum) and np.all(quad.min(axis=0) <= own_maximum)
    ]


def mask_cells_inside_other_quads(profiles: SideScoreProfiles, dilated_other_quads: list[np.ndarray]) -> int:
    """Set score cells lying inside any other (dilated) quad to -inf in place; returns how many."""
    if not dilated_other_quads:
        return 0
    number_of_samples, number_of_offsets = profiles.scores.shape
    positions = np.repeat(profiles.positions_pixels, number_of_offsets)
    offsets = np.tile(profiles.normal_offsets, number_of_samples)
    cell_points = profiles.points_at(positions, offsets)
    covered = np.zeros(len(cell_points), dtype=bool)
    for quad in dilated_other_quads:
        covered |= points_inside_convex_quad(cell_points, quad)
    covered_grid = covered.reshape(number_of_samples, number_of_offsets)
    profiles.scores[covered_grid] = -np.inf
    return int(covered_grid.sum())
