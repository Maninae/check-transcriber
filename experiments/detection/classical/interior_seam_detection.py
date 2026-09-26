"""Detect a straight border running across a candidate's interior (a "seam").

A quad assembled from line segments can span two or more checks plus the background
between them: its four sides borrow real borders from different checks, so edge support
is high. Its tell is a straight, strong step edge crossing the interior parallel to one
of its sides, the border of a check inside it. A real check has none: printed rules are
erased by the text-suppression closing, and a hand-shadow boundary is soft and curved.

For each of the two side directions we sample lines across the quad (interpolated
between opposite sides, so perspective is respected), measure the gradient along the
line's normal at each sample, and record the fraction of samples above threshold. The
seam strength is the maximum over all lines.
"""

import cv2
import numpy as np

from experiments.detection.classical.working_image_channels import WorkingImageChannels

SEAM_LINE_FRACTIONS = np.linspace(0.12, 0.88, 17)  # positions across the quad, away from its own sides
SAMPLES_PER_SEAM_LINE = 48
SEAM_LINE_END_MARGIN = 0.04  # skip where the line meets the quad's own sides
SEAM_GRADIENT_SEARCH_OFFSETS = (-1.5, 0.0, 1.5)


def sample_gradient_along_normal(channels: WorkingImageChannels, points: np.ndarray, unit_normals: np.ndarray) -> np.ndarray:
    """|gradient . normal| at each point, maximized over a few pixels of normal offset."""
    best_response = np.zeros(len(points))
    for offset in SEAM_GRADIENT_SEARCH_OFFSETS:
        shifted = (points + offset * unit_normals).astype(np.float32)
        map_x, map_y = shifted[:, 0].reshape(1, -1), shifted[:, 1].reshape(1, -1)
        gradient_x = cv2.remap(channels.gradient_x, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).ravel()
        gradient_y = cv2.remap(channels.gradient_y, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).ravel()
        best_response = np.maximum(best_response, np.abs(gradient_x * unit_normals[:, 0] + gradient_y * unit_normals[:, 1]))
    return best_response


def measure_interior_seam_strength(corners: np.ndarray, channels: WorkingImageChannels, gradient_threshold: float) -> float:
    """Largest fraction of on-edge samples along any interior line parallel to a side."""
    image_height, image_width = channels.gradient_x.shape
    along_fractions = np.linspace(SEAM_LINE_END_MARGIN, 1.0 - SEAM_LINE_END_MARGIN, SAMPLES_PER_SEAM_LINE)
    strongest_seam = 0.0
    for first_side_index in (0, 1):
        # lines run from side (first_side_index) toward its opposite side, parallel to the other pair
        side_start, side_end = corners[first_side_index], corners[(first_side_index + 1) % 4]
        opposite_start, opposite_end = corners[(first_side_index + 3) % 4], corners[(first_side_index + 2) % 4]
        for across_fraction in SEAM_LINE_FRACTIONS:
            line_start = side_start + across_fraction * (opposite_start - side_start)
            line_end = side_end + across_fraction * (opposite_end - side_end)
            line_vector = line_end - line_start
            line_length = float(np.linalg.norm(line_vector))
            if line_length < 1.0:
                continue
            points = line_start + along_fractions[:, None] * line_vector
            in_frame = (points[:, 0] >= 0) & (points[:, 0] < image_width) & (points[:, 1] >= 0) & (points[:, 1] < image_height)
            if in_frame.sum() < SAMPLES_PER_SEAM_LINE // 2:
                continue
            unit_normal = np.array([-line_vector[1], line_vector[0]]) / line_length
            response = sample_gradient_along_normal(channels, points[in_frame], np.repeat(unit_normal[None, :], in_frame.sum(), axis=0))
            strongest_seam = max(strongest_seam, float((response > gradient_threshold).mean()))
    return strongest_seam
