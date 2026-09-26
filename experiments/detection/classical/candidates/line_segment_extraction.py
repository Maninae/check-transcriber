"""Straight edge segments for the line-based hypothesis generator.

Canny on the text-suppressed lightness OR color (`combined_edge_map`), then `HoughLinesP`, then a greedy merge of
collinear pieces (a check border broken by a shadow or a faint stretch comes back as one
segment). Check borders are long and straight; bedsheet wrinkles, carpet and crochet
texture are curvy or short, so a length floor removes most clutter.

Edges deep inside textured surfaces (carpet, crochet, rugs) are dropped before Hough: a
min filter of the texture map stays high there but is low next to any smooth paper, so
real check borders keep their edges. Without this, crochet yields ~8k raw segments.

Segments are returned as an (N, 4) float array of (x1, y1, x2, y2), longest first.
"""

import cv2
import numpy as np

from experiments.detection.classical.classical_detector_config import (
    ClassicalDetectorConfig,
    odd_kernel_size,
)
from experiments.detection.classical.preprocessing.combined_edge_map import build_combined_canny_edges
from experiments.detection.classical.preprocessing.working_image_channels import WorkingImageChannels

HOUGH_ANGLE_RESOLUTION_RADIANS = np.pi / 360
MERGE_ANGLE_TOLERANCE_DEGREES = 2.0
MERGE_OFFSET_TOLERANCE_PIXELS = 3.0
HOUGH_VOTE_FRACTION_OF_MINIMUM_LENGTH = 0.6
HOUGH_PIECE_FRACTION_OF_MINIMUM_LENGTH = 0.6  # pieces this short may still merge into a long line
MAXIMUM_RAW_SEGMENTS_TO_MERGE = 400


def segment_angle_degrees(segments: np.ndarray) -> np.ndarray:
    """Undirected angle of each segment in [0, 180)."""
    return np.degrees(np.arctan2(segments[:, 3] - segments[:, 1], segments[:, 2] - segments[:, 0])) % 180.0


def merge_collinear_segments(segments: np.ndarray, maximum_gap_pixels: float) -> np.ndarray:
    """Greedily fuse segments on the same line whose extents overlap or nearly touch.

    Longest first; each segment is tested against every merged line at once (vectorized).
    """
    order = np.argsort(-np.hypot(segments[:, 2] - segments[:, 0], segments[:, 3] - segments[:, 1]))
    merged_starts = np.zeros((0, 2))
    merged_ends = np.zeros((0, 2))
    minimum_cosine = np.cos(np.radians(MERGE_ANGLE_TOLERANCE_DEGREES))
    for segment in segments[order]:
        segment_start, segment_end = segment[:2], segment[2:]
        segment_direction = segment_end - segment_start
        segment_length = float(np.linalg.norm(segment_direction))
        if segment_length < 1e-6:
            continue
        if len(merged_starts):
            merged_vectors = merged_ends - merged_starts
            merged_lengths = np.linalg.norm(merged_vectors, axis=1)
            unit_directions = merged_vectors / merged_lengths[:, None]
            unit_normals = np.stack([-unit_directions[:, 1], unit_directions[:, 0]], axis=1)
            cosines = np.abs(unit_directions @ (segment_direction / segment_length))
            start_offsets = ((segment_start - merged_starts) * unit_normals).sum(axis=1)
            end_offsets = ((segment_end - merged_starts) * unit_normals).sum(axis=1)
            start_positions = ((segment_start - merged_starts) * unit_directions).sum(axis=1)
            end_positions = ((segment_end - merged_starts) * unit_directions).sum(axis=1)
            mergeable = (
                (cosines >= minimum_cosine)
                & (np.maximum(np.abs(start_offsets), np.abs(end_offsets)) <= MERGE_OFFSET_TOLERANCE_PIXELS)
                & (np.minimum(start_positions, end_positions) <= merged_lengths + maximum_gap_pixels)
                & (np.maximum(start_positions, end_positions) >= -maximum_gap_pixels)
            )
            if mergeable.any():
                merged_index = int(np.flatnonzero(mergeable)[0])
                low_position = min(0.0, start_positions[merged_index], end_positions[merged_index])
                high_position = max(merged_lengths[merged_index], start_positions[merged_index], end_positions[merged_index])
                anchor = merged_starts[merged_index].copy()
                merged_starts[merged_index] = anchor + low_position * unit_directions[merged_index]
                merged_ends[merged_index] = anchor + high_position * unit_directions[merged_index]
                continue
        merged_starts = np.vstack([merged_starts, segment_start])
        merged_ends = np.vstack([merged_ends, segment_end])
    return np.concatenate([merged_starts, merged_ends], axis=1)


def extract_line_segments(channels: WorkingImageChannels, config: ClassicalDetectorConfig) -> np.ndarray:
    """Long straight edge segments, longest first, capped at `line_maximum_segments`."""
    long_side = channels.long_side_pixels
    edges = build_combined_canny_edges(channels, config.line_canny_low, config.line_canny_high, config)
    texture_window = odd_kernel_size(config.texture_window_fraction, long_side)
    nearest_smooth_texture = cv2.erode(
        channels.texture_std, cv2.getStructuringElement(cv2.MORPH_RECT, (2 * texture_window + 1, 2 * texture_window + 1))
    )
    edges[nearest_smooth_texture > config.line_texture_suppression_std] = 0
    minimum_length = config.line_minimum_length_fraction * long_side
    raw_segments = cv2.HoughLinesP(
        edges,
        1,
        HOUGH_ANGLE_RESOLUTION_RADIANS,
        threshold=int(minimum_length * HOUGH_VOTE_FRACTION_OF_MINIMUM_LENGTH),
        minLineLength=minimum_length * HOUGH_PIECE_FRACTION_OF_MINIMUM_LENGTH,
        maxLineGap=config.line_maximum_gap_fraction * long_side,
    )
    if raw_segments is None:
        return np.zeros((0, 4))
    raw_segments = raw_segments.reshape(-1, 4).astype(np.float64)
    raw_lengths = np.hypot(raw_segments[:, 2] - raw_segments[:, 0], raw_segments[:, 3] - raw_segments[:, 1])
    raw_segments = raw_segments[np.argsort(-raw_lengths)][:MAXIMUM_RAW_SEGMENTS_TO_MERGE]
    merged_segments = merge_collinear_segments(raw_segments, config.line_merge_gap_fraction * long_side)
    lengths = np.hypot(merged_segments[:, 2] - merged_segments[:, 0], merged_segments[:, 3] - merged_segments[:, 1])
    merged_segments = merged_segments[lengths >= minimum_length]
    lengths = lengths[lengths >= minimum_length]
    return merged_segments[np.argsort(-lengths)][: config.line_maximum_segments]
