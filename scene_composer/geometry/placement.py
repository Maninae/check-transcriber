"""Physical layout of checks on the sheet, in inches, before any camera is chosen.

A property manager lays the checks out, then frames the photo around them, so layout
comes first and the camera is fitted to the group afterwards (scene_framing.py).

Layout modes:
- grid: rows packed left to right with real gaps (0.1 to 1.5 in), rows sometimes ragged
  or offset, mostly one orientation with a few flipped 180 or quarter-turned, small jitter.
- loose_overlap: the same packing with small or negative gaps and more angle jitter, so a
  corner lands over a neighbour.
- loose_fan: a shingled row, like playing cards spread on a table.
Loose layouts are spread apart until every check keeps at least 70% of its area visible.

Units: inches on the sheet plane, y pointing down (same as images). Rotation is
counter-clockwise on screen in degrees, matching cv2.getRotationMatrix2D.
"""

from dataclasses import dataclass

import cv2
import numpy as np

GAP_RANGE_INCHES = (0.1, 1.5)
MIN_GAP_INCHES = 0.1
GAP_JITTER_FRACTION = 0.35
LOOSE_GAP_RANGE_INCHES = (-0.9, 0.3)
FLIP_180_PROBABILITY = 0.15
QUARTER_TURN_PROBABILITY = 0.07
PORTRAIT_CHECKS_PROBABILITY = 0.08
GRID_ANGLE_JITTER_SIGMA_DEGREES = 1.5
GRID_ANGLE_JITTER_MAX_DEGREES = 5.0
LOOSE_ANGLE_JITTER_SIGMA_DEGREES = 5.0
LOOSE_ANGLE_JITTER_MAX_DEGREES = 14.0
ROW_VERTICAL_JITTER_INCHES = 0.08
RAGGED_ROW_PROBABILITY = 0.4
RAGGED_ROW_MAX_OFFSET_FRACTION = 0.35
PHOTO_ASPECT = 4 / 3
COLUMN_CHOICE_TEMPERATURE = 0.25
EMPTY_CELL_PENALTY = 0.12
MAX_COMFORTABLE_LINE = 4
LONG_LINE_PENALTY = 0.5
FAN_MAX_CHECKS = 4
FAN_STEP_FRACTION_RANGE = (0.74, 0.86)
FAN_ANGLE_STEP_DEGREES_RANGE = (1.5, 6.0)
FAN_VERTICAL_PROBABILITY = 0.4
LOOSE_MIN_VISIBLE_FRACTION = 0.7
VISIBILITY_RASTER_PIXELS_PER_INCH = 12
SPREAD_STEP_FACTOR = 1.06
SPREAD_MAX_STEPS = 40
LOOSE_FAN_SHARE = 0.4


@dataclass
class CheckPlacement:
    """Where one check lies on the sheet: center in inches and rotation (ccw, degrees)."""

    center_x_inches: float
    center_y_inches: float
    rotation_degrees: float


def rotated_extent(width_inches: float, height_inches: float, degrees: float) -> tuple[float, float]:
    """Axis-aligned width and height of a rotated rectangle."""
    radians = np.deg2rad(degrees)
    cos_value, sin_value = abs(np.cos(radians)), abs(np.sin(radians))
    return width_inches * cos_value + height_inches * sin_value, width_inches * sin_value + height_inches * cos_value


def placement_corners_inches(size_inches: tuple[float, float], placement: CheckPlacement) -> np.ndarray:
    """The 4 corners (check's own TL, TR, BR, BL) of a placed flat check, in sheet inches."""
    width, height = size_inches
    local = np.array([[-width / 2, -height / 2], [width / 2, -height / 2], [width / 2, height / 2], [-width / 2, height / 2]])
    radians = np.deg2rad(placement.rotation_degrees)
    rotation = np.array([[np.cos(radians), np.sin(radians)], [-np.sin(radians), np.cos(radians)]])
    return local @ rotation.T + [placement.center_x_inches, placement.center_y_inches]


def clipped_normal(rng: np.random.Generator, sigma: float, limit: float) -> float:
    """A normal sample clipped to [-limit, limit]."""
    return float(np.clip(rng.normal(0, sigma), -limit, limit))


def sample_rotations(count: int, rng: np.random.Generator, jitter_sigma: float, jitter_max: float) -> list[float]:
    """Mostly one shared orientation; some flipped 180 or quarter-turned; small angle jitter."""
    base = 90.0 * rng.choice([1, 3]) if rng.random() < PORTRAIT_CHECKS_PROBABILITY else 0.0
    rotations = []
    for _ in range(count):
        roll = rng.random()
        angle = base + (180 if roll < FLIP_180_PROBABILITY else 0)
        if FLIP_180_PROBABILITY <= roll < FLIP_180_PROBABILITY + QUARTER_TURN_PROBABILITY:
            angle += 90 * rng.choice([1, 3])
        rotations.append(float(angle + clipped_normal(rng, jitter_sigma, jitter_max)))
    return rotations


def choose_column_count(extents: list[tuple[float, float]], gap_inches: float, rng: np.random.Generator) -> int:
    """Columns for the grid: groups whose shape suits a 4:3 photo (either way round) are likelier."""
    count = len(extents)
    mean_width = float(np.mean([e[0] for e in extents]))
    mean_height = float(np.mean([e[1] for e in extents]))
    scores = []
    for columns in range(1, count + 1):
        rows = int(np.ceil(count / columns))
        aspect = (columns * mean_width + (columns - 1) * gap_inches) / (rows * mean_height + (rows - 1) * gap_inches)
        mismatch = abs(abs(np.log(aspect)) - np.log(PHOTO_ASPECT))
        long_line = max(0, max(rows, columns) - MAX_COMFORTABLE_LINE)  # nobody lays 7 checks in one line
        scores.append(mismatch + EMPTY_CELL_PENALTY * (rows * columns - count) + LONG_LINE_PENALTY * long_line)
    weights = np.exp(-(np.array(scores) - min(scores)) / COLUMN_CHOICE_TEMPERATURE)
    return int(rng.choice(np.arange(1, count + 1), p=weights / weights.sum()))


def pack_rows(extents: list[tuple[float, float]], rotations: list[float], columns: int, gap_range: tuple[float, float],
              min_gap: float, rng: np.random.Generator) -> list[CheckPlacement]:
    """Pack checks row by row with jittered gaps; rows may be ragged (offset) and the last row partial."""
    base_gap_x = rng.uniform(*gap_range)
    base_gap_y = base_gap_x * rng.uniform(0.6, 1.4) if rng.random() < 0.7 else rng.uniform(*gap_range)
    ragged = rng.random() < RAGGED_ROW_PROBABILITY
    mean_width = float(np.mean([e[0] for e in extents]))
    placements, cursor_y = [], 0.0
    rows = [list(range(start, min(start + columns, len(extents)))) for start in range(0, len(extents), columns)]
    full_row_width = columns * mean_width + (columns - 1) * base_gap_x
    for row in rows:
        row_height = max(extents[i][1] for i in row)
        row_width = sum(extents[i][0] for i in row) + (len(row) - 1) * base_gap_x
        cursor_x = rng.uniform(0, RAGGED_ROW_MAX_OFFSET_FRACTION) * mean_width if ragged else 0.0
        if len(row) < columns and not ragged:
            cursor_x = rng.choice([0.0, (full_row_width - row_width) / 2, full_row_width - row_width])
        for index in row:
            width, height = extents[index]
            slack = (row_height - height) / 2
            center_y = cursor_y + row_height / 2 + clipped_normal(rng, ROW_VERTICAL_JITTER_INCHES, ROW_VERTICAL_JITTER_INCHES * 2)
            center_y += rng.uniform(-slack, slack)
            placements.append(CheckPlacement(cursor_x + width / 2, center_y, rotations[index]))
            cursor_x += width + max(min_gap, base_gap_x * (1 + rng.uniform(-GAP_JITTER_FRACTION, GAP_JITTER_FRACTION)))
        cursor_y += row_height + max(min_gap, base_gap_y * (1 + rng.uniform(-GAP_JITTER_FRACTION, GAP_JITTER_FRACTION)))
    return placements


def plan_grid_layout(sizes_inches: list[tuple[float, float]], rng: np.random.Generator) -> list[CheckPlacement]:
    """Rough grid with realistic gaps; returns placements in paste order."""
    rotations = sample_rotations(len(sizes_inches), rng, GRID_ANGLE_JITTER_SIGMA_DEGREES, GRID_ANGLE_JITTER_MAX_DEGREES)
    extents = [rotated_extent(w, h, angle) for (w, h), angle in zip(sizes_inches, rotations)]
    columns = choose_column_count(extents, float(np.mean(GAP_RANGE_INCHES)), rng)
    return pack_rows(extents, rotations, columns, GAP_RANGE_INCHES, MIN_GAP_INCHES, rng)


def plan_loose_overlap_layout(sizes_inches: list[tuple[float, float]], rng: np.random.Generator) -> list[CheckPlacement]:
    """Grid-like packing with small or negative gaps and freer angles, so corners overlap neighbours."""
    rotations = sample_rotations(len(sizes_inches), rng, LOOSE_ANGLE_JITTER_SIGMA_DEGREES, LOOSE_ANGLE_JITTER_MAX_DEGREES)
    extents = [rotated_extent(w, h, angle) for (w, h), angle in zip(sizes_inches, rotations)]
    columns = choose_column_count(extents, 0.0, rng)
    order = rng.permutation(len(sizes_inches))  # paste order differs from reading order, so overlaps go both ways
    placements = pack_rows(extents, rotations, columns, LOOSE_GAP_RANGE_INCHES, LOOSE_GAP_RANGE_INCHES[0], rng)
    return spread_until_visible(sizes_inches, [placements[i] for i in order], [int(i) for i in order])


def plan_fan_layout(sizes_inches: list[tuple[float, float]], rng: np.random.Generator) -> list[CheckPlacement]:
    """A shingled row like spread playing cards: each check steps along and turns a little further."""
    vertical = rng.random() < FAN_VERTICAL_PROBABILITY
    angle = 180.0 if rng.random() < FLIP_180_PROBABILITY else 0.0
    angle_step = rng.uniform(*FAN_ANGLE_STEP_DEGREES_RANGE) * rng.choice([-1, 1])
    angle -= angle_step * (len(sizes_inches) - 1) / 2
    placements, cursor = [], 0.0
    for index, (width, height) in enumerate(sizes_inches):
        step_length = height if vertical else width
        if index:
            cursor += rng.uniform(*FAN_STEP_FRACTION_RANGE) * step_length
        lateral = 0.02 * (index - (len(sizes_inches) - 1) / 2) ** 2 * step_length  # a slight arc
        rotation = angle + index * angle_step + clipped_normal(rng, 1.0, 2.5)
        center = (lateral, cursor) if vertical else (cursor, lateral)
        placements.append(CheckPlacement(center[0], center[1], float(rotation)))
    return spread_until_visible(sizes_inches, placements, list(range(len(sizes_inches))))


def visible_fractions(sizes_inches: list[tuple[float, float]], placements: list[CheckPlacement], order: list[int]) -> list[float]:
    """Share of each check's area left uncovered by checks pasted after it (raster estimate)."""
    all_corners = [placement_corners_inches(sizes_inches[i], p) for i, p in zip(order, placements)]
    origin = np.min([c.min(axis=0) for c in all_corners], axis=0) - 1
    extent = np.max([c.max(axis=0) for c in all_corners], axis=0) - origin + 1
    shape = tuple(int(np.ceil(v * VISIBILITY_RASTER_PIXELS_PER_INCH)) for v in extent[::-1])
    owner = np.zeros(shape, np.int32)
    for index, corners in enumerate(all_corners):
        cv2.fillPoly(owner, [np.round((corners - origin) * VISIBILITY_RASTER_PIXELS_PER_INCH).astype(np.int32)], index + 1)
    fractions = []
    for index, corners in enumerate(all_corners):
        mask = np.zeros(shape, np.uint8)
        cv2.fillPoly(mask, [np.round((corners - origin) * VISIBILITY_RASTER_PIXELS_PER_INCH).astype(np.int32)], 1)
        fractions.append(np.count_nonzero(owner[mask > 0] == index + 1) / max(1, np.count_nonzero(mask)))
    return fractions


def spread_until_visible(sizes_inches: list[tuple[float, float]], placements: list[CheckPlacement], order: list[int],
                         min_visible_fraction: float = LOOSE_MIN_VISIBLE_FRACTION) -> list[CheckPlacement]:
    """Push checks apart from the group center until each keeps `min_visible_fraction` visible; returns them in paste order.

    `placements[k]` belongs to check `order[k]`; the result is re-indexed so result[i] is check i,
    and paste order is the caller's check order, so the visibility check uses that order.
    """
    by_check = [None] * len(placements)
    for check_index, placement in zip(order, placements):
        by_check[check_index] = placement
    identity = list(range(len(by_check)))
    for _ in range(SPREAD_MAX_STEPS):
        if len(by_check) < 2 or min(visible_fractions(sizes_inches, by_check, identity)) >= min_visible_fraction:
            break
        center_x = np.mean([p.center_x_inches for p in by_check]); center_y = np.mean([p.center_y_inches for p in by_check])
        for placement in by_check:
            placement.center_x_inches = center_x + (placement.center_x_inches - center_x) * SPREAD_STEP_FACTOR
            placement.center_y_inches = center_y + (placement.center_y_inches - center_y) * SPREAD_STEP_FACTOR
    return by_check


def plan_layout(sizes_inches: list[tuple[float, float]], loose_probability: float,
                rng: np.random.Generator) -> tuple[str, list[CheckPlacement]]:
    """Pick a layout mode and plan it. Returns (mode name, placements indexed like `sizes_inches`)."""
    if len(sizes_inches) > 1 and rng.random() < loose_probability:
        if rng.random() < LOOSE_FAN_SHARE and len(sizes_inches) <= FAN_MAX_CHECKS:
            return "loose_fan", plan_fan_layout(sizes_inches, rng)
        return "loose_overlap", plan_loose_overlap_layout(sizes_inches, rng)
    return "grid", plan_grid_layout(sizes_inches, rng)
