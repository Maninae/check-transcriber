"""Plan where each check lies on the sheet plane: position, rotation, and scale.

Two layout modes, mirroring how a property manager lays checks out:
- grid: a rough grid, mostly one shared orientation, some flipped or quarter-turned,
  small angle jitter, optionally one nudged into a neighbour (overlap).
- pile: loose placement at any angle, overlaps allowed but kept small.
Either mode may push one edge check partly out of the camera's view.

Scale is physical: one plane pixels-per-inch for the whole scene, times a tiny per-check
jitter, so a business check really is larger than a personal one.
"""

from dataclasses import dataclass

import numpy as np

GRID_FILL_RANGE = (0.8, 0.95)
PILE_COVERAGE_RANGE = (0.22, 0.36)
PER_CHECK_SCALE_JITTER = 0.03
GRID_ANGLE_JITTER_DEGREES = 5.0
OVERLAP_PROBABILITY = 0.25
OUT_OF_FRAME_PROBABILITY = 0.2
PILE_POSITION_TRIES = 60


@dataclass
class CheckPlacement:
    """Placement of one check: plane center, content rotation (ccw, degrees), plane px per inch."""

    center_x: float
    center_y: float
    rotation_degrees: float
    pixels_per_inch: float


def rotated_extent(width_inches: float, height_inches: float, degrees: float) -> tuple[float, float]:
    """Axis-aligned width and height of a rotated rectangle."""
    radians = np.deg2rad(degrees)
    cos_value, sin_value = abs(np.cos(radians)), abs(np.sin(radians))
    return width_inches * cos_value + height_inches * sin_value, width_inches * sin_value + height_inches * cos_value


def sample_grid_rotations(count: int, safe_rect: tuple[float, float, float, float], rng: np.random.Generator) -> list[float]:
    """Mostly a shared orientation (landscape, or portrait on a tall frame), some flips and quarter turns."""
    x0, y0, x1, y1 = safe_rect
    base = 90.0 if (y1 - y0) > (x1 - x0) * 1.2 and rng.random() < 0.6 else 0.0
    rotations = []
    for _ in range(count):
        angle = base
        roll = rng.random()
        if roll < 0.2:
            angle += 180
        elif roll < 0.3:
            angle += 90 * rng.choice([1, 3])
        elif roll < 0.35:
            angle = rng.uniform(0, 360)
        rotations.append(float(angle + np.clip(rng.normal(0, GRID_ANGLE_JITTER_DEGREES / 2), -GRID_ANGLE_JITTER_DEGREES, GRID_ANGLE_JITTER_DEGREES)))
    return rotations


def plan_grid_layout(sizes_inches: list[tuple[float, float]], safe_rect: tuple[float, float, float, float],
                     rng: np.random.Generator) -> list[CheckPlacement]:
    """Rough-grid placement; returns placements in paste order."""
    count = len(sizes_inches)
    x0, y0, x1, y1 = safe_rect
    rotations = sample_grid_rotations(count, safe_rect, rng)
    extents = [rotated_extent(w, h, angle) for (w, h), angle in zip(sizes_inches, rotations)]
    max_extent_w = max(e[0] for e in extents)
    max_extent_h = max(e[1] for e in extents)
    fill = rng.uniform(*GRID_FILL_RANGE)
    best = None
    for columns in range(1, count + 1):
        rows = int(np.ceil(count / columns))
        ppi = fill * min((x1 - x0) / columns / max_extent_w, (y1 - y0) / rows / max_extent_h)
        if best is None or ppi > best[0]:
            best = (ppi, columns, rows)
    ppi, columns, rows = best
    cell_w, cell_h = (x1 - x0) / columns, (y1 - y0) / rows
    cells = rng.permutation(columns * rows)[:count]
    placements = []
    for cell, angle in zip(cells, rotations):
        column, row = cell % columns, cell // columns
        jitter_x = rng.uniform(-0.5, 0.5) * cell_w * (1 - fill) * 0.8
        jitter_y = rng.uniform(-0.5, 0.5) * cell_h * (1 - fill) * 0.8
        placements.append(CheckPlacement(x0 + (column + 0.5) * cell_w + jitter_x, y0 + (row + 0.5) * cell_h + jitter_y,
                                         angle, ppi * rng.uniform(1 - PER_CHECK_SCALE_JITTER, 1 + PER_CHECK_SCALE_JITTER)))
    if count > 1 and rng.random() < OVERLAP_PROBABILITY:
        mover, target = rng.choice(count, 2, replace=False)
        direction = np.array([placements[target].center_x - placements[mover].center_x, placements[target].center_y - placements[mover].center_y])
        distance = np.linalg.norm(direction)
        if distance > 0:
            extent = np.array(extents[mover]) * ppi
            shift = direction / distance * rng.uniform(0.12, 0.25) * float(np.abs(direction / distance) @ extent)
            placements[mover].center_x += shift[0]
            placements[mover].center_y += shift[1]
    return placements


def plan_pile_layout(sizes_inches: list[tuple[float, float]], safe_rect: tuple[float, float, float, float],
                     rng: np.random.Generator) -> list[CheckPlacement]:
    """Loose placement at arbitrary angles, choosing each spot to limit overlap."""
    x0, y0, x1, y1 = safe_rect
    total_area_inches = sum(w * h for w, h in sizes_inches)
    ppi = float(np.sqrt(rng.uniform(*PILE_COVERAGE_RANGE) * (x1 - x0) * (y1 - y0) / total_area_inches))
    placements: list[CheckPlacement] = []
    for width_inches, height_inches in sizes_inches:
        angle = float(rng.uniform(0, 360))
        extent_w, extent_h = (v * ppi for v in rotated_extent(width_inches, height_inches, angle))
        radius = 0.5 * np.hypot(width_inches, height_inches) * ppi * 0.6
        best_spot, best_overlap = None, np.inf
        for _ in range(PILE_POSITION_TRIES):
            spot = (rng.uniform(x0 + extent_w / 2, max(x0 + extent_w / 2 + 1, x1 - extent_w / 2)),
                    rng.uniform(y0 + extent_h / 2, max(y0 + extent_h / 2 + 1, y1 - extent_h / 2)))
            overlap = sum(max(0.0, 2 * radius - np.hypot(spot[0] - p.center_x, spot[1] - p.center_y)) for p in placements)
            if overlap < best_overlap:
                best_spot, best_overlap = spot, overlap
        placements.append(CheckPlacement(best_spot[0], best_spot[1], angle,
                                         ppi * rng.uniform(1 - PER_CHECK_SCALE_JITTER, 1 + PER_CHECK_SCALE_JITTER)))
    return placements


def push_one_check_out_of_frame(placements: list[CheckPlacement], sizes_inches: list[tuple[float, float]],
                                safe_rect: tuple[float, float, float, float], rng: np.random.Generator) -> int:
    """Move the check nearest a random safe-rect edge outward so part of it leaves the view; return its index."""
    x0, y0, x1, y1 = safe_rect
    edge = rng.choice(["left", "right", "top", "bottom"])
    key = {"left": lambda p: p.center_x, "right": lambda p: -p.center_x,
           "top": lambda p: p.center_y, "bottom": lambda p: -p.center_y}[edge]
    index = min(range(len(placements)), key=lambda i: key(placements[i]))
    placement = placements[index]
    extent_w, extent_h = (v * placement.pixels_per_inch for v in rotated_extent(*sizes_inches[index], placement.rotation_degrees))
    outside_fraction = rng.uniform(0.2, 0.45)
    if edge == "left":
        placement.center_x = x0 - extent_w * (outside_fraction - 0.5)
    elif edge == "right":
        placement.center_x = x1 + extent_w * (outside_fraction - 0.5)
    elif edge == "top":
        placement.center_y = y0 - extent_h * (outside_fraction - 0.5)
    else:
        placement.center_y = y1 + extent_h * (outside_fraction - 0.5)
    return index
