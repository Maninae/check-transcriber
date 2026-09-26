"""Shapes of things held above the sheet, drawn as their shadow footprint on a coarse plane grid.

A phone held flat, the hand gripping it (palm, fingers wrapped over the long edges, thumb),
and the forearm running back toward the photographer; or a bare hand reaching in. Each
drawer returns a coverage mask plus a height map (inches above the sheet per covered pixel),
because the penumbra widens with height (cast_shadows.py).

All sizes are adult-hand and phone dimensions in inches; `pixels_per_inch` is the coarse grid's.
"""

from dataclasses import dataclass

import cv2
import numpy as np

PHONE_SIZE_INCHES = (2.9, 6.1)
PALM_HALF_SIZE_INCHES = (1.9, 1.7)      # half-length (wrist to knuckles), half-width
FINGER_LENGTHS_INCHES = (2.7, 3.1, 2.9, 2.3)   # index, middle, ring, little
FINGER_WIDTH_INCHES = 0.75
THUMB_LENGTH_INCHES = 2.3
THUMB_WIDTH_INCHES = 0.9
WRAPPED_FINGER_REACH_INCHES = 0.55      # how far fingers curled over the phone edge stick out
FOREARM_WIDTH_INCHES = (2.4, 3.6)       # at the wrist, near the elbow
FOREARM_LENGTH_INCHES = 26.0
ARM_RISE_PER_INCH = 0.2                 # the forearm climbs toward the elbow


@dataclass(frozen=True)
class OccluderPose:
    """Where the grip sits on the sheet and which way the arm runs (toward the photographer)."""

    anchor_px: np.ndarray        # coarse-grid position of the phone or hand centre
    body_direction: np.ndarray   # unit vector from the hand toward the elbow
    height_inches: float         # the hand's height above the sheet
    holds_phone: bool


def rotated_rectangle(center: np.ndarray, along: np.ndarray, length: float, width: float) -> np.ndarray:
    """4 corners of a rectangle with its long side along `along`."""
    across = np.array([-along[1], along[0]])
    half_l, half_w = along * length / 2, across * width / 2
    return np.array([center - half_l - half_w, center + half_l - half_w, center + half_l + half_w, center - half_l + half_w])


def draw_ellipse(mask: np.ndarray, center: np.ndarray, radii: tuple[float, float], along: np.ndarray) -> None:
    """Filled ellipse whose first radius lies along `along`."""
    angle = float(np.degrees(np.arctan2(along[1], along[0])))
    cv2.ellipse(mask, (int(round(center[0])), int(round(center[1]))), (max(1, int(round(radii[0]))), max(1, int(round(radii[1])))),
                angle, 0, 360, 1, thickness=-1, lineType=cv2.LINE_AA)


def draw_capsule(mask: np.ndarray, start: np.ndarray, end: np.ndarray, width_px: float) -> None:
    """A stroke with round caps: a finger, thumb or wrist."""
    cv2.line(mask, (int(round(start[0])), int(round(start[1]))), (int(round(end[0])), int(round(end[1]))), 1,
             thickness=max(1, int(round(width_px))), lineType=cv2.LINE_AA)


def rotate_vector(vector: np.ndarray, radians: float) -> np.ndarray:
    """Rotate a 2D vector counter-clockwise (in the plane's x-right, y-down frame, clockwise on screen)."""
    return np.array([np.cos(radians) * vector[0] - np.sin(radians) * vector[1], np.sin(radians) * vector[0] + np.cos(radians) * vector[1]])


def draw_open_hand(mask: np.ndarray, palm_center: np.ndarray, reach: np.ndarray, ppi: float, rng: np.random.Generator) -> None:
    """Palm plus four near-parallel fingers pointing along `reach` and a thumb splayed to one side."""
    across = np.array([-reach[1], reach[0]])
    draw_ellipse(mask, palm_center, (PALM_HALF_SIZE_INCHES[0] * ppi, PALM_HALF_SIZE_INCHES[1] * ppi), reach)
    spread = rng.uniform(0.02, 0.1)
    curl = rng.uniform(0.6, 1.0)   # fingers bent down shorten their shadow
    for index, length in enumerate(FINGER_LENGTHS_INCHES):
        offset = (index - 1.5) * 0.8 * FINGER_WIDTH_INCHES * ppi
        direction = rotate_vector(reach, (index - 1.5) * spread)
        base = palm_center + reach * (PALM_HALF_SIZE_INCHES[0] - 0.4) * ppi + across * offset
        draw_capsule(mask, base, base + direction * length * curl * ppi, FINGER_WIDTH_INCHES * ppi)
    thumb_side = rng.choice([-1, 1])
    thumb_base = palm_center + across * thumb_side * PALM_HALF_SIZE_INCHES[1] * 0.8 * ppi - reach * 0.4 * ppi
    thumb_direction = rotate_vector(reach, thumb_side * rng.uniform(0.5, 0.9))
    draw_capsule(mask, thumb_base, thumb_base + thumb_direction * THUMB_LENGTH_INCHES * ppi, THUMB_WIDTH_INCHES * ppi)


def draw_occluder(shape: tuple[int, int], pose: OccluderPose, pixels_per_inch: float,
                  rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """(coverage mask float32 [0, 1], height map in inches) of the phone/hand/forearm silhouette."""
    mask = np.zeros(shape, np.uint8)
    body = pose.body_direction / np.linalg.norm(pose.body_direction)
    across = np.array([-body[1], body[0]])
    anchor = pose.anchor_px
    ppi = pixels_per_inch
    if pose.holds_phone:
        grip_along = rotate_vector(body, rng.uniform(-0.35, 0.35))
        grip_across = np.array([-grip_along[1], grip_along[0]])
        phone = rotated_rectangle(anchor, grip_along, PHONE_SIZE_INCHES[1] * ppi, PHONE_SIZE_INCHES[0] * ppi)
        cv2.fillPoly(mask, [np.round(phone).astype(np.int32)], 1, lineType=cv2.LINE_AA)
        palm_center = anchor + grip_along * (PHONE_SIZE_INCHES[1] * 0.3 * ppi)
        draw_ellipse(mask, palm_center, (PALM_HALF_SIZE_INCHES[0] * ppi, PALM_HALF_SIZE_INCHES[1] * ppi), grip_along)
        side = rng.choice([-1, 1])
        edge_offset = grip_across * side * PHONE_SIZE_INCHES[0] / 2 * ppi
        for index in range(int(rng.integers(2, 5))):  # finger tips curled over one long edge
            along_offset = grip_along * (0.6 - index * FINGER_WIDTH_INCHES * 0.95) * ppi
            tip = anchor + along_offset + edge_offset + grip_across * side * WRAPPED_FINGER_REACH_INCHES * ppi
            draw_capsule(mask, anchor + along_offset + edge_offset * 0.7, tip, FINGER_WIDTH_INCHES * ppi)
        thumb_base = palm_center - edge_offset
        draw_capsule(mask, thumb_base, thumb_base - grip_along * THUMB_LENGTH_INCHES * 0.7 * ppi - grip_across * side * 0.3 * ppi,
                     THUMB_WIDTH_INCHES * ppi)
        wrist = anchor + grip_along * (PHONE_SIZE_INCHES[1] * 0.3 + PALM_HALF_SIZE_INCHES[0] + 0.3) * ppi
    else:
        draw_open_hand(mask, anchor, -body, ppi, rng)
        wrist = anchor + body * (PALM_HALF_SIZE_INCHES[0] + 0.3) * ppi
    elbow = wrist + body * FOREARM_LENGTH_INCHES * ppi
    near_across, far_across = across * FOREARM_WIDTH_INCHES[0] / 2 * ppi, across * FOREARM_WIDTH_INCHES[1] / 2 * ppi
    arm = np.array([wrist - body * 0.6 * ppi - near_across, wrist - body * 0.6 * ppi + near_across, elbow + far_across, elbow - far_across])
    cv2.fillPoly(mask, [np.round(arm).astype(np.int32)], 1, lineType=cv2.LINE_AA)

    grid_y, grid_x = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
    distance_along_arm = np.clip((grid_x - wrist[0]) * body[0] + (grid_y - wrist[1]) * body[1], 0, None) / ppi
    height = pose.height_inches + ARM_RISE_PER_INCH * distance_along_arm
    return mask.astype(np.float32), height.astype(np.float32)
