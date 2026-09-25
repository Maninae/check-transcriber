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
PALM_RADII_INCHES = (1.7, 2.1)
FINGER_RADII_INCHES = (0.36, 0.9)
FOREARM_WIDTH_INCHES = (2.3, 3.6)       # at the wrist, near the elbow
FOREARM_LENGTH_INCHES = 26.0
ARM_RISE_PER_INCH = 0.35                # the forearm climbs toward the elbow and shoulder


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


def draw_occluder(shape: tuple[int, int], pose: OccluderPose, pixels_per_inch: float,
                  rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """(coverage mask float32 [0, 1], height map in inches) of the phone/hand/forearm silhouette."""
    mask = np.zeros(shape, np.uint8)
    body = pose.body_direction / np.linalg.norm(pose.body_direction)
    across = np.array([-body[1], body[0]])
    anchor = pose.anchor_px
    ppi = pixels_per_inch
    if pose.holds_phone:
        grip_along = body.copy()
        tilt = rng.uniform(-0.35, 0.35)
        grip_along = np.array([np.cos(tilt) * body[0] - np.sin(tilt) * body[1], np.sin(tilt) * body[0] + np.cos(tilt) * body[1]])
        phone = rotated_rectangle(anchor, grip_along, PHONE_SIZE_INCHES[1] * ppi, PHONE_SIZE_INCHES[0] * ppi)
        cv2.fillPoly(mask, [np.round(phone).astype(np.int32)], 1, lineType=cv2.LINE_AA)
        palm_center = anchor + grip_along * (PHONE_SIZE_INCHES[1] * 0.28 * ppi)
        draw_ellipse(mask, palm_center, (PALM_RADII_INCHES[1] * ppi, PALM_RADII_INCHES[0] * ppi), grip_along)
        grip_across = np.array([-grip_along[1], grip_along[0]])
        side = rng.choice([-1, 1])
        for finger_index in range(int(rng.integers(3, 5))):  # fingers curl over one long edge
            finger_center = (anchor + grip_along * ((0.5 - finger_index * 0.75) * ppi)
                             + grip_across * side * (PHONE_SIZE_INCHES[0] / 2 + 0.25) * ppi)
            draw_ellipse(mask, finger_center, (FINGER_RADII_INCHES[0] * ppi, FINGER_RADII_INCHES[1] * ppi), grip_along)
        thumb_center = palm_center - grip_across * side * (PHONE_SIZE_INCHES[0] / 2 + 0.2) * ppi - grip_along * 0.6 * ppi
        draw_ellipse(mask, thumb_center, (0.9 * ppi, 0.42 * ppi), grip_along)
        wrist = anchor + grip_along * (PHONE_SIZE_INCHES[1] * 0.5 + 0.8) * ppi
    else:
        draw_ellipse(mask, anchor, (PALM_RADII_INCHES[1] * ppi, PALM_RADII_INCHES[0] * ppi * 1.15), body)
        spread = rng.uniform(0.15, 0.35)
        for finger_index in range(4):  # fingers reaching away from the body, slightly fanned
            angle = (finger_index - 1.5) * spread
            direction = -(np.cos(angle) * body + np.sin(angle) * across)
            finger_length = rng.uniform(1.4, 2.0) * ppi
            finger_center = anchor + direction * (PALM_RADII_INCHES[1] * ppi + finger_length * 0.4) + across * (finger_index - 1.5) * 0.15 * ppi
            draw_ellipse(mask, finger_center, (finger_length / 2, FINGER_RADII_INCHES[0] * ppi), direction)
        wrist = anchor + body * (PALM_RADII_INCHES[1] + 0.9) * ppi
    elbow = wrist + body * FOREARM_LENGTH_INCHES * ppi
    near_across, far_across = across * FOREARM_WIDTH_INCHES[0] / 2 * ppi, across * FOREARM_WIDTH_INCHES[1] / 2 * ppi
    arm = np.array([wrist - near_across, wrist + near_across, elbow + far_across, elbow - far_across])
    cv2.fillPoly(mask, [np.round(arm).astype(np.int32)], 1, lineType=cv2.LINE_AA)

    grid_y, grid_x = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
    distance_along_arm = np.clip((grid_x - wrist[0]) * body[0] + (grid_y - wrist[1]) * body[1], 0, None) / ppi
    height = pose.height_inches + ARM_RISE_PER_INCH * distance_along_arm
    return mask.astype(np.float32), height.astype(np.float32)
