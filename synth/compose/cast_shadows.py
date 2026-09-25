"""Cast shadows of the phone and the hand holding it (or a bare hand) across sheet and checks alike.

Placement, in order of preference:
1. Physical: the phone taking the photo hangs over the camera nadir at the camera height, so
   its shadow lands at nadir - height / tan(elevation) along the key's direction. Used when
   that footprint lands partly in frame.
2. Edge: otherwise the grip is placed just inside a frame edge (the photographer's side for a
   phone, any edge for a reaching hand), as if held lower and off to one side.

The shadow removes key light only (scene_irradiance.py), so it takes the ambient/fill colour.
Its penumbra is Gaussian with sigma ~ height * key angular radius / sin(elevation), widening
up the arm. Rules checked on a coarse grid, moving the grip outward until they hold (else no
shadow): it covers 2-30% of the frame, and no check has more than half its visible area in
the umbra, so a shadow never hides a whole check.
"""

import cv2
import numpy as np

from synth.compose.occluder_silhouettes import OccluderPose, draw_occluder
from synth.compose.scene_light import SceneLight

SHADOW_GRID_DOWNSCALE = 8
PHONE_SHARE = 0.65                       # of scenes with a cast shadow, how many show the phone (vs a bare hand)
HAND_HEIGHT_INCHES_RANGE = (2.5, 7.0)
SHADOW_OPACITY_RANGE = (0.8, 1.0)
PENUMBRA_SIGMA_PER_SPREAD = 0.8          # Gaussian sigma per (height * angular radius)
FRAME_SHARE_RANGE = (0.02, 0.3)
MAX_UMBRA_SHARE_OF_A_CHECK = 0.5
UMBRA_LEVEL = 0.45
EDGE_DEPTH_RANGE = (0.04, 0.28)          # how far inside the frame edge the grip sits, as a share of frame size
PLACEMENT_ATTEMPTS = 10
OUTWARD_STEP_INCHES = 1.5
AMBIENT_BLOCKED_SHARE = 0.12             # a near occluder also hides a little of the room
PENUMBRA_BLUR_LEVELS = 3


def edge_midpoints(visible_quad_plane: np.ndarray) -> dict[str, np.ndarray]:
    """Mid-points of the photo's edges in plane coordinates (quad order TL, TR, BR, BL)."""
    quad = visible_quad_plane
    return {"bottom": (quad[2] + quad[3]) / 2, "top": (quad[0] + quad[1]) / 2,
            "left": (quad[0] + quad[3]) / 2, "right": (quad[1] + quad[2]) / 2}


def penumbra_shadow(mask: np.ndarray, height_inches: np.ndarray, light: SceneLight, grid_ppi: float) -> np.ndarray:
    """Blur the silhouette with a penumbra that widens with height (blend of a few blur levels)."""
    spread_per_inch = PENUMBRA_SIGMA_PER_SPREAD * light.key_angular_radius / np.sin(light.key_elevation_radians) * grid_ppi
    heights = np.linspace(float(height_inches[mask > 0].min()) if (mask > 0).any() else 1.0,
                          float(np.percentile(height_inches[mask > 0], 90)) if (mask > 0).any() else 1.0, PENUMBRA_BLUR_LEVELS)
    blurred = [cv2.GaussianBlur(mask, (0, 0), sigmaX=max(0.6, h * spread_per_inch)) for h in heights]
    # Where the nearby occluder is higher, take the softer level (interpolate between levels).
    local_height = cv2.GaussianBlur(height_inches * mask, (0, 0), 4) / np.maximum(cv2.GaussianBlur(mask, (0, 0), 4), 1e-3)
    position = np.clip((local_height - heights[0]) / max(1e-3, heights[-1] - heights[0]) * (PENUMBRA_BLUR_LEVELS - 1),
                       0, PENUMBRA_BLUR_LEVELS - 1)
    lower = np.floor(position).astype(int).clip(0, PENUMBRA_BLUR_LEVELS - 2)
    weight = position - lower
    stack = np.stack(blurred)
    rows, cols = np.indices(mask.shape)
    return stack[lower, rows, cols] * (1 - weight) + stack[lower + 1, rows, cols] * weight


def shadow_rules_hold(shadow: np.ndarray, frame_mask: np.ndarray, owner_ids: np.ndarray) -> tuple[bool, float, int]:
    """(rules hold, frame share, direction hint): hint +1 = too much shadow (move out), -1 = too little (move in)."""
    frame_share = float((shadow > 0.5)[frame_mask].mean())
    if frame_share < FRAME_SHARE_RANGE[0]:
        return False, frame_share, -1
    if frame_share > FRAME_SHARE_RANGE[1]:
        return False, frame_share, 1
    for owner_id in np.unique(owner_ids[frame_mask]):
        if owner_id == 0:
            continue
        check_pixels = (owner_ids == owner_id) & frame_mask
        if (shadow[check_pixels] > UMBRA_LEVEL).mean() > MAX_UMBRA_SHARE_OF_A_CHECK:
            return False, frame_share, 1
    return True, frame_share, 0


def initial_pose(visible_quad_plane: np.ndarray, nadir_plane: tuple[float, float], camera_height_inches: float,
                 light: SceneLight, plane_ppi: float, holds_phone: bool, rng: np.random.Generator) -> tuple[OccluderPose, str]:
    """Grip position (plane px) and arm direction; physical for the camera phone when it lands in frame."""
    midpoints = edge_midpoints(visible_quad_plane)
    center = visible_quad_plane.mean(axis=0)
    if holds_phone:
        edge_name = "bottom" if rng.random() < 0.7 else str(rng.choice(["left", "right"]))
        key_xy = light.key_direction()[:2] / max(1e-6, np.linalg.norm(light.key_direction()[:2]))
        physical = np.asarray(nadir_plane) - camera_height_inches / np.tan(light.key_elevation_radians) * plane_ppi * key_xy
        body = midpoints[edge_name] - center
        pose = OccluderPose(physical, body / np.linalg.norm(body), camera_height_inches, True)
        return pose, "physical"
    edge_name = str(rng.choice(["bottom", "left", "right", "top"], p=[0.4, 0.25, 0.25, 0.1]))
    return edge_pose(visible_quad_plane, edge_name, rng.uniform(*HAND_HEIGHT_INCHES_RANGE), False, rng), "edge"


def edge_pose(visible_quad_plane: np.ndarray, edge_name: str, height_inches: float, holds_phone: bool,
              rng: np.random.Generator) -> OccluderPose:
    """A grip just inside the named frame edge, arm running out through it."""
    corners = {"bottom": (3, 2), "top": (0, 1), "left": (0, 3), "right": (1, 2)}[edge_name]
    along = rng.uniform(0.2, 0.8)
    edge_point = visible_quad_plane[corners[0]] * (1 - along) + visible_quad_plane[corners[1]] * along
    center = visible_quad_plane.mean(axis=0)
    body = edge_point - center
    body_unit = body / np.linalg.norm(body)
    anchor = edge_point - body * rng.uniform(*EDGE_DEPTH_RANGE) * 2
    return OccluderPose(anchor, body_unit, height_inches, holds_phone)


def cast_occluder_shadow(canvas_shape: tuple[int, int], visible_quad_plane: np.ndarray, nadir_plane: tuple[float, float],
                         camera_height_inches: float, plane_ppi: float, id_map: np.ndarray, light: SceneLight,
                         rng: np.random.Generator) -> tuple[np.ndarray | None, dict]:
    """Coarse shadow coverage [0, 1] (canvas / SHADOW_GRID_DOWNSCALE) and a label record; None when no rule-abiding placement exists."""
    holds_phone = rng.random() < PHONE_SHARE
    pose, placement = initial_pose(visible_quad_plane, nadir_plane, camera_height_inches, light, plane_ppi, holds_phone, rng)
    grid_shape = (max(4, canvas_shape[0] // SHADOW_GRID_DOWNSCALE), max(4, canvas_shape[1] // SHADOW_GRID_DOWNSCALE))
    grid_ppi = plane_ppi / SHADOW_GRID_DOWNSCALE
    frame_mask = np.zeros(grid_shape, np.uint8)
    cv2.fillPoly(frame_mask, [np.round(visible_quad_plane / SHADOW_GRID_DOWNSCALE).astype(np.int32)], 1)
    frame_mask = frame_mask > 0
    owner_ids = cv2.resize(id_map, (grid_shape[1], grid_shape[0]), interpolation=cv2.INTER_NEAREST)
    opacity = rng.uniform(*SHADOW_OPACITY_RANGE)
    for attempt in range(PLACEMENT_ATTEMPTS):
        grid_pose = OccluderPose(pose.anchor_px / SHADOW_GRID_DOWNSCALE, pose.body_direction, pose.height_inches, pose.holds_phone)
        mask, height = draw_occluder(grid_shape, grid_pose, grid_ppi, rng)
        shadow = penumbra_shadow(mask, height, light, grid_ppi) * opacity
        rules_hold, frame_share, hint = shadow_rules_hold(shadow, frame_mask, owner_ids)
        if rules_hold:
            record = {"kind": "phone_and_hand" if pose.holds_phone else "hand", "placement": placement,
                      "anchor_plane": [round(float(v), 1) for v in pose.anchor_px], "height_inches": round(pose.height_inches, 2),
                      "opacity": round(float(opacity), 3), "frame_share": round(frame_share, 3)}
            return shadow.astype(np.float32), record
        if placement == "physical" and (hint < 0 or attempt >= 2):
            # The physical footprint misses the frame (or swamps it): hold the phone lower, near an edge.
            pose = edge_pose(visible_quad_plane, "bottom" if rng.random() < 0.7 else str(rng.choice(["left", "right"])),
                             rng.uniform(*HAND_HEIGHT_INCHES_RANGE), pose.holds_phone, rng)
            placement = "edge"
            continue
        pose = OccluderPose(pose.anchor_px + hint * pose.body_direction * OUTWARD_STEP_INCHES * plane_ppi,
                            pose.body_direction, pose.height_inches, pose.holds_phone)
    return None, {}


def apply_cast_shadow(key_factor: np.ndarray, ambient_factor: np.ndarray, coarse_shadow: np.ndarray) -> None:
    """Multiply the upsampled shadow into the key buffer (and a little into ambient), in place."""
    height, width = key_factor.shape
    shadow = cv2.resize(coarse_shadow, (width, height), interpolation=cv2.INTER_LINEAR)
    key_factor *= (1 - shadow).astype(key_factor.dtype)
    ambient_factor *= (1 - AMBIENT_BLOCKED_SHARE * shadow).astype(ambient_factor.dtype)
