"""Frame the photo around the laid-out checks, the way a person photographs them.

Order of decisions (the camera is fitted to the checks, never the checks to the camera):
1. Photo orientation follows the group's shape (portrait for a tall group), rarely the other way.
2. A hand-held view is sampled (perspective.sample_camera_view): tilt, roll, lens distortion.
3. The sheet's scale (plane pixels per inch) and offset are solved so the group's projected
   bounding box spans FILL_RANGE of the frame along its tighter dimension, centered with a
   small hand-held offset. About 8% of scenes are wider shots.
4. Optionally one edge check is slid outward until only part of it stays in frame.
"""

from dataclasses import dataclass

import numpy as np

from scene_composer.geometry.perspective import CameraView, apply_homography, clip_polygon_to_rect, polygon_area, sample_camera_view
from scene_composer.geometry.placement import CheckPlacement, placement_corners_inches

FILL_RANGE = (0.7, 0.95)
WIDE_SHOT_PROBABILITY = 0.08
WIDE_FILL_RANGE = (0.55, 0.7)   # below ~0.55 a 12-check group shrinks to unreadable
ORIENTATION_MISMATCH_PROBABILITY = 0.1
MISMATCH_ASPECT_RANGE = (0.6, 1.7)   # only near-square groups get shot the 'wrong' way round
MIN_CHECKS_FOR_OUT_OF_FRAME = 3
CENTER_OFFSET_FRACTION = 0.03
FIT_ITERATIONS = 8
OUT_OF_FRAME_PROBABILITY = 0.18
IN_FRAME_FRACTION_RANGE = (0.5, 0.8)
PUSH_SEARCH_STEPS = 30
CAMERA_FIELD_OF_VIEW_DEGREES = 65.0


@dataclass
class SceneFraming:
    """The camera, the sheet's scale and origin, and what the framing aimed for."""

    view: CameraView
    plane_pixels_per_inch: float
    origin_plane: np.ndarray        # plane pixel position of sheet inch (0, 0)
    nadir_plane: tuple[float, float]
    camera_height_plane_px: float
    fill_target: float
    wide_shot: bool
    pushed_out_check_index: int | None = None

    def inches_to_plane(self, points_inches: np.ndarray) -> np.ndarray:
        """Sheet inches -> plane pixels."""
        return self.origin_plane + np.asarray(points_inches) * self.plane_pixels_per_inch

    def to_dict(self) -> dict:
        """Plain-JSON summary for the scene label's effects."""
        return {"fill_target": round(self.fill_target, 3), "wide_shot": self.wide_shot,
                "plane_pixels_per_inch": round(self.plane_pixels_per_inch, 2),
                "camera_height_inches": round(self.camera_height_plane_px / self.plane_pixels_per_inch, 2),
                "pushed_out_check_index": self.pushed_out_check_index}


def group_corners_inches(sizes_inches: list[tuple[float, float]], placements: list[CheckPlacement]) -> np.ndarray:
    """All flat check corners of the group, (4N, 2) inches."""
    return np.vstack([placement_corners_inches(size, placement) for size, placement in zip(sizes_inches, placements)])


def choose_photo_size(group_corners: np.ndarray, long_side_range: tuple[int, int], photo_aspect: float,
                      rng: np.random.Generator) -> tuple[int, int]:
    """Photo (width, height): landscape for a wide group, portrait for a tall one, rarely mismatched."""
    extent = group_corners.max(axis=0) - group_corners.min(axis=0)
    portrait = extent[1] > extent[0]
    aspect = extent[0] / extent[1]
    if MISMATCH_ASPECT_RANGE[0] < aspect < MISMATCH_ASPECT_RANGE[1] and rng.random() < ORIENTATION_MISMATCH_PROBABILITY:
        portrait = not portrait
    long_side = int(rng.integers(*long_side_range))
    short_side = int(round(long_side / photo_aspect))
    return (short_side, long_side) if portrait else (long_side, short_side)


def projected_fill(points_plane: np.ndarray, view: CameraView) -> tuple[float, np.ndarray]:
    """(fraction of the frame the points' photo bbox spans on its tighter axis, bbox center in the photo)."""
    photo_points = apply_homography(points_plane, view.plane_to_photo)
    low, high = photo_points.min(axis=0), photo_points.max(axis=0)
    fill = float(max((high[0] - low[0]) / view.photo_width, (high[1] - low[1]) / view.photo_height))
    return fill, (low + high) / 2


def fit_group_to_view(group_corners: np.ndarray, view: CameraView, fill_target: float,
                      photo_center_target: np.ndarray) -> tuple[float, np.ndarray]:
    """Solve plane pixels per inch and origin so the group fills `fill_target` around the target center."""
    inverse = np.linalg.inv(view.plane_to_photo)
    group_center = (group_corners.min(axis=0) + group_corners.max(axis=0)) / 2
    quad = view.visible_quad_plane
    quad_extent = quad.max(axis=0) - quad.min(axis=0)
    ppi = fill_target * float(np.min(quad_extent / (group_corners.max(axis=0) - group_corners.min(axis=0))))
    anchor_plane = apply_homography(photo_center_target[None], inverse)[0]
    for _ in range(FIT_ITERATIONS):
        fill, bbox_center = projected_fill(anchor_plane + (group_corners - group_center) * ppi, view)
        ppi *= fill_target / fill
        correction = apply_homography(photo_center_target[None], inverse)[0] - apply_homography(bbox_center[None], inverse)[0]
        anchor_plane = anchor_plane + correction
    return ppi, anchor_plane - group_center * ppi


def camera_nadir_and_height(view: CameraView) -> tuple[tuple[float, float], float]:
    """Approximate nadir (plane point under the photo center) and camera height in plane pixels."""
    inverse = np.linalg.inv(view.plane_to_photo)
    center = np.array([[view.photo_width / 2, view.photo_height / 2]])
    half_long = max(view.photo_width, view.photo_height) / 2
    axis = np.array([[1.0, 0.0]]) if view.photo_width >= view.photo_height else np.array([[0.0, 1.0]])
    ends = apply_homography(np.vstack([center - axis * half_long, center + axis * half_long]), inverse)
    nadir = apply_homography(center, inverse)[0]
    height = np.linalg.norm(ends[1] - ends[0]) / 2 / np.tan(np.deg2rad(CAMERA_FIELD_OF_VIEW_DEGREES / 2))
    return (float(nadir[0]), float(nadir[1])), float(height)


def in_frame_fraction_of_check(corners_inches: np.ndarray, framing: SceneFraming) -> float:
    """Share of a flat check's projected area inside the photo."""
    photo = apply_homography(framing.inches_to_plane(corners_inches), framing.view.plane_to_photo)
    area = polygon_area(photo)
    return polygon_area(clip_polygon_to_rect(photo, framing.view.photo_width, framing.view.photo_height)) / area if area else 0.0


def push_one_check_out_of_frame(sizes_inches: list[tuple[float, float]], placements: list[CheckPlacement],
                                framing: SceneFraming, rng: np.random.Generator) -> int:
    """Slide the check nearest a random photo edge outward until IN_FRAME_FRACTION_RANGE of it remains."""
    view = framing.view
    edge_normals = {"left": (-1.0, 0.0), "right": (1.0, 0.0), "top": (0.0, -1.0), "bottom": (0.0, 1.0)}
    normal = np.array(edge_normals[str(rng.choice(list(edge_normals)))])
    centers_inches = np.array([[p.center_x_inches, p.center_y_inches] for p in placements])
    centers_photo = apply_homography(framing.inches_to_plane(centers_inches), view.plane_to_photo)
    index = int(np.argmax(centers_photo @ normal))
    inverse = np.linalg.inv(view.plane_to_photo)
    ends = apply_homography(np.vstack([centers_photo[index], centers_photo[index] + normal * 50]), inverse)
    direction_inches = (ends[1] - ends[0]) / np.linalg.norm(ends[1] - ends[0])
    target = rng.uniform(*IN_FRAME_FRACTION_RANGE)
    placement = placements[index]
    start = np.array([placement.center_x_inches, placement.center_y_inches])
    low, high = 0.0, float(sum(sizes_inches[index])) * 2
    for _ in range(PUSH_SEARCH_STEPS):  # in-frame fraction falls as the check slides out: bisect
        middle = (low + high) / 2
        placement.center_x_inches, placement.center_y_inches = start + direction_inches * middle
        if in_frame_fraction_of_check(placement_corners_inches(sizes_inches[index], placement), framing) > target:
            low = middle
        else:
            high = middle
    return index


def frame_scene(sizes_inches: list[tuple[float, float]], placements: list[CheckPlacement], long_side_range: tuple[int, int],
                photo_aspect: float, max_plane_pixels_per_inch: float, rng: np.random.Generator) -> SceneFraming:
    """Choose photo size, camera, and sheet scale for a laid-out group; may push one check partly out.

    If filling the frame would need more plane pixels per inch than the checks were rendered at,
    the photo is taken at a proportionally lower resolution instead of upsampling the paper.
    """
    corners = group_corners_inches(sizes_inches, placements)
    photo_width, photo_height = choose_photo_size(corners, long_side_range, photo_aspect, rng)
    wide_shot = bool(rng.random() < WIDE_SHOT_PROBABILITY)
    fill_target = float(rng.uniform(*(WIDE_FILL_RANGE if wide_shot else FILL_RANGE)))
    offset_unit = rng.uniform(-1, 1, 2) * min(CENTER_OFFSET_FRACTION, (1 - fill_target) / 2)
    view_seed = int(rng.integers(2**31))
    for _ in range(2):
        view = sample_camera_view(photo_width, photo_height, np.random.default_rng(view_seed))
        center_target = np.array([photo_width / 2, photo_height / 2]) * (1 + 2 * offset_unit)
        ppi, origin = fit_group_to_view(corners, view, fill_target, center_target)
        if ppi <= max_plane_pixels_per_inch:
            break
        shrink = max_plane_pixels_per_inch / ppi
        photo_width, photo_height = int(photo_width * shrink), int(photo_height * shrink)
    nadir, camera_height = camera_nadir_and_height(view)
    framing = SceneFraming(view, ppi, origin, nadir, camera_height, fill_target, wide_shot)
    if len(placements) >= MIN_CHECKS_FOR_OUT_OF_FRAME and rng.random() < OUT_OF_FRAME_PROBABILITY:
        framing.pushed_out_check_index = push_one_check_out_of_frame(sizes_inches, placements, framing, rng)
    return framing
