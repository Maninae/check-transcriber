"""Labels for every check in a scene, in final photo pixels, computed by moving points.

Each label point goes check pixels -> plane (CheckPlaneMap, deformation included) -> photo
(homography, lens distortion). Nothing is re-detected from pixels.

- corners: the 4 physical paper corners (check's own TL, TR, BR, BL).
- outline (contract C3): the deformed paper edge, OUTLINE_SAMPLES_PER_EDGE points per edge
  plus the crease ends, starting at TL and running clockwise in the check's own frame.
- field quads: 4 points in the field's own TL, TR, BR, BL order. A curled or folded field's
  edges bow, so the quad is widened just enough to enclose its densely sampled edge
  (see `enclosing_field_quad`); on flat paper it equals the projected box exactly.
"""

import cv2
import numpy as np

from synth.compose.check_paste import check_boundary_points
from synth.compose.check_plane_map import CheckPlaneMap
from synth.compose.perspective import CameraView, apply_homography, clip_polygon_to_rect, plane_points_to_photo, polygon_area
from synth.compose.scene_label import SceneCheckLabel, SceneFieldLabel
from synth.render.check_fields import CheckLabel

OUTLINE_SAMPLES_PER_EDGE = 12
FIELD_EDGE_SAMPLES = 12
UNIT_SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


def box_corners(box: tuple[float, float, float, float]) -> np.ndarray:
    """(x0, y0, x1, y1) -> 4 corners TL, TR, BR, BL."""
    x0, y0, x1, y1 = box
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float64)


def check_points_to_photo(points_px: np.ndarray, plane_map: CheckPlaneMap, view: CameraView) -> np.ndarray:
    """Check pixel points all the way to photo pixels."""
    return plane_points_to_photo(plane_map.check_to_plane(points_px), view)


def outline_check_points(plane_map: CheckPlaneMap) -> np.ndarray:
    """Outline sample points in check pixels: even samples plus every crease end, in clockwise order."""
    width, height = plane_map.check_width_px, plane_map.check_height_px
    points = check_boundary_points(width, height, OUTLINE_SAMPLES_PER_EDGE)
    fold = plane_map.deformation.fold
    if fold is None:
        return points
    creases_px = [knot * plane_map.dpi for knot in fold["knots"][1:-1]]
    extra = ([(c, 0.0) for c in creases_px] + [(c, height) for c in creases_px]) if fold["axis"] == 0 else \
        ([(width, c) for c in creases_px] + [(0.0, c) for c in creases_px])
    combined = np.vstack([points, np.array(extra)])
    return combined[np.argsort(clockwise_perimeter_position(combined, width, height), kind="stable")]


def clockwise_perimeter_position(points: np.ndarray, width: float, height: float) -> np.ndarray:
    """Distance along the perimeter from TL, clockwise, for points lying on the check's edge."""
    x, y = points[:, 0], points[:, 1]
    position = np.where(np.isclose(y, 0) & (x < width), x, 0.0)
    position = np.where(np.isclose(x, width) & (y < height), width + y, position)
    position = np.where(np.isclose(y, height) & (x > 0), width + height + (width - x), position)
    position = np.where(np.isclose(x, 0) & (y > 0), 2 * width + height + (height - y), position)
    return position


def enclosing_field_quad(box: tuple[int, int, int, int], plane_map: CheckPlaneMap, view: CameraView) -> np.ndarray:
    """Smallest widening of the projected field box whose quad contains the field's bowed edges."""
    corners_photo = check_points_to_photo(box_corners(box), plane_map, view)
    x0, y0, x1, y1 = box
    edge_points = check_boundary_points(1, 1, FIELD_EDGE_SAMPLES) * [x1 - x0, y1 - y0] + [x0, y0]
    edge_photo = check_points_to_photo(edge_points, plane_map, view)
    to_photo = cv2.getPerspectiveTransform(UNIT_SQUARE.astype(np.float32), corners_photo.astype(np.float32)).astype(np.float64)
    try:
        in_unit_square = apply_homography(edge_photo, np.linalg.inv(to_photo))
    except np.linalg.LinAlgError:
        return corners_photo
    low = np.minimum(in_unit_square.min(axis=0), 0.0)
    high = np.maximum(in_unit_square.max(axis=0), 1.0)
    return apply_homography(box_corners((low[0], low[1], high[0], high[1])), to_photo)


def owned_fraction_of_polygon(id_map: np.ndarray, polygon: np.ndarray, owner_id: int) -> float:
    """Share of `polygon`'s full area whose photo pixels belong to `owner_id`."""
    full_area = polygon_area(polygon)
    if full_area < 1:
        return 0.0
    height, width = id_map.shape
    x0, y0 = np.floor(np.clip(polygon.min(axis=0), 0, [width, height])).astype(int)
    x1, y1 = np.ceil(np.clip(polygon.max(axis=0), 0, [width, height])).astype(int)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(mask, [np.round(polygon - [x0, y0]).astype(np.int32)], 1)
    owned = np.count_nonzero(mask & (id_map[y0:y1, x0:x1] == owner_id))
    return float(min(1.0, owned / full_area))


def build_check_labels(check_labels: list[CheckLabel], plane_maps: list[CheckPlaneMap], view: CameraView,
                       photo_ids: np.ndarray) -> list[SceneCheckLabel]:
    """Project every check, its outline and its fields to the photo; measure framing and occlusion."""
    labels = []
    for index, (check_label, plane_map) in enumerate(zip(check_labels, plane_maps)):
        corners = check_points_to_photo(box_corners((0, 0, check_label.width_px, check_label.height_px)), plane_map, view)
        outline = check_points_to_photo(outline_check_points(plane_map), plane_map, view)
        full_area = polygon_area(outline)
        in_frame_fraction = polygon_area(clip_polygon_to_rect(outline, view.photo_width, view.photo_height)) / full_area if full_area else 0.0
        top_edge = corners[1] - corners[0]
        rotation_clockwise = float(np.degrees(np.arctan2(top_edge[1], top_edge[0])) % 360)
        fields = []
        for field_label in check_label.fields:
            quad = enclosing_field_quad(field_label.box, plane_map, view)
            clipped_quad = clip_polygon_to_rect(quad, view.photo_width, view.photo_height)
            bbox = None
            if len(clipped_quad) >= 3 and polygon_area(clipped_quad) > 0:
                bbox = [round(float(v), 2) for v in (*clipped_quad.min(axis=0), *clipped_quad.max(axis=0))]
            fields.append(SceneFieldLabel(field_label.field_name, field_label.text, field_label.handwritten,
                                          quad.round(2).tolist(), bbox,
                                          round(owned_fraction_of_polygon(photo_ids, quad, index + 1), 4)))
        labels.append(SceneCheckLabel(
            check_index=index, template_id=check_label.template_id, size_kind=check_label.size_kind,
            corners=corners.round(2).tolist(), outline=outline.round(2).tolist(),
            rotation_degrees_clockwise=round(rotation_clockwise, 2),
            orientation_class=int(round(rotation_clockwise / 90) % 4 * 90),
            fully_in_frame=bool(in_frame_fraction > 0.999), in_frame_fraction=round(float(in_frame_fraction), 4),
            visible_fraction=round(owned_fraction_of_polygon(photo_ids, outline, index + 1), 4),
            deformation=plane_map.deformation.to_dict(), fields=fields, canonical=check_label.canonical))
    return labels
