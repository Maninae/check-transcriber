"""Label records for one composed scene, all in final photo pixel coordinates.

The outline (the paper's edge after curl or fold) is what segmentation exports use; the 4
corners stay the physical corners for oriented boxes and keypoints.

Corner order is always the check's own top-left, top-right, bottom-right, bottom-left
(as printed), so the polygon also encodes orientation: an upside-down check has its
"top-left" corner near the bottom-right of its image footprint.
"""

from dataclasses import asdict, dataclass, field


@dataclass
class SceneFieldLabel:
    """One check field in photo coordinates."""

    field_name: str
    text: str
    handwritten: bool
    quad: list[list[float]]              # 4 corners of the field box (may leave the photo)
    bbox_clipped: list[float] | None     # [x0, y0, x1, y1] clipped to the photo, None if fully outside
    visible_fraction: float              # share of the field box inside the photo and not covered by another check


@dataclass
class SceneCheckLabel:
    """One check as it appears in the photo."""

    check_index: int                     # paste order; higher indices lie on top
    template_id: str
    size_kind: str
    corners: list[list[float]]           # the 4 physical paper corners: TL, TR, BR, BL of the check content
    outline: list[list[float]]           # deformed paper edge, N >= 4 points from TL, clockwise in the check's frame
    rotation_degrees_clockwise: float    # angle of the check's top edge, clockwise from +x
    orientation_class: int               # 0, 90, 180 or 270 (nearest quarter turn)
    fully_in_frame: bool
    in_frame_fraction: float             # polygon area inside the photo / full polygon area
    visible_fraction: float              # pixels actually showing this check / full outline area
    deformation: dict                    # curl / corner_lift / fold / waves parameters (inches); {} when flat
    fields: list[SceneFieldLabel]
    canonical: dict


@dataclass
class SceneLabel:
    """Everything known about one scene."""

    scene_id: str
    image_file: str
    image_width: int
    image_height: int
    background_id: str
    layout_mode: str
    harmonized: bool
    jpeg_quality: int
    effects: dict
    checks: list[SceneCheckLabel] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Plain-JSON form."""
        return asdict(self)
