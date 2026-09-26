"""Knobs for scene composition, in a leaf module so the coordinator and its helpers share one type."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SceneConfig:
    """Defaults aim at typical phone photos of checks laid out on a sheet."""

    photo_long_side_range: tuple[int, int] = (2000, 3200)
    photo_aspect: float = 4 / 3
    loose_probability: float = 0.3        # loose overlap or fanned row instead of a gapped grid
    deformation_strength: float = 1.0     # scales curl / fold / wave frequencies; 0 keeps paper flat
    cast_shadow_probability: float = 0.35  # a phone or hand shadow falls across the scene
    glare_probability: float = 0.2        # a broad specular sheen on the paper
    lens_blur_probability: float = 0.15   # defocus; exclusive with motion blur
    motion_blur_probability: float = 0.07
    sharpen_probability: float = 0.9
    jpeg_quality_range: tuple[int, int] = (68, 92)
    harmonize: bool = False
    harmonize_blend: float = 0.5
