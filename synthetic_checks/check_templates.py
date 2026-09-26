"""The catalog of check template designs, derived deterministically from a template index.

A template fixes everything a check-printing company would fix: the layout family, size,
paper tint, security pattern, border, inks, fonts, label wording and small layout shifts.
Fill-ins (names, amounts, handwriting) vary per check. Dataset splits are made by template
id, so templates must be stable across runs: design `i` depends only on `i` (numpy seeded
from `i`, never Python's salted `hash()`).

- Family of design `i` is `LAYOUT_FAMILY_ORDER[i % 6]`, so families stay balanced at any count.
- Geometry inside a family is jittered by the family module from the same template seed.
- Printed fonts come from `printed_font_pools` on their own rng stream (`FONT_STREAM`), one font per
  role; templates with index % 5 == 4 are the hold-out class and mostly use held-out fonts, which
  no other template uses. `template_printed_font_ids` lists a template's fonts for split overlap reports.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from synthetic_checks.check_layout import FAMILY_SIZE_KIND, CheckSizeKind, LayoutFamily
from synthetic_checks.fonts.font_registry import FontRole, font_ids_with_role
from synthetic_checks.fonts.printed_font_pools import (
    HOLDOUT_FONT_PREFERENCE, PRINTED_FONT_PROFILES_BY_ID, SHARED_PRINTED_FONT_IDS, PrintedRole, draw_printed_font,
    is_holdout_template_index,
)

DEFAULT_TEMPLATE_COUNT = 66
TEMPLATE_CATALOG_SEED = 1_000_003
FONT_STREAM = 7
MAX_PLATE_MISREGISTRATION_PX = 2.0
PATTERN_INK_FLOOR = 70
NEAR_WHITE_PATTERN_INK_RGB = (150, 158, 170)

LAYOUT_FAMILY_ORDER: list[LayoutFamily] = list(LayoutFamily)


class SecurityPatternKind(str, Enum):
    """Background texture printed on the check paper by the colour plate."""

    DIAGONAL_LINES = "diagonal_lines"
    GUILLOCHE_WAVES = "guilloche_waves"
    FAN_GUILLOCHE = "fan_guilloche"
    ROSETTE = "rosette"
    DOT_SCREEN = "dot_screen"
    MICROPRINT = "microprint"
    CROSSHATCH = "crosshatch"
    SOFT_GRADIENT = "soft_gradient"


class BorderKind(str, Enum):
    """Frame drawn around the check face."""

    NONE = "none"
    THIN = "thin"
    DOUBLE = "double"
    GUILLOCHE_BAND = "guilloche_band"
    DASHED = "dashed"
    MICROPRINT = "microprint"


class ScenicKind(str, Enum):
    """Faded illustrated background used by some personal stock (PERSONAL_NAME_ONLY)."""

    NONE = "none"
    MOUNTAINS = "mountains"
    WAVES = "waves"
    WATERCOLOR = "watercolor"
    SUNBURST = "sunburst"


# Common check-stock paper tints (light, low saturation).
PAPER_TINTS_RGB: list[tuple[int, int, int]] = [
    (226, 236, 246),  # blue
    (228, 241, 228),  # green
    (243, 234, 216),  # tan
    (246, 230, 234),  # pink
    (234, 234, 236),  # gray
    (236, 230, 245),  # lavender
    (248, 244, 222),  # yellow
    (246, 245, 240),  # near white
]

# Dark (offset "key") plate inks: labels, lines, borders.
DARK_PLATE_INKS_RGB: list[tuple[int, int, int]] = [
    (34, 34, 38),     # black
    (28, 40, 78),     # navy
    (30, 58, 46),     # bottle green
    (70, 32, 36),     # maroon
    (48, 54, 66),     # slate
]

BANK_LOGO_SHAPES = ["circle", "square", "diamond", "building", "none"]


@dataclass(frozen=True)
class TemplateDesign:
    """Everything fixed by the check stock; see module docstring. Hashable (used as a cache key)."""

    template_id: str
    template_index: int
    layout_family: LayoutFamily
    size_kind: CheckSizeKind
    paper_tint_rgb: tuple[int, int, int]
    pattern_kind: SecurityPatternKind
    pattern_rgb: tuple[int, int, int]    # colour-plate ink (multiplies the paper)
    pattern_strength: float
    border_kind: BorderKind
    dark_ink_rgb: tuple[int, int, int]
    header_font_id: str
    body_font_id: str
    label_font_id: str
    labels_uppercase: bool
    amount_box_outlined: bool
    printed_fill_font_id: str
    bank_logo_shape: str
    micr_layout: str                     # "personal" (routing, account, number) or "business" (number first)
    scenic_kind: ScenicKind
    has_pantograph: bool                 # hidden "VOID" in the dot screen
    has_security_fibres: bool            # short coloured fibres in the paper
    microprint_signature_line: bool      # signature line made of tiny repeated text
    plate_misregistration_px: tuple[float, float]  # colour plate offset against the dark plate
    variant: int                         # small wording/arrangement choice inside the family (0..2)


def darker_shade(rgb: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    """Scale an RGB color toward black by `factor` (0..1)."""
    return tuple(int(channel * factor) for channel in rgb)


def saturated_ink_for_tint(tint_rgb: tuple[int, int, int], saturation: float) -> tuple[int, int, int]:
    """A printing ink of the tint's hue: each channel's distance from white scaled up by `saturation`.

    Inks multiply the paper, so a blue security pattern on blue paper needs a genuinely blue ink,
    not a darker copy of the paper colour (that would apply the tint twice).
    """
    return tuple(int(max(PATTERN_INK_FLOOR, 255 - (255 - channel) * saturation)) for channel in tint_rgb)


def pick(rng: np.random.Generator, options: list):
    """One element of `options`, uniformly."""
    return options[int(rng.integers(len(options)))]


def choose_template_fonts(template_index: int, size_kind: CheckSizeKind) -> dict[PrintedRole, str]:
    """One printed font per role for design `template_index`, from its own rng stream.

    Hold-out-class templates pick from the held-out subset of each pool with probability
    `HOLDOUT_FONT_PREFERENCE`; every other template never sees a held-out font.
    """
    rng = np.random.default_rng([TEMPLATE_CATALOG_SEED + template_index, FONT_STREAM])
    holdout_class = is_holdout_template_index(template_index)
    fonts = {}
    for role in PrintedRole:
        use_holdout = holdout_class and rng.random() < HOLDOUT_FONT_PREFERENCE
        fonts[role] = draw_printed_font(rng, size_kind, role, use_holdout)
    return fonts


def template_printed_font_ids(template: TemplateDesign, include_shared: bool = False) -> tuple[str, ...]:
    """Sorted distinct printed font ids a template draws with (for split font-overlap reports).

    `include_shared` adds the fonts every template uses (serial, microprint), which overlap by design.
    """
    font_ids = {template.header_font_id, template.body_font_id, template.label_font_id, template.printed_fill_font_id}
    if include_shared:
        font_ids.update(SHARED_PRINTED_FONT_IDS)
    return tuple(sorted(font_ids))


def build_template_design(template_index: int) -> TemplateDesign:
    """Deterministically build design number `template_index`."""
    rng = np.random.default_rng(TEMPLATE_CATALOG_SEED + template_index)
    family = LAYOUT_FAMILY_ORDER[template_index % len(LAYOUT_FAMILY_ORDER)]
    size_kind = FAMILY_SIZE_KIND[family]
    tint = pick(rng, PAPER_TINTS_RGB)
    is_business = size_kind == CheckSizeKind.BUSINESS
    scenic_kind = pick(rng, [kind for kind in ScenicKind if kind != ScenicKind.NONE]) \
        if family == LayoutFamily.PERSONAL_NAME_ONLY and rng.random() < 0.8 else ScenicKind.NONE
    pattern_kind = pick(rng, list(SecurityPatternKind))
    if family == LayoutFamily.MONEY_ORDER:
        pattern_kind = pick(rng, [SecurityPatternKind.FAN_GUILLOCHE, SecurityPatternKind.ROSETTE, SecurityPatternKind.GUILLOCHE_WAVES])
    pattern_ink = saturated_ink_for_tint(tint, float(rng.uniform(3.0, 5.0)))
    if min(pattern_ink) > 200:  # near-white stock: use a neutral blue-gray security ink instead
        pattern_ink = NEAR_WHITE_PATTERN_INK_RGB
    fonts = choose_template_fonts(template_index, size_kind)
    misregistration_angle = rng.uniform(0, 2 * np.pi)
    misregistration_length = rng.uniform(0, MAX_PLATE_MISREGISTRATION_PX)
    return TemplateDesign(
        template_id=f"tpl_{template_index:03d}",
        template_index=template_index,
        layout_family=family,
        size_kind=size_kind,
        paper_tint_rgb=tint,
        pattern_kind=pattern_kind,
        pattern_rgb=pattern_ink,
        pattern_strength=float(rng.uniform(0.18, 0.42) if scenic_kind != ScenicKind.NONE else rng.uniform(0.25, 0.6)),
        border_kind=pick(rng, list(BorderKind)),
        dark_ink_rgb=pick(rng, DARK_PLATE_INKS_RGB),
        header_font_id=fonts[PrintedRole.HEADER],
        body_font_id=fonts[PrintedRole.BODY],
        label_font_id=fonts[PrintedRole.LABEL],
        labels_uppercase=bool(rng.random() < (0.85 if is_business else 0.6)),
        amount_box_outlined=bool(rng.random() < 0.75),
        printed_fill_font_id=fonts[PrintedRole.FILL],
        bank_logo_shape=pick(rng, BANK_LOGO_SHAPES),
        micr_layout="business" if is_business else "personal",
        scenic_kind=scenic_kind,
        has_pantograph=bool(rng.random() < 0.35),
        has_security_fibres=bool(rng.random() < 0.4),
        microprint_signature_line=bool(rng.random() < 0.5),
        plate_misregistration_px=(float(misregistration_length * np.cos(misregistration_angle)),
                                  float(misregistration_length * np.sin(misregistration_angle))),
        variant=int(rng.integers(3)),
    )


def build_template_catalog(template_count: int = DEFAULT_TEMPLATE_COUNT) -> list[TemplateDesign]:
    """All template designs, ids tpl_000 .. tpl_{n-1}."""
    registered_printed_fonts = set(font_ids_with_role(FontRole.PRINTED))
    if set(PRINTED_FONT_PROFILES_BY_ID) != registered_printed_fonts:
        raise ValueError("printed_font_pools profiles and fonts.py PRINTED registry disagree: "
                         f"{sorted(set(PRINTED_FONT_PROFILES_BY_ID) ^ registered_printed_fonts)}")
    return [build_template_design(index) for index in range(template_count)]


def template_family_by_id(catalog: list[TemplateDesign]) -> dict[str, str]:
    """Template id -> layout family (the stratum for the template split)."""
    return {template.template_id: template.layout_family.value for template in catalog}
