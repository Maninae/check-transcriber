"""The catalog of check template designs, derived deterministically from a template index.

A template fixes everything a check-printing company would fix: size, paper tint,
security pattern, border, fonts, label wording, and a small layout shift. Fill-ins
(names, amounts, handwriting) vary per check. Dataset splits are made by template id,
so templates must be stable across runs: design `i` depends only on `i`.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from synth.render.check_layout import CheckLayout, CheckSizeKind
from synth.render.fonts import FontRole, font_ids_with_role

DEFAULT_TEMPLATE_COUNT = 24
TEMPLATE_CATALOG_SEED = 1_000_003
LAYOUT_JITTER_FRACTION = 0.006
MIN_DOLLAR_SIGN_GAP = 0.022
BUSINESS_TEMPLATE_PROBABILITY = 0.3


class SecurityPatternKind(str, Enum):
    """Background texture printed on the check paper."""

    DIAGONAL_LINES = "diagonal_lines"
    GUILLOCHE_WAVES = "guilloche_waves"
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

PRINTED_BODY_FONT_IDS = ["libre_baskerville", "eb_garamond", "source_sans_3", "pt_sans", "courier_prime"]
PRINTED_HEADER_FONT_IDS = ["libre_baskerville", "eb_garamond", "pt_sans_bold", "oswald", "courier_prime_bold", "source_sans_3"]


@dataclass(frozen=True)
class TemplateDesign:
    """Everything fixed by the check stock; see module docstring."""

    template_id: str
    size_kind: CheckSizeKind
    paper_tint_rgb: tuple[int, int, int]
    pattern_kind: SecurityPatternKind
    pattern_rgb: tuple[int, int, int]
    pattern_strength: float
    border_kind: BorderKind
    header_font_id: str
    body_font_id: str
    label_font_id: str
    labels_uppercase: bool
    amount_box_outlined: bool
    printed_fill_font_id: str
    layout: CheckLayout
    bank_logo_shape: str
    micr_layout: str  # "personal" (routing, account, number) or "business" (number first)


def darker_shade(rgb: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    """Scale an RGB color toward black by `factor` (0..1)."""
    return tuple(int(channel * factor) for channel in rgb)


def jitter_layout(rng: np.random.Generator) -> CheckLayout:
    """Shift each anchor group by a small random amount, keeping the standard arrangement."""
    base = CheckLayout()
    shifted_values = {}
    for name, value in base.__dict__.items():
        shifted_values[name] = value + float(rng.uniform(-LAYOUT_JITTER_FRACTION, LAYOUT_JITTER_FRACTION))
    # Keep the "$" clear of the amount box it labels.
    shifted_values["dollar_sign_x"] = min(shifted_values["dollar_sign_x"], shifted_values["amount_box_x0"] - MIN_DOLLAR_SIGN_GAP)
    return CheckLayout(**shifted_values)


def build_template_design(template_index: int) -> TemplateDesign:
    """Deterministically build design number `template_index`."""
    rng = np.random.default_rng(TEMPLATE_CATALOG_SEED + template_index)
    size_kind = CheckSizeKind.BUSINESS if rng.random() < BUSINESS_TEMPLATE_PROBABILITY else CheckSizeKind.PERSONAL
    tint = PAPER_TINTS_RGB[int(rng.integers(len(PAPER_TINTS_RGB)))]
    pattern_kind = list(SecurityPatternKind)[int(rng.integers(len(SecurityPatternKind)))]
    printed_fill_candidates = ["courier_prime", "source_sans_3", "pt_sans", "libre_baskerville"]
    return TemplateDesign(
        template_id=f"tpl_{template_index:03d}",
        size_kind=size_kind,
        paper_tint_rgb=tint,
        pattern_kind=pattern_kind,
        pattern_rgb=darker_shade(tint, float(rng.uniform(0.55, 0.8))),
        pattern_strength=float(rng.uniform(0.25, 0.6)),
        border_kind=list(BorderKind)[int(rng.integers(len(BorderKind)))],
        header_font_id=PRINTED_HEADER_FONT_IDS[int(rng.integers(len(PRINTED_HEADER_FONT_IDS)))],
        body_font_id=PRINTED_BODY_FONT_IDS[int(rng.integers(len(PRINTED_BODY_FONT_IDS)))],
        label_font_id=PRINTED_BODY_FONT_IDS[int(rng.integers(len(PRINTED_BODY_FONT_IDS)))],
        labels_uppercase=bool(rng.random() < 0.6),
        amount_box_outlined=bool(rng.random() < 0.75),
        printed_fill_font_id=printed_fill_candidates[int(rng.integers(len(printed_fill_candidates)))],
        layout=jitter_layout(rng),
        bank_logo_shape=["circle", "square", "diamond", "none"][int(rng.integers(4))],
        micr_layout="business" if size_kind == CheckSizeKind.BUSINESS else "personal",
    )


def build_template_catalog(template_count: int = DEFAULT_TEMPLATE_COUNT) -> list[TemplateDesign]:
    """All template designs, ids tpl_000 .. tpl_{n-1}."""
    assert set(PRINTED_BODY_FONT_IDS) <= set(font_ids_with_role(FontRole.PRINTED))
    return [build_template_design(index) for index in range(template_count)]
