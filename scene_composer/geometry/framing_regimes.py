"""How close the phone is: the three framing regimes and every knob that differs between them.

- wide: the v1 behaviour, 1 to 8 checks (sometimes 12) framed at 70-95% fill. Its policy holds the
  exact constants the v1 code used, and its code path makes the same rng draws in the same order,
  so a wide scene is byte-identical to one composed before regimes existed.
- close: a manager photographs 2-6 checks (mostly 5-6) tightly: the group fills 85-98% of the frame
  from a lower camera, gaps are small or zero (edges touch), slight overlaps are common, and one
  check is cut by the frame edge more often.
- single: one check fills 70-100% of the frame at any rotation; sometimes an edge or corner is cut
  off, sometimes the photo is cropped to the check so only a thin band of surface shows, sometimes
  it is shot at a steep angle so the far edge is clearly shorter.

Labels record the regime as `effects.framing.framing_regime` only for close and single (wide labels
stay byte-identical); read it with `framing_regime_of_label`, which defaults to wide as v1 labels need.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np


class FramingRegime(str, Enum):
    """The closed set of framing regimes; the value is what configs, plans and labels store."""

    WIDE = "wide"
    CLOSE = "close"
    SINGLE = "single"


@dataclass(frozen=True)
class FramingPolicy:
    """Everything `scene_framing.frame_scene` varies by regime. The WIDE entry is the v1 constants verbatim."""

    fill_range: tuple[float, float]
    wide_shot_probability: float
    wide_fill_range: tuple[float, float]
    out_of_frame_probability: float
    in_frame_fraction_range: tuple[float, float]
    min_checks_for_out_of_frame: int
    orientation_mismatch_allowed: bool = True     # near-square groups sometimes shot the 'wrong' way round (wide only)
    corner_push_probability: float = 0.0          # push along a diagonal so a corner, not an edge, leaves the frame
    cropped_to_subject_probability: float = 0.0   # photo aspect matched to the group: only a thin band of surface
    cropped_aspect_slack_range: tuple[float, float] = (1.03, 1.15)
    steep_angle_probability: float = 0.0          # camera tilted hard: far edge clearly shorter
    steep_tilt_range: tuple[float, float] = (0.45, 0.8)


FRAMING_POLICIES: dict[FramingRegime, FramingPolicy] = {
    FramingRegime.WIDE: FramingPolicy(
        fill_range=(0.7, 0.95), wide_shot_probability=0.08, wide_fill_range=(0.55, 0.7),
        out_of_frame_probability=0.18, in_frame_fraction_range=(0.5, 0.8), min_checks_for_out_of_frame=3),
    FramingRegime.CLOSE: FramingPolicy(
        fill_range=(0.85, 0.98), wide_shot_probability=0.0, wide_fill_range=(0.85, 0.98),
        out_of_frame_probability=0.4, in_frame_fraction_range=(0.55, 0.9), min_checks_for_out_of_frame=2,
        orientation_mismatch_allowed=False, steep_angle_probability=0.1, steep_tilt_range=(0.35, 0.55)),
    FramingRegime.SINGLE: FramingPolicy(
        fill_range=(0.7, 1.0), wide_shot_probability=0.0, wide_fill_range=(0.7, 1.0),
        out_of_frame_probability=0.3, in_frame_fraction_range=(0.72, 0.95), min_checks_for_out_of_frame=1,
        orientation_mismatch_allowed=False, corner_push_probability=0.5, cropped_to_subject_probability=0.25, steep_angle_probability=0.3),
}

# close: mostly 5-6 checks, never more than 6 (a phone photo of 12 at this distance is impossible)
CLOSE_CHECK_COUNT_WEIGHTS = {
    2: 0.06,
    3: 0.10,
    4: 0.16,
    5: 0.34,
    6: 0.34,
}


def parse_framing_regime(value: str | FramingRegime) -> FramingRegime:
    """'close' -> FramingRegime.CLOSE; fails loud on a typo."""
    try:
        return FramingRegime(value)
    except ValueError:
        raise ValueError(f"unknown framing regime {value!r}; choose from {[regime.value for regime in FramingRegime]}") from None


def sample_close_check_count(rng: np.random.Generator) -> int:
    """2 to 6 checks, mode at 5-6."""
    counts = np.array(list(CLOSE_CHECK_COUNT_WEIGHTS))
    weights = np.array(list(CLOSE_CHECK_COUNT_WEIGHTS.values()))
    return int(rng.choice(counts, p=weights / weights.sum()))


def framing_regime_of_label(label_record: dict) -> str:
    """The regime a scene label was composed under; labels without the key (wide, and all of v1) are wide."""
    return label_record.get("effects", {}).get("framing", {}).get("framing_regime", FramingRegime.WIDE.value)
