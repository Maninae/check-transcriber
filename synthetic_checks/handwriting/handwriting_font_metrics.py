"""Per-font measurements the handwriting engine needs: glyph coverage and physical size calibration.

- Coverage: Pillow exposes no cmap, so a character counts as missing when it renders nothing
  or renders exactly like a private-use codepoint no font defines (the font's notdef glyph).
- Calibration (fill-ins by x-height, signatures by ascender height): handwriting fonts disagree wildly about how big an em is (Reenie Beanie's
  x-height is a third of Patrick Hand's). `em_px_for_x_height` returns the em that gives a
  wanted x-height, so every font writes at the same physical size on the check.
"""

import functools

import numpy as np

from synthetic_checks.fonts.font_registry import load_font

REQUIRED_HANDWRITING_CHARACTERS = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" + "$,./-#&'()"
)
NOTDEF_PROBE_CHARACTER = "\U000F0000"  # supplementary private use plane: never defined by these fonts
COVERAGE_PROBE_EM_PX = 64
CALIBRATION_EM_PX = 200
X_HEIGHT_PROBE_CHARACTERS = "xuvwnzm"  # flat-topped lowercase letters without ascenders or descenders
DIGIT_HEIGHT_PROBE_CHARACTERS = "0123456789"
ASCENDER_HEIGHT_PROBE_CHARACTERS = "bdhklABDEHKLR"  # tall letters: what sets a signature's visual size


def glyph_mask_signature(font_id: str, character: str) -> tuple[tuple[int, int], bytes]:
    """Rendered bitmap of one character at the probe size, as (size, bytes) for comparison."""
    mask = load_font(font_id, COVERAGE_PROBE_EM_PX).getmask(character)
    return mask.size, bytes(mask)


def missing_characters(font_id: str, characters: str = REQUIRED_HANDWRITING_CHARACTERS) -> list[str]:
    """Characters in `characters` the font cannot draw (renders blank or as its notdef box)."""
    notdef_signature = glyph_mask_signature(font_id, NOTDEF_PROBE_CHARACTER)
    missing = []
    for character in characters:
        size, pixels = glyph_mask_signature(font_id, character)
        if size[0] * size[1] == 0 or not any(pixels) or (size, pixels) == notdef_signature:
            missing.append(character)
    return missing


def median_ink_height_px(font_id: str, em_px: int, characters: str) -> float:
    """Median height of the inked bounding box over `characters` at `em_px`."""
    font = load_font(font_id, em_px)
    heights = []
    for character in characters:
        x0, y0, x1, y1 = font.getbbox(character, anchor="ls")
        if y1 > y0:
            heights.append(y1 - y0)
    return float(np.median(heights)) if heights else em_px * 0.5


@functools.lru_cache(maxsize=256)
def x_height_per_em(font_id: str) -> float:
    """Font's x-height as a fraction of its em (e.g. 0.45)."""
    return median_ink_height_px(font_id, CALIBRATION_EM_PX, X_HEIGHT_PROBE_CHARACTERS) / CALIBRATION_EM_PX


@functools.lru_cache(maxsize=256)
def digit_height_per_em(font_id: str) -> float:
    """Font's digit height as a fraction of its em."""
    return median_ink_height_px(font_id, CALIBRATION_EM_PX, DIGIT_HEIGHT_PROBE_CHARACTERS) / CALIBRATION_EM_PX


@functools.lru_cache(maxsize=256)
def ascender_height_per_em(font_id: str) -> float:
    """Height of tall letters (ascenders and capitals) as a fraction of the em."""
    return median_ink_height_px(font_id, CALIBRATION_EM_PX, ASCENDER_HEIGHT_PROBE_CHARACTERS) / CALIBRATION_EM_PX


def x_height_for_ascender_height(font_id: str, ascender_height_px: float) -> float:
    """The x-height this font has when its tall letters are `ascender_height_px` tall.

    Script fonts vary most in their x-height to capital ratio (Monsieur La Doulaise: 0.2 em
    x-height under 0.8 em capitals), so signatures are sized by their tall letters.
    """
    return ascender_height_px * x_height_per_em(font_id) / ascender_height_per_em(font_id)


def em_px_for_x_height(font_id: str, x_height_px: float) -> int:
    """Em size (px) at which this font's x-height is `x_height_px`."""
    return max(4, int(round(x_height_px / x_height_per_em(font_id))))
