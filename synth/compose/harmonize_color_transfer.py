"""Apply a harmonization network's recolouring to paper without letting white paper go grey.

PCT-Net (harmonize.py) learns to match a pasted object's colour statistics to its
surroundings. For checks on a bedsheet that also means pulling the paper's brightness toward
the fabric's, which darkens white paper; real paper under the same light stays the brightest
thing in the frame. So the network's output is split into luminance and chroma (YCrCb):
- chroma: take the network's shift (times `blend`), capped in magnitude, so paper picks up
  the room's cast but stays recognisably white;
- luminance: keep the original, moving at most `max_luminance_drop` down (or
  `max_luminance_rise` up), so paper never darkens toward the sheet.
No torch here, so the rule is testable without the network.
"""

import cv2
import numpy as np

MAX_LUMINANCE_DROP = 0.02
MAX_LUMINANCE_RISE = 0.03
MAX_CHROMA_SHIFT = 0.035      # YCrCb chroma units ([0, 1] scale); a warm or cool tint, never a colour change


def luminance_preserving_transfer(original: np.ndarray, harmonized: np.ndarray, blend: float,
                                  max_luminance_drop: float = MAX_LUMINANCE_DROP,
                                  max_luminance_rise: float = MAX_LUMINANCE_RISE,
                                  max_chroma_shift: float = MAX_CHROMA_SHIFT) -> np.ndarray:
    """Original luminance (within small bounds) with the network's capped chroma shift; float32 RGB [0, 1]."""
    original_ycrcb = cv2.cvtColor(np.clip(original, 0, 1).astype(np.float32), cv2.COLOR_RGB2YCrCb)
    harmonized_ycrcb = cv2.cvtColor(np.clip(harmonized, 0, 1).astype(np.float32), cv2.COLOR_RGB2YCrCb)
    delta = (harmonized_ycrcb - original_ycrcb) * blend
    delta[..., 0] = np.clip(delta[..., 0], -max_luminance_drop, max_luminance_rise)
    chroma_magnitude = np.linalg.norm(delta[..., 1:], axis=-1, keepdims=True)
    delta[..., 1:] *= np.minimum(1.0, max_chroma_shift / np.maximum(chroma_magnitude, 1e-6))
    return np.clip(cv2.cvtColor(original_ycrcb + delta, cv2.COLOR_YCrCb2RGB), 0, 1)
