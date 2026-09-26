"""Near-duplicate detection for backgrounds: a two-band perceptual signature and an index over it.

Why not a 64-bit pHash: surface textures carry almost no low-frequency content, so the signs of
their low DCT coefficients are noise (a half-size JPEG re-encode flipped 10/64 bits of a fabric),
and photos are dominated by their lighting gradient, so two different photos lit alike look
similar at low frequency alone. The signature therefore has two bands and a duplicate must match
in both:
- `coarse`: 16x16 grey thumbnail, the image's layout and lighting;
- `fine`: 64x64 grey thumbnail minus its blur (border trimmed), the texture detail.
Each band is mean-centred and unit-normalized, so similarity is a normalized cross-correlation.
Measured on 30 real backgrounds: a half-size q70 copy keeps coarse >= 0.90 and fine >= 0.67;
distinct images never exceeded 0.16 in the weaker band. Flips and rotations are not matched.
"""

from dataclasses import dataclass

import cv2
import numpy as np

THUMBNAIL_SIDE = 64
COARSE_SIDE = 16
FINE_BAND_BLUR_SIGMA = 2.0
# The blur's reflected border mis-estimates curved lighting there, leaking it into the fine band.
FINE_BAND_BORDER_PX = 6
COARSE_DUPLICATE_CORRELATION = 0.85
FINE_DUPLICATE_CORRELATION = 0.5


@dataclass(frozen=True)
class PerceptualSignature:
    """Unit vectors for the coarse and fine bands of one image."""

    coarse: np.ndarray
    fine: np.ndarray


def centred_unit_vector(values: np.ndarray) -> np.ndarray:
    """Flatten, subtract the mean, scale to unit length (all-zero stays zero)."""
    vector = values.astype(np.float64).flatten()
    vector -= vector.mean()
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def perceptual_signature(image_rgb: np.ndarray) -> PerceptualSignature:
    """Two-band signature of a uint8 RGB (or grey) image of any size."""
    grey = image_rgb if image_rgb.ndim == 2 else cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    thumbnail = cv2.resize(grey.astype(np.float32), (THUMBNAIL_SIDE, THUMBNAIL_SIDE), interpolation=cv2.INTER_AREA)
    coarse = cv2.resize(thumbnail, (COARSE_SIDE, COARSE_SIDE), interpolation=cv2.INTER_AREA)
    fine = (thumbnail - cv2.GaussianBlur(thumbnail, (0, 0), FINE_BAND_BLUR_SIGMA))[
        FINE_BAND_BORDER_PX:-FINE_BAND_BORDER_PX, FINE_BAND_BORDER_PX:-FINE_BAND_BORDER_PX]
    return PerceptualSignature(centred_unit_vector(coarse), centred_unit_vector(fine))


def band_correlations(first: PerceptualSignature, second: PerceptualSignature) -> tuple[float, float]:
    """(coarse, fine) normalized cross-correlations, each in [-1, 1]."""
    return float(first.coarse @ second.coarse), float(first.fine @ second.fine)


class PerceptualDuplicateIndex:
    """Signatures of known images; answers "is this a near-duplicate of something we already have?"."""

    def __init__(self):
        self.names: list[str] = []
        self.coarse_rows: list[np.ndarray] = []
        self.fine_rows: list[np.ndarray] = []

    def add(self, signature: PerceptualSignature, name: str) -> None:
        """Remember one image under `name`."""
        self.names.append(name)
        self.coarse_rows.append(signature.coarse)
        self.fine_rows.append(signature.fine)

    def find_duplicate(self, signature: PerceptualSignature) -> str | None:
        """Name of a known image matching in both bands (the best fine-band match), or None."""
        if not self.names:
            return None
        coarse = np.array(self.coarse_rows) @ signature.coarse
        fine = np.array(self.fine_rows) @ signature.fine
        matches = np.flatnonzero((coarse >= COARSE_DUPLICATE_CORRELATION) & (fine >= FINE_DUPLICATE_CORRELATION))
        return self.names[matches[np.argmax(fine[matches])]] if matches.size else None

    def __len__(self) -> int:
        return len(self.names)
