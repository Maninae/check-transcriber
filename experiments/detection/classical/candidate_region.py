"""The `CandidateRegion` record shared by the mask extraction and cell-merging modules."""

from dataclasses import dataclass

import numpy as np


@dataclass
class CandidateRegion:
    """One connected region that may be a check."""

    contour: np.ndarray  # (N, 2) int32 external contour, working pixels
    region_area: float  # pixel count of the region
    source_name: str  # which mask produced it, for diagnostics and tuning
