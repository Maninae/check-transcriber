"""How a check's paper departs from flat: a height field over the check, in inches.

The paper is described by its height z(s, t) above the sheet at each check point (s, t),
measured in inches from the check's own top-left. Components, summed:
- curl: one edge lifts, rising quadratically from where the paper leaves the sheet.
- corner lift: a dog-ear corner rises along a diagonal hinge line.
- fold: the check was folded in half or in thirds for an envelope and opened again. Panels
  are nearly flat planes joined at smoothed creases (a ridge tent or a valley).
- waves: gentle low-frequency waviness.

Curl and fold are 1D profiles along one check axis, so their arc length can be preserved:
a tilted panel's shadow on the sheet is slightly shorter than the panel (`arc_compression`).
Corner lift and waves are too gentle for that to matter and are not compressed.

All functions are vectorised over numpy arrays and pure, so the same numbers drive the image
warp and the labels (check_plane_map.py).
"""

from dataclasses import asdict, dataclass, field

import numpy as np

CURL_PROBABILITY = 0.35
CORNER_LIFT_PROBABILITY = 0.12
FOLD_PROBABILITY = 0.22
FOLD_THIRDS_SHARE = 0.35
FOLD_ACROSS_LONG_AXIS_SHARE = 0.85
WAVE_PROBABILITY = 0.45
CURL_LENGTH_INCHES_RANGE = (0.5, 2.0)
CURL_LIFT_INCHES_RANGE = (0.06, 0.35)
CURL_MAX_SLOPE = 0.7
CORNER_LIFT_REACH_INCHES_RANGE = (0.6, 1.6)
CORNER_LIFT_INCHES_RANGE = (0.08, 0.3)
FOLD_LIFT_INCHES_RANGE = (0.1, 0.4)
CREASE_ROUNDING_INCHES_RANGE = (0.01, 0.03)
WAVE_AMPLITUDE_INCHES_RANGE = (0.01, 0.04)
WAVE_LENGTH_INCHES_RANGE = (2.0, 6.0)
PROFILE_SAMPLES_PER_INCH = 200
GRADIENT_STEP_INCHES = 1e-3
CREASE_LINE_HALF_WIDTH_INCHES = 0.012
CREASE_LINE_STRENGTH_RANGE = (0.06, 0.16)


def smooth_hinge(x: np.ndarray, rounding: float) -> np.ndarray:
    """max(0, x) with its kink rounded over about `rounding`; exactly 0 at x=0 and slope 1 far right."""
    return (x + np.sqrt(x * x + rounding * rounding) - rounding) / 2


@dataclass
class PaperDeformation:
    """Height field of one check's paper. Axis 0 is s (along the check's width), axis 1 is t (height)."""

    width_inches: float
    height_inches: float
    curl: dict | None = None         # {"edge": left|right|top|bottom, "length": in, "lift": in}
    corner_lift: dict | None = None  # {"corner": 0..3 (TL, TR, BR, BL), "reach": in, "lift": in}
    fold: dict | None = None         # {"axis": 0|1, "knots": [pos in], "heights": [in], "rounding": in, "crease_line": k}
    waves: list[dict] = field(default_factory=list)  # [{"amplitude", "wavelength", "angle", "phase"}]

    @property
    def is_flat(self) -> bool:
        """True when nothing lifts the paper."""
        return self.curl is None and self.corner_lift is None and self.fold is None and not self.waves

    def to_dict(self) -> dict:
        """Plain-JSON record for the scene label."""
        return {key: value for key, value in asdict(self).items() if value not in (None, [])}

    def axis_profile(self, coordinate: np.ndarray, axis: int) -> np.ndarray:
        """Height from the 1D components (curl, fold) that vary along `axis` only."""
        height = np.zeros_like(coordinate, dtype=np.float64)
        length = self.width_inches if axis == 0 else self.height_inches
        if self.curl is not None and (self.curl["edge"] in ("left", "right")) == (axis == 0):
            distance = coordinate if self.curl["edge"] in ("left", "top") else length - coordinate
            height += self.curl["lift"] * np.clip(1 - distance / self.curl["length"], 0, None) ** 2
        if self.fold is not None and self.fold["axis"] == axis:
            knots, heights = np.array(self.fold["knots"]), np.array(self.fold["heights"])
            slopes = np.diff(heights) / np.diff(knots)
            height += heights[0] + slopes[0] * (coordinate - knots[0])
            for knot, slope_change in zip(knots[1:-1], np.diff(slopes)):
                # A convex kink (slope rises) is a valley; rounding keeps it smooth but exact at knots.
                hinge = smooth_hinge(coordinate - knot, self.fold["rounding"])
                height += slope_change * (hinge - smooth_hinge(knots[0] - knot, self.fold["rounding"]))
        return height

    def height(self, s: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Paper height above the sheet (inches) at check points (s, t); clamped to the check rectangle."""
        s = np.clip(s, 0, self.width_inches)
        t = np.clip(t, 0, self.height_inches)
        height = self.axis_profile(s, 0) + self.axis_profile(t, 1)
        if self.corner_lift is not None:
            corner = self.corner_lift["corner"]
            distance_s = s if corner in (0, 3) else self.width_inches - s
            distance_t = t if corner in (0, 1) else self.height_inches - t
            hinge_distance = (distance_s + distance_t) / np.sqrt(2)
            height += self.corner_lift["lift"] * np.clip(1 - hinge_distance / self.corner_lift["reach"], 0, None) ** 2
        for wave in self.waves:
            projection = s * np.cos(wave["angle"]) + t * np.sin(wave["angle"])
            height += wave["amplitude"] * (0.5 + 0.5 * np.sin(2 * np.pi * projection / wave["wavelength"] + wave["phase"]))
        return height

    def height_gradient(self, s: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(dz/ds, dz/dt) by central differences (unitless slopes)."""
        step = GRADIENT_STEP_INCHES
        slope_s = (self.height(s + step, t) - self.height(s - step, t)) / (2 * step)
        slope_t = (self.height(s, t + step) - self.height(s, t - step)) / (2 * step)
        return slope_s, slope_t

    def arc_compression(self, s: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """In-plane shift (inches) of each point because tilted paper covers less sheet than its length.

        The lowest point of each 1D profile stays put; the rest slides toward it by the lost length.
        """
        shifts = []
        for coordinate, axis, length in ((s, 0, self.width_inches), (t, 1, self.height_inches)):
            samples = np.linspace(0, length, max(2, int(length * PROFILE_SAMPLES_PER_INCH)))
            profile = self.axis_profile(samples, axis)
            if not profile.any():
                shifts.append(np.zeros_like(coordinate, dtype=np.float64))
                continue
            shrink = np.cos(np.arctan(np.gradient(profile, samples))) - 1
            cumulative = np.concatenate([[0.0], np.cumsum((shrink[1:] + shrink[:-1]) / 2 * np.diff(samples))])
            cumulative -= cumulative[int(np.argmin(profile))]
            clamped = np.clip(coordinate, 0, length)
            shifts.append(np.interp(clamped, samples, cumulative))
        return shifts[0], shifts[1]

    def crease_line_shading(self, s: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Multiplicative darkening along fold creases (broken fibres catch less light)."""
        shading = np.ones_like(s, dtype=np.float64)
        if self.fold is None:
            return shading
        coordinate = s if self.fold["axis"] == 0 else t
        for knot in self.fold["knots"][1:-1]:
            shading -= self.fold["crease_line"] * np.exp(-(((coordinate - knot) / CREASE_LINE_HALF_WIDTH_INCHES) ** 2))
        return shading


def sample_fold(width_inches: float, height_inches: float, rng: np.random.Generator) -> dict:
    """Half or thirds fold, opened again: panel heights at the knots (edges and creases), one knot on the sheet."""
    axis = 0 if (rng.random() < FOLD_ACROSS_LONG_AXIS_SHARE) == (width_inches >= height_inches) else 1
    length = width_inches if axis == 0 else height_inches
    thirds = rng.random() < FOLD_THIRDS_SHARE
    creases = [length / 3, 2 * length / 3] if thirds else [length / 2]
    creases = [c + rng.uniform(-0.04, 0.04) * length for c in creases]  # hand folds are never exactly centered
    knots = [0.0, *creases, length]
    lift = rng.uniform(*FOLD_LIFT_INCHES_RANGE)
    style = rng.choice(["tent", "valley", "zigzag"]) if thirds else rng.choice(["tent", "valley"])
    if style == "tent":  # creases up, outer edges on the sheet
        heights = [0.0, *[lift * rng.uniform(0.7, 1.0) for _ in creases], 0.0]
    elif style == "valley":  # creases on the sheet, outer panels lifted
        heights = [lift * rng.uniform(0.6, 1.0), *[0.0 for _ in creases], lift * rng.uniform(0.6, 1.0)]
    else:  # a Z fold: alternating ridge and valley
        heights = [lift * rng.uniform(0.5, 1.0), 0.0, lift * rng.uniform(0.6, 1.0), 0.0]
    return {"axis": axis, "knots": [round(k, 4) for k in knots], "heights": [round(h, 4) for h in heights],
            "style": str(style), "rounding": round(rng.uniform(*CREASE_ROUNDING_INCHES_RANGE), 4),
            "crease_line": round(rng.uniform(*CREASE_LINE_STRENGTH_RANGE), 4)}


def sample_paper_deformation(width_inches: float, height_inches: float, rng: np.random.Generator,
                             strength: float = 1.0) -> PaperDeformation:
    """Random deformation for one check; `strength` scales how often each component appears (0 = flat)."""
    deformation = PaperDeformation(width_inches, height_inches)
    if rng.random() < CURL_PROBABILITY * strength:
        edge = str(rng.choice(["left", "right", "top", "bottom"], p=[0.3, 0.3, 0.2, 0.2]))
        length = rng.uniform(*CURL_LENGTH_INCHES_RANGE)
        length = min(length, 0.6 * (width_inches if edge in ("left", "right") else height_inches))
        lift = min(rng.uniform(*CURL_LIFT_INCHES_RANGE), CURL_MAX_SLOPE * length / 2)
        deformation.curl = {"edge": edge, "length": round(length, 4), "lift": round(lift, 4)}
    if rng.random() < CORNER_LIFT_PROBABILITY * strength:
        deformation.corner_lift = {"corner": int(rng.integers(4)), "reach": round(rng.uniform(*CORNER_LIFT_REACH_INCHES_RANGE), 4),
                                   "lift": round(rng.uniform(*CORNER_LIFT_INCHES_RANGE), 4)}
    if rng.random() < FOLD_PROBABILITY * strength:
        deformation.fold = sample_fold(width_inches, height_inches, rng)
    if rng.random() < WAVE_PROBABILITY * strength:
        for _ in range(int(rng.integers(1, 3))):
            deformation.waves.append({"amplitude": round(rng.uniform(*WAVE_AMPLITUDE_INCHES_RANGE), 4),
                                      "wavelength": round(rng.uniform(*WAVE_LENGTH_INCHES_RANGE), 4),
                                      "angle": round(rng.uniform(0, np.pi), 4), "phase": round(rng.uniform(0, 2 * np.pi), 4)})
    return deformation
