"""Hard-set flags per field row, cached as JSONL under the reports root.

Flags (all per row):
- `handwritten`: the field was written with a pen font.
- `small_text`: 14 <= text height in the photo < 18 px (usable but small).
- `money_order`: layout_family == money_order.
- `low_contrast`: ink-contrast statistic in the bottom quartile of `ok` rows OF THE SAME FIELD in
  this split. Statistic = median luminance (paper) minus 2nd-percentile luminance (darkest ink)
  after a 3x3 Gaussian blur. Verified by eye: low values are dim, shadowed, grey-paper or blurry
  crops. Blind spot: a dark printed baseline rule inside the crop can mask faint ink.
- `too_small`: status == too_small (text under 14 px); reported separately, never in the hard set.
Slices over ok rows (what reports and `--subset` use):
- `handwritten_degraded` = handwritten AND (small_text OR low_contrast)
- `printed_degraded` = printed AND (small_text OR low_contrast OR money_order)
- `in_hard_set` = handwritten_degraded OR printed_degraded
- easy remainder = printed and none of the above
All eval handwriting fonts are held out from train by construction, so there is no separate
held-out-font flag (breakdowns carry pen_font_id).

CLI: `python -m experiments.field_reading.metrics.hard_set --split val [--workers 8]`
"""

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from experiments.field_reading.config import REPORTS_ROOT
from experiments.field_reading.data_access.field_manifest import load_field_rows

logger = logging.getLogger(__name__)

SMALL_TEXT_MIN_HEIGHT_PX = 14.0
SMALL_TEXT_MAX_HEIGHT_PX = 18.0
LOW_CONTRAST_QUANTILE = 0.25
CONTRAST_BLUR_KERNEL_SIZE = 3
PAPER_LUMINANCE_PERCENTILE = 50
INK_LUMINANCE_PERCENTILE = 2
MONEY_ORDER_LAYOUT_FAMILY = "money_order"
HARD_SET_FLAG_NAMES = ["handwritten", "small_text", "low_contrast", "money_order"]
HARD_SET_SLICE_NAMES = ["handwritten", "handwritten_degraded", "printed_degraded"]
CACHE_COLUMN_NAMES = ["row_key", "field_name", "status", "contrast_statistic", "low_contrast_threshold",
                      *HARD_SET_FLAG_NAMES, "too_small", "handwritten_degraded", "printed_degraded", "in_hard_set"]


def hard_set_cache_path(split_name: str) -> Path:
    """Default cache location for a split's hard-set flags."""
    return REPORTS_ROOT / f"hard_set_flags__split={split_name}.jsonl"


def ink_contrast_statistic(field_crop_path: str | None) -> float:
    """Paper-minus-ink luminance gap of a field crop (0-255, higher = crisper); NaN if the crop is missing."""
    if not isinstance(field_crop_path, str):
        return float("nan")
    grayscale_crop = cv2.imread(field_crop_path, cv2.IMREAD_GRAYSCALE)
    if grayscale_crop is None:
        return float("nan")
    blur_kernel = (CONTRAST_BLUR_KERNEL_SIZE, CONTRAST_BLUR_KERNEL_SIZE)
    blurred_crop = cv2.GaussianBlur(grayscale_crop.astype(np.float32), blur_kernel, 0)
    paper_luminance, ink_luminance = np.percentile(blurred_crop, [PAPER_LUMINANCE_PERCENTILE, INK_LUMINANCE_PERCENTILE])
    return float(paper_luminance - ink_luminance)


def compute_contrast_statistics(field_crop_paths: list[str | None], worker_count: int) -> list[float]:
    """ink_contrast_statistic for every path, in order, across a process pool."""
    if worker_count <= 1 or len(field_crop_paths) < 200:
        return [ink_contrast_statistic(path) for path in field_crop_paths]
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        return list(executor.map(ink_contrast_statistic, field_crop_paths, chunksize=256))


def build_hard_set_flags(field_rows: pd.DataFrame, worker_count: int = 8) -> pd.DataFrame:
    """Flags for every row of `field_rows` (one split's target-field rows, any status); see module docstring.

    The low-contrast quartile is taken over the ok rows passed in, so pass the whole split for the
    real cache (tests may pass a toy subset and get toy-relative thresholds).
    """
    flags = field_rows[["row_key", "field_name", "status"]].copy()
    flags["contrast_statistic"] = compute_contrast_statistics(field_rows.field_crop_path.tolist(), worker_count)
    ok_mask = field_rows.status.eq("ok")
    per_field_threshold = flags[ok_mask].groupby("field_name").contrast_statistic.quantile(LOW_CONTRAST_QUANTILE)
    flags["low_contrast_threshold"] = flags.field_name.map(per_field_threshold)
    flags["handwritten"] = field_rows.handwritten.fillna(False).astype(bool)
    text_height = field_rows.text_height_in_photo_px
    flags["small_text"] = text_height.ge(SMALL_TEXT_MIN_HEIGHT_PX) & text_height.lt(SMALL_TEXT_MAX_HEIGHT_PX)
    flags["money_order"] = field_rows.layout_family.eq(MONEY_ORDER_LAYOUT_FAMILY)
    flags["low_contrast"] = ok_mask & flags.contrast_statistic.lt(flags.low_contrast_threshold)
    flags["too_small"] = field_rows.status.eq("too_small")
    degraded_mask = flags.small_text | flags.low_contrast
    flags["handwritten_degraded"] = ok_mask & flags.handwritten & degraded_mask
    flags["printed_degraded"] = ok_mask & ~flags.handwritten & (degraded_mask | flags.money_order)
    flags["in_hard_set"] = flags.handwritten_degraded | flags.printed_degraded
    return flags[CACHE_COLUMN_NAMES]


def write_hard_set_flags(flags: pd.DataFrame, cache_path: Path) -> None:
    """Write the flags table as JSONL (overwrites)."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    flags.to_json(cache_path, orient="records", lines=True)
    logger.info("wrote %d hard-set flag rows to %s", len(flags), cache_path)


def load_or_build_hard_set_flags(split_name: str, cache_path: Path | None = None, worker_count: int = 8) -> pd.DataFrame:
    """Read the cached flags, building (whole split) and caching them first if the file is missing."""
    cache_path = cache_path or hard_set_cache_path(split_name)
    if cache_path.exists():
        return pd.read_json(cache_path, lines=True)
    logger.info("no hard-set cache at %s; building it for the whole %s split", cache_path, split_name)
    flags = build_hard_set_flags(load_field_rows(split_name), worker_count)
    write_hard_set_flags(flags, cache_path)
    return flags


def main() -> None:
    """CLI: (re)build a split's hard-set flag cache and log the flag rates."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--split", required=True, choices=["train", "val", "eval"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=None, help="cache path (default: reports root)")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    flags = build_hard_set_flags(load_field_rows(arguments.split), arguments.workers)
    write_hard_set_flags(flags, arguments.output or hard_set_cache_path(arguments.split))
    ok_flags = flags[flags.status.eq("ok")]
    slice_counts = {name: int(ok_flags[name].sum()) for name in [*HARD_SET_SLICE_NAMES, "in_hard_set"]}
    logger.info("ok rows %d; flag rates among ok: %s; slice sizes: %s; too_small rows %d",
                len(ok_flags), {name: round(float(ok_flags[name].mean()), 3) for name in HARD_SET_FLAG_NAMES},
                slice_counts, int(flags.too_small.sum()))


if __name__ == "__main__":
    main()
