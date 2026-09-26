"""Run the Python classical detector on dumped scenes and write reference outputs for the JS parity runner.

    <venv>/bin/python app/tests/parity/dump_classical_reference.py --scenes-dir <dir> [--debug] val_000000 ...

Run from the repo root so `experiments.` imports resolve. Reads the raw BGR bytes written by
`dump_scene_pixels.py` (so both sides see identical pixels) and writes `<output>/<scene_id>.json`:

- `detections`: corners (full precision), score, source_name, side_supports, as `detect_checks_classical` returns.
- `seconds`: wall time of the Python detector on this scene.
- `debug` (with `--debug`): per-stage intermediates, to localize a JS divergence to one stage:
  channel-map float64 sums, mask nonzero counts, region counts per mask, fitted quads,
  line segments, hypotheses, verified candidates and the selected set.
With `--dump-maps`, every channel map is also written as raw float32 (`<scene_id>__<map>.f32`).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from experiments.detection.classical.candidates.candidate_regions import build_candidate_masks, extract_regions_from_mask
from experiments.detection.classical.candidates.line_quadrilateral_hypotheses import build_line_quadrilateral_hypotheses
from experiments.detection.classical.candidates.line_segment_extraction import extract_line_segments
from experiments.detection.classical.classical_detector_config import ClassicalDetectorConfig
from experiments.detection.classical.detect_checks_classical import collect_verified_candidates, detect_checks_classical
from experiments.detection.classical.geometry.quadrilateral_fitting import fit_quadrilateral_to_contour
from experiments.detection.classical.preprocessing.working_image_channels import build_working_image_channels
from experiments.detection.classical.verification.candidate_selection import select_non_overlapping_candidates

DEFAULT_SCENES_DIRECTORY = Path("/Volumes/vega/datasets/check-transcriber/tools/app-parity/classical/scenes")
DEFAULT_OUTPUT_DIRECTORY = Path("/Volumes/vega/datasets/check-transcriber/tools/app-parity/classical/python_reference")
CHANNEL_MAP_NAMES = (
    "lightness", "chroma", "lab_a", "lab_b", "paper_score", "print_residue",
    "texture_std", "gradient_x", "gradient_y", "gradient_magnitude",
)


def read_scene_bgr(scenes_directory: Path, scene_id: str) -> np.ndarray:
    """The dumped H x W x 3 uint8 BGR scene."""
    shape_json = json.loads((scenes_directory / f"{scene_id}.json").read_text())
    pixel_bytes = (scenes_directory / f"{scene_id}.bgr").read_bytes()
    return np.frombuffer(pixel_bytes, dtype=np.uint8).reshape(shape_json["height"], shape_json["width"], 3).copy()


def collect_stage_debug(image_bgr: np.ndarray, config: ClassicalDetectorConfig, maps_prefix: Path | None) -> dict:
    """Per-stage intermediates of one scene (recomputed; the detector itself is untouched)."""
    channels = build_working_image_channels(image_bgr, config)
    debug = {
        "working_shape": list(channels.lightness.shape),
        "scale_to_full_resolution": channels.scale_to_full_resolution,
        "channel_sums": {name: float(getattr(channels, name).astype(np.float64).sum()) for name in CHANNEL_MAP_NAMES},
    }
    if maps_prefix is not None:
        for name in CHANNEL_MAP_NAMES:
            getattr(channels, name).astype(np.float32).tofile(f"{maps_prefix}__{name}.f32")
    masks = build_candidate_masks(channels, config)
    debug["mask_nonzero_counts"] = {name: int(np.count_nonzero(mask)) for name, mask in masks.items()}
    regions_per_mask, fitted_quads = {}, []
    for name, mask in masks.items():
        regions = extract_regions_from_mask(mask, name, channels, config)
        regions_per_mask[name] = len(regions)
        for region in regions:
            fitted = fit_quadrilateral_to_contour(
                region.contour, region.source_name, minimum_rectangularity=config.minimum_region_rectangularity,
                aspect_range=(min(config.minimum_aspect_ratio, config.border_truncated_aspect_range[0]),
                              max(config.maximum_aspect_ratio, config.border_truncated_aspect_range[1])),
            )
            fitted_quads.append({
                "source_name": region.source_name, "contour_length": int(len(region.contour)), "region_area": region.region_area,
                "corners": None if fitted is None else fitted.corners.tolist(),
                "rectangularity": None if fitted is None else fitted.rectangularity,
            })
    debug["regions_per_mask"] = regions_per_mask
    debug["fitted_quads"] = fitted_quads
    segments = extract_line_segments(channels, config)
    debug["line_segments"] = segments.tolist()
    hypotheses = build_line_quadrilateral_hypotheses(segments, channels.long_side_pixels, config)
    debug["line_hypotheses"] = [corners.tolist() for corners in hypotheses]
    verified = collect_verified_candidates(channels, config)
    debug["verified_candidates"] = [
        {"corners": candidate.corners.tolist(), "score": candidate.score, "source_name": candidate.source_name,
         "rectangularity": candidate.rectangularity, "side_supports": candidate.side_supports.tolist()}
        for candidate in verified
    ]
    selected = select_non_overlapping_candidates(verified, config, channels.lightness.shape)
    debug["selected_candidates"] = [{"corners": candidate.corners.tolist(), "source_name": candidate.source_name} for candidate in selected]
    return debug


def main() -> None:
    """Write one reference JSON per scene."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenes-dir", type=Path, default=DEFAULT_SCENES_DIRECTORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--debug", action="store_true", help="also record per-stage intermediates")
    parser.add_argument("--dump-maps", action="store_true", help="also write every channel map as raw float32")
    parser.add_argument("scene_ids", nargs="+")
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    config = ClassicalDetectorConfig()
    for scene_id in arguments.scene_ids:
        image_bgr = read_scene_bgr(arguments.scenes_dir, scene_id)
        start_time = time.perf_counter()
        detections = detect_checks_classical(image_bgr, config)
        elapsed_seconds = time.perf_counter() - start_time
        payload = {
            "scene_id": scene_id,
            "seconds": elapsed_seconds,
            "detections": [
                {"corners": detection.corners.tolist(), "score": detection.score,
                 "source_name": detection.extras["source_name"], "side_supports": detection.extras["side_supports"]}
                for detection in detections
            ],
        }
        if arguments.debug or arguments.dump_maps:
            maps_prefix = arguments.output / scene_id if arguments.dump_maps else None
            payload["debug"] = collect_stage_debug(image_bgr, config, maps_prefix)
        (arguments.output / f"{scene_id}.json").write_text(json.dumps(payload))
        print(f"{scene_id}: {len(detections)} checks in {elapsed_seconds:.2f}s", flush=True)


if __name__ == "__main__":
    main()
