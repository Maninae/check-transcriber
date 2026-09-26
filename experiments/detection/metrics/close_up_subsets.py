"""Score several detectors on the close-up subsets of a split (the regime the operator shoots).

the operator's photos are closer up than most v1 scenes: 5-6 checks filling the frame, sometimes
one check filling it. v1 under-represents that, so this module scores the scenes that
come closest:

- `large-checks`: scenes whose mean check width (long side of the GT corner quad, in
  photo pixels) is in the split's top quartile.
- `large-checks-frame-relative`: same, but width as a fraction of the photo's long side,
  because v1 photos range from ~2 to ~6 MP and pixel width partly measures resolution.
- `one-or-two-checks`: scenes with 1-2 checks.

    python -m experiments.detection.metrics.close_up_subsets --split eval \
        --detector name=<predictions.json> [--detector ...] --output-dir <dir>

Writes `close_up_subsets.json` and `close_up_subsets.md` (one headline table per subset).
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from experiments.detection.dataset.scene_annotations import SceneAnnotation, load_split_scene_annotations
from experiments.detection.metrics.score_predictions import score_predictions_against_split
from experiments.detection.predictions.detected_check import load_predictions_file

logger = logging.getLogger(__name__)

TOP_QUARTILE_PERCENTILE = 75
MAX_CHECKS_FOR_FEW_CHECK_SUBSET = 2


def mean_check_width_pixels(scene: SceneAnnotation) -> float:
    """Mean over checks of the long side of the GT corner quad (average of top and bottom edges)."""
    widths = [
        (np.linalg.norm(check.corners[1] - check.corners[0]) + np.linalg.norm(check.corners[2] - check.corners[3])) / 2
        for check in scene.checks
    ]
    return float(np.mean(widths))


def select_close_up_subsets(scenes: list[SceneAnnotation]) -> dict[str, list[SceneAnnotation]]:
    """Subset name -> scenes; thresholds come from the split itself."""
    pixel_widths = np.array([mean_check_width_pixels(scene) for scene in scenes])
    relative_widths = np.array(
        [width / max(scene.image_width, scene.image_height) for width, scene in zip(pixel_widths, scenes)]
    )
    pixel_threshold = np.percentile(pixel_widths, TOP_QUARTILE_PERCENTILE)
    relative_threshold = np.percentile(relative_widths, TOP_QUARTILE_PERCENTILE)
    logger.info("top-quartile thresholds: %.0f px mean check width, %.3f of photo long side", pixel_threshold, relative_threshold)
    return {
        "all": scenes,
        "large-checks": [scene for scene, width in zip(scenes, pixel_widths) if width >= pixel_threshold],
        "large-checks-frame-relative": [scene for scene, width in zip(scenes, relative_widths) if width >= relative_threshold],
        "one-or-two-checks": [scene for scene in scenes if len(scene.checks) <= MAX_CHECKS_FOR_FEW_CHECK_SUBSET],
    }


def headline_row(detector_name: str, metrics: dict) -> str:
    """One markdown row: detection, all-correct photos, corner error, orientation."""
    detection = metrics["detection"]
    localization = metrics["localization"]
    corner_error = localization["corner_error_mean_px"]
    orientation = localization["orientation_accuracy"]
    return (
        f"| {detector_name} | {100 * detection['0.50']['precision']:.1f} | {100 * detection['0.50']['recall']:.1f}"
        f" | {100 * detection['0.90']['recall']:.1f} | {100 * metrics['scenes']['scene_perfect_rate']:.1f}"
        f" | {corner_error['median']:.2f} | {corner_error['p90']:.1f}"
        f" | {100 * localization['fraction_corner_error_below_px']['5']:.0f}"
        f" | {'–' if orientation is None else f'{100 * orientation:.1f}'} |"
    )


def main() -> None:
    """Score every detector on every subset and write JSON + markdown."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", default="eval")
    parser.add_argument("--detector", action="append", required=True, help="name=<predictions.json>")
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    subsets = select_close_up_subsets(load_split_scene_annotations(arguments.split))
    detectors = [argument.split("=", 1) for argument in arguments.detector]
    predictions_by_detector = {name: load_predictions_file(Path(path))[1] for name, path in detectors}
    summary, markdown_sections = {}, [f"# Close-up subsets of {arguments.split}\n"]
    for subset_name, subset_scenes in subsets.items():
        check_count = sum(len(scene.checks) for scene in subset_scenes)
        markdown_sections.append(
            f"## {subset_name} ({len(subset_scenes)} photos, {check_count} checks)\n\n"
            "| Pipeline | Precision | Recall | Recall@0.9 | Photos all-correct | Corner median px | p90 | < 5 px % | Orientation |\n"
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
        summary[subset_name] = {}
        for detector_name, predictions in predictions_by_detector.items():
            metrics = score_predictions_against_split(
                predictions, subset_scenes, detector_name=detector_name, include_records=False
            )
            summary[subset_name][detector_name] = {key: metrics[key] for key in ("detection", "localization", "scenes")}
            markdown_sections.append(headline_row(detector_name, metrics))
        markdown_sections.append("")
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    (arguments.output_dir / "close_up_subsets.json").write_text(json.dumps(summary, indent=2))
    (arguments.output_dir / "close_up_subsets.md").write_text("\n".join(markdown_sections) + "\n")
    logger.info("wrote %s", arguments.output_dir / "close_up_subsets.md")


if __name__ == "__main__":
    main()
