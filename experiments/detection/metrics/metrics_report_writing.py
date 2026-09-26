"""Write a scored run as `metrics.json` (everything) and `metrics.md` (human tables).

Markdown precision is fixed per column kind so columns read cleanly:
rates as percentages with 1 decimal, IoU with 4 decimals, pixel and
percent-of-short-side errors with 2 decimals, counts as integers.
Missing values (empty denominators) print as an en dash.
"""

import json
from pathlib import Path

from experiments.detection.metrics.attribute_breakdowns import SCENE_ATTRIBUTE_NAMES, STRICT_RECALL_IOU_THRESHOLD
from experiments.detection.metrics.metric_records import PRIMARY_MATCHING_IOU_THRESHOLD, threshold_key

MISSING_VALUE_TEXT = "–"
IOU_DECIMALS = 4
ERROR_DECIMALS = 2
PERCENT_DECIMALS = 1
MILLISECONDS_PER_SECOND = 1000.0


def format_percent(ratio: float | None) -> str:
    """0.9734 -> "97.3"."""
    return MISSING_VALUE_TEXT if ratio is None else f"{100.0 * ratio:.{PERCENT_DECIMALS}f}"


def format_fixed(value: float | None, decimals: int) -> str:
    """Fixed-point text with a set number of decimals."""
    return MISSING_VALUE_TEXT if value is None else f"{value:.{decimals}f}"


def markdown_table(header_cells: list[str], rows: list[list[str]], right_aligned_from_column: int = 1) -> str:
    """Pipe table; text columns left-aligned, numeric columns (from the given index) right-aligned."""
    alignment_cells = [
        "---:" if column_index >= right_aligned_from_column else ":---" for column_index in range(len(header_cells))
    ]
    lines = ["| " + " | ".join(header_cells) + " |", "| " + " | ".join(alignment_cells) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def headline_summary_line(metrics: dict) -> str:
    """One line for logs and tuning loops, e.g. "F1@0.50 97.1% | R@0.90 81.0% | ..."."""
    primary_key = threshold_key(PRIMARY_MATCHING_IOU_THRESHOLD)
    strict_key = threshold_key(STRICT_RECALL_IOU_THRESHOLD)
    detection = metrics["detection"]
    localization = metrics["localization"]
    corner_error = localization["corner_error_mean_px"]
    return (
        f"F1@{primary_key} {format_percent(detection[primary_key]['f1'])}%"
        f" | P@{primary_key} {format_percent(detection[primary_key]['precision'])}%"
        f" | R@{primary_key} {format_percent(detection[primary_key]['recall'])}%"
        f" | R@{strict_key} {format_percent(detection[strict_key]['recall'])}%"
        f" | IoU {format_fixed(localization['iou_with_outline']['mean'], IOU_DECIMALS)}"
        f" | corner px med {format_fixed(corner_error['median'], ERROR_DECIMALS)}"
        f" p90 {format_fixed(corner_error['p90'], ERROR_DECIMALS)}"
        f" | orient {format_percent(localization['orientation_accuracy'])}%"
        f" | scenes perfect {format_percent(metrics['scenes']['scene_perfect_rate'])}%"
    )


def headline_markdown_sections(metrics: dict) -> list[str]:
    """Detection, IoU, corner-error and rate tables."""
    detection, localization, scenes = metrics["detection"], metrics["localization"], metrics["scenes"]
    detection_rows = [
        [key]
        + [format_percent(rates[rate_name]) for rate_name in ("precision", "recall", "f1")]
        + [str(rates[count_name]) for count_name in ("true_positives", "predictions", "ground_truth")]
        for key, rates in detection.items()
    ]
    iou_rows = [
        [
            label,
            format_fixed(localization[key]["mean"], IOU_DECIMALS),
            format_fixed(localization[key]["median"], IOU_DECIMALS),
        ]
        for label, key in (("outline polygon", "iou_with_outline"), ("4-corner quad", "iou_with_corner_quad"))
    ]
    corner_rows = [
        [label]
        + [format_fixed(localization[key][statistic], ERROR_DECIMALS) for statistic in ("mean", "median", "p90", "p95")]
        for label, key in (
            ("mean of corners, px", "corner_error_mean_px"),
            ("worst corner, px", "corner_error_max_px"),
            ("mean of corners, % short side", "corner_error_percent_short_side"),
        )
    ]
    rate_rows = [[f"corner error < {cutoff} px", format_percent(ratio)]
                 for cutoff, ratio in localization["fraction_corner_error_below_px"].items()]
    rate_rows += [
        [f"orientation accuracy (n={localization['orientation_known_checks']} orientation-known)",
         format_percent(localization["orientation_accuracy"])],
        ["axis accuracy (all matched)", format_percent(localization["axis_accuracy"])],
        ["count accuracy (scenes)", format_percent(scenes["count_accuracy"])],
        ["scene perfect", format_percent(scenes["scene_perfect_rate"])],
    ]
    seconds_per_image = metrics["latency"]["seconds_per_image"]
    milliseconds_per_image = None if seconds_per_image is None else seconds_per_image * MILLISECONDS_PER_SECOND
    note_items = [
        f"matched checks at 0.50: {localization['matched_checks']}",
        f"with some GT corners out of frame: {localization['checks_with_partial_corners']}",
        f"with no usable GT corner: {localization['checks_without_usable_corners']}",
        f"counter-clockwise predictions: {localization['counter_clockwise_predictions']}",
        f"scenes missing from predictions: {scenes['scenes_missing_from_predictions']}",
        f"mean count error: {format_fixed(scenes['mean_absolute_count_error'], ERROR_DECIMALS)}",
        f"latency: {format_fixed(milliseconds_per_image, PERCENT_DECIMALS)} ms/image",
    ]
    notes = "; ".join(note_items) + "."
    return [
        "## Detection\n\n"
        + markdown_table(["IoU ≥", "Precision %", "Recall %", "F1 %", "TP", "Predictions", "GT"], detection_rows),
        "## IoU of matched checks (at 0.50)\n\n" + markdown_table(["Against", "Mean", "Median"], iou_rows),
        "## Corner error of matched checks (best cyclic shift)\n\n"
        + markdown_table(["Corner error", "Mean", "Median", "p90", "p95"], corner_rows),
        "## Rates\n\n" + markdown_table(["Rate", "%"], rate_rows) + "\n\n" + notes,
    ]


def breakdown_markdown_section(attribute_name: str, rows_by_group: dict[str, dict]) -> str:
    """One breakdown table; scene attributes also get Scenes and Precision columns."""
    primary_key = threshold_key(PRIMARY_MATCHING_IOU_THRESHOLD)
    strict_key = threshold_key(STRICT_RECALL_IOU_THRESHOLD)
    is_scene_attribute = attribute_name in SCENE_ATTRIBUTE_NAMES
    header_cells = [attribute_name] + (["Scenes"] if is_scene_attribute else []) + ["Checks"]
    header_cells += [f"Recall@{primary_key} %", f"Recall@{strict_key} %"]
    header_cells += [f"Precision@{primary_key} %"] if is_scene_attribute else []
    header_cells += ["Mean IoU", "Median corner px"]
    table_rows = []
    for group_value, row in rows_by_group.items():
        cells = [group_value] + ([str(row["scenes"])] if is_scene_attribute else []) + [str(row["checks"])]
        cells += [format_percent(row[f"recall@{primary_key}"]), format_percent(row[f"recall@{strict_key}"])]
        cells += [format_percent(row[f"precision@{primary_key}"])] if is_scene_attribute else []
        cells += [
            format_fixed(row["mean_iou_with_outline"], IOU_DECIMALS),
            format_fixed(row["median_corner_error_px"], ERROR_DECIMALS),
        ]
        table_rows.append(cells)
    return f"### {attribute_name}\n\n" + markdown_table(header_cells, table_rows)


def render_metrics_markdown(metrics: dict) -> str:
    """Full markdown report: headline line, headline tables, one table per breakdown."""
    run = metrics["run"]
    sections = [
        f"# Detection metrics: {run['detector_name']} on {run['split_name']} ({metrics['scenes']['scenes']} scenes)",
        f"`{headline_summary_line(metrics)}`\n\nScore threshold {run['score_threshold']}. "
        "Deformation groups overlap (a folded, curled check counts in both).",
    ]
    sections.extend(headline_markdown_sections(metrics))
    sections.append("## Breakdowns")
    sections.extend(
        breakdown_markdown_section(attribute_name, rows_by_group)
        for attribute_name, rows_by_group in metrics["breakdowns"].items()
    )
    return "\n\n".join(sections) + "\n"


def write_metrics_report(metrics: dict, output_directory: Path) -> tuple[Path, Path]:
    """Write metrics.json and metrics.md into `output_directory`; returns both paths."""
    output_directory.mkdir(parents=True, exist_ok=True)
    metrics_json_path = output_directory / "metrics.json"
    metrics_markdown_path = output_directory / "metrics.md"
    with open(metrics_json_path, "w") as metrics_json_file:
        json.dump(metrics, metrics_json_file, indent=1)
    metrics_markdown_path.write_text(render_metrics_markdown(metrics))
    return metrics_json_path, metrics_markdown_path
