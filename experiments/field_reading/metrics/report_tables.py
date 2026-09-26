"""Render report sections as compact markdown tables.

Numbers are percentages with one decimal (accuracy, coverage) or three-decimal rates (CER), so
columns line up. Every table is method-major: one column per method, so a reader compares
methods left to right within a field.
"""

import pandas as pd

MISSING_CELL = "-"


def format_percent(value: float | None) -> str:
    """0.9534 -> `95.3`; None/NaN -> `-`."""
    return MISSING_CELL if value is None or pd.isna(value) else f"{100 * value:.1f}"


def format_rate(value: float | None) -> str:
    """0.0421 -> `0.042`; None/NaN -> `-`."""
    return MISSING_CELL if value is None or pd.isna(value) else f"{value:.3f}"


def markdown_table(header_cells: list[str], body_rows: list[list[str]], right_align_from_column: int = 1) -> str:
    """Pipe table; columns from `right_align_from_column` on are right-aligned (numbers)."""
    alignment_cells = [":--" if index < right_align_from_column else "--:" for index in range(len(header_cells))]
    lines = ["| " + " | ".join(header_cells) + " |", "| " + " | ".join(alignment_cells) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in body_rows]
    return "\n".join(lines)


def accuracy_and_cer_pivot(summary: pd.DataFrame, row_column: str, row_order: list[str] | None = None) -> str:
    """Rows = `row_column` values, columns = `n` then `acc% / CER` per method."""
    if summary.empty:
        return "_no rows_"
    method_labels = sorted(summary.method_label.unique())
    row_values = row_order or sorted(summary[row_column].astype(str).unique())
    indexed = summary.assign(row_value=summary[row_column].astype(str)).set_index(["row_value", "method_label"])
    body_rows = []
    for row_value in row_values:
        if not any((row_value, label) in indexed.index for label in method_labels):
            continue
        row_count = next(int(indexed.loc[(row_value, label), "n"]) for label in method_labels if (row_value, label) in indexed.index)
        cells = [row_value, str(row_count)]
        for label in method_labels:
            if (row_value, label) not in indexed.index:
                cells.append(MISSING_CELL)
                continue
            record = indexed.loc[(row_value, label)]
            cells.append(f"{format_percent(record.accuracy)} / {format_rate(record.mean_cer)}")
        body_rows.append(cells)
    return markdown_table([row_column, "n", *[f"{label} acc% / CER" for label in method_labels]], body_rows)


def headline_detail_table(summary: pd.DataFrame) -> str:
    """Per method x field: n, coverage, accuracy, accuracy on filled, exact raw, text-level accuracy, CER, WER."""
    body_rows = [
        [record.method_label, record.field_name, str(int(record.n)), format_percent(record.coverage),
         format_percent(record.accuracy), format_percent(record.accuracy_on_filled), format_percent(record.exact_raw),
         format_percent(record.text_accuracy), format_rate(record.mean_cer), format_rate(record.mean_wer)]
        for record in summary.itertuples()
    ]
    header_cells = ["method", "field", "n", "cov%", "acc%", "acc on filled%", "exact raw%", "text acc%", "CER", "WER"]
    return markdown_table(header_cells, body_rows, right_align_from_column=2)


def gating_table(gating_entries: list[dict]) -> str:
    """Per method x field: accuracy at 80/90/95/100% coverage and max coverage (@threshold) at 95/98/99% accuracy."""
    body_rows = []
    for entry in gating_entries:
        gating = entry["gating"]
        accuracy_at_coverage = gating.get("accuracy_at_coverage", {})
        max_coverage = gating.get("max_coverage_at_accuracy", {})
        coverage_cells = [format_percent(accuracy_at_coverage.get(level)) for level in ("0.80", "0.90", "0.95", "1.00")]
        target_cells = []
        for target in ("0.95", "0.98", "0.99"):
            point = max_coverage.get(target)
            threshold_text = MISSING_CELL if not point or point["threshold"] is None else f"{point['threshold']:.3f}"
            target_cells.append(MISSING_CELL if not point else f"{format_percent(point['coverage'])} @{threshold_text}")
        fill_all = gating["fill_all"]
        body_rows.append([entry["method_label"], entry["field_name"], "yes" if gating["has_confidence"] else "no",
                          f"{format_percent(fill_all['accuracy'])} @ {format_percent(fill_all['coverage'])}",
                          *coverage_cells, *target_cells])
    header_cells = ["method", "field", "conf", "fill-all acc @ cov", "acc@80", "acc@90", "acc@95", "acc@100",
                    "cov@95%acc", "cov@98%acc", "cov@99%acc"]
    return markdown_table(header_cells, body_rows, right_align_from_column=3)


def methods_by_fields_markdown(comparison: pd.DataFrame, value_formatter=format_percent) -> str:
    """Render a methods x fields pivot (index = method_label, columns = field names)."""
    body_rows = [[str(method_label), *[value_formatter(value) for value in row]] for method_label, row in comparison.iterrows()]
    return markdown_table(["method", *[str(column) for column in comparison.columns]], body_rows)
