/**
 * Clipboard text for the review grid (spec 4.5): plain tab-separated values, one line
 * per check, columns in the copy order, so a paste into Google Sheets or Excel fills one
 * cell per field with no dialog.
 *
 * Writing to the clipboard from a click needs no permission prompt in Chrome and Edge.
 */

import { DEFAULT_COPY_COLUMN_ORDER } from "./field_definitions.js";

// A tab or newline inside a typed value would split one cell into two when pasted.
function sanitizeCellText(value) {
  return value.replace(/[\t\r\n]+/g, " ").trim();
}

/** One tab-separated line for one row's field values. */
export function formatRowAsTabSeparatedLine(fieldValues, columnOrder = DEFAULT_COPY_COLUMN_ORDER) {
  return columnOrder.map((fieldKey) => sanitizeCellText(fieldValues[fieldKey] ?? "")).join("\t");
}

/** Several rows, one line each. */
export function formatRowsAsTabSeparatedText(fieldValueRecords, columnOrder = DEFAULT_COPY_COLUMN_ORDER) {
  return fieldValueRecords.map((fieldValues) => formatRowAsTabSeparatedLine(fieldValues, columnOrder)).join("\n");
}

/** Writes plain text to the clipboard; resolves when written. */
export function writeTextToClipboard(text) {
  return navigator.clipboard.writeText(text);
}
