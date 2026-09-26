/**
 * What happens when the operator works inside one field of the review grid (spec 4.3
 * "Keyboard flow", autocomplete, date handling). Stateless helpers over the ReviewGrid
 * coordinator (review_grid.js), which holds every row's field records.
 *
 * - Typing, Ctrl+Z, an autocomplete pick or the email-date button: the field is edited
 *   (touched; a re-read will not overwrite it) and the duplicate warning is recomputed.
 * - Tab past or Enter: the field is committed (amount to two decimals, a parseable date to
 *   the display format, raw text kept otherwise) and an unsure field is confirmed; Enter then
 *   moves to the next unsure or blank field. Leaving the field any other way commits too.
 * - Focus on an unsure or blank field opens the magnifier; Escape or leaving closes it.
 */

import { committedTextForField, copyValueForField, isReviewedValue, isReviewTabStop, markConfirmed, markEdited, wantsMagnifier } from "./review_field_states.js";
import { formatIsoDateForDisplay } from "../fields/date_parsing.js";

/** The email date as the one-click suggestion for row `index`'s date, or null (spec 4.3 date handling). */
function emailDateSuggestionFor(grid, index) {
  if (!grid.emailDateIso || copyValueForField("date", grid.rows[index].fields.date.text)) return null;
  return formatIsoDateForDisplay(grid.emailDateIso, grid.settings.getDateDisplayFormat());
}

/** Pushes one field record into its view. */
export function renderField(grid, index, fieldKey) {
  const row = grid.rows[index];
  const emailDateSuggestion = fieldKey === "date" ? emailDateSuggestionFor(grid, index) : null;
  row.view.fieldViews.get(fieldKey).render(row.fields[fieldKey], { emailDateSuggestion });
  grid.scheduleFieldStatesChanged?.();
}

function updateRecord(grid, index, fieldKey, nextRecord) {
  grid.rows[index].fields[fieldKey] = nextRecord;
  renderField(grid, index, fieldKey);
  grid.updateDuplicateWarning(index);
}

function commitField(grid, index, fieldKey) {
  const record = grid.rows[index].fields[fieldKey];
  const committedText = committedTextForField(fieldKey, record.text, grid.settings.getDateDisplayFormat());
  if (committedText !== record.text) updateRecord(grid, index, fieldKey, { ...record, text: committedText });
}

function commitAndConfirmField(grid, index, fieldKey) {
  commitField(grid, index, fieldKey);
  updateRecord(grid, index, fieldKey, markConfirmed(grid.rows[index].fields[fieldKey]));
}

/** The callbacks review_field_view.js needs for field `fieldKey` of row `index`. */
export function createFieldCallbacks(grid, index, fieldKey) {
  const fieldView = () => grid.rows[index].view.fieldViews.get(fieldKey);
  const handleEditedText = (text) => updateRecord(grid, index, fieldKey, markEdited(grid.rows[index].fields[fieldKey], text));
  return {
    onInput: handleEditedText,
    onCommit: () => commitField(grid, index, fieldKey),
    onTabForward: () => commitAndConfirmField(grid, index, fieldKey),
    onEnter: () => {
      commitAndConfirmField(grid, index, fieldKey);
      grid.focusNextReviewField(fieldView().inputElement);
    },
    onFocus: () => {
      if (wantsMagnifier(grid.rows[index].fields[fieldKey])) grid.openMagnifier(index, fieldKey);
      else grid.closeMagnifier();
    },
    onBlur: () => grid.closeMagnifier(),
    onEscape: () => grid.closeMagnifier(),
    onCopy: (button) => {
      commitField(grid, index, fieldKey);
      grid.copyField(index, fieldKey, button);
    },
    onUseEmailDate: () => {
      const suggestion = emailDateSuggestionFor(grid, index);
      if (!suggestion) return;
      fieldView().setTextUndoably(suggestion);
      handleEditedText(suggestion);
    },
    suggestPayerNames: (typedText) => grid.batchHistory.suggestPayerNames(typedText),
    onPickPayer: (name) => {
      fieldView().setTextUndoably(name);
      handleEditedText(name);
    },
  };
}

/** Text-only snapshot of one row for the test hook (never pixels). */
export function describeRowForDebug(grid, index) {
  const row = grid.rows[index];
  const copyValues = grid.rowCopyValues(index);
  const fields = {};
  for (const [fieldKey, record] of Object.entries(row.fields)) {
    fields[fieldKey] = {
      value: copyValues[fieldKey],
      text: record.text,
      state: record.reviewState,
      gatedState: record.gatedState,
      touched: record.touched,
      note: record.note,
      snappedFrom: record.snappedFrom,
      box: record.box,
      tabStop: isReviewTabStop(record),
    };
  }
  const previousBatchDate = grid.batchHistory.findPreviousBatchDate(copyValues.checkNumber, copyValues.payer);
  const cropSize = row.displayedCropCanvas ? [row.displayedCropCanvas.width, row.displayedCropCanvas.height] : null;
  return {
    done: row.done,
    rotatedHalfTurn: row.rotatedHalfTurn,
    cropSize,
    fields,
    previousBatchDate,
    emailDateSuggestion: emailDateSuggestionFor(grid, index),
  };
}

/**
 * Finish batch (spec 4.6): per row, the copy values the history may learn from. Only values
 * read confidently or confirmed by the operator; an amber read nobody looked at becomes "".
 */
export function collectReviewedRowValues(grid) {
  return grid.rows.map((row, index) => {
    const copyValues = grid.rowCopyValues(index);
    return Object.fromEntries(Object.entries(copyValues).map(([key, value]) => [key, isReviewedValue(row.fields[key]) ? value : ""]));
  });
}
