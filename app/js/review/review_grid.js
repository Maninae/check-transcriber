/**
 * The review grid (spec 4.3, 4.5): owns every row's state; review_row.js builds the DOM
 * and review_field_editing.js handles what the operator does inside a field.
 *
 * Row state: the upright crop from the worker (full 1600 px resolution) and the displayed
 * one (rotated, MICR blurred), the classifier's upside-down probability, the operator's
 * half-turn, whether the MICR band is shown, the last raw field reads (kept so the rows can
 * be re-gated when the known-names lists change, without touching pixels), one field
 * record per field (review_field_states.js), and the done flag. Rows keep the count step's
 * numbering, so copied rows line up with the numbers on the photo.
 *
 * Field reads arrive through `receiveFieldReads` (the worker's per-check message, the
 * re-read after Rotate, and the test hook all take this one path) and go through
 * fields/field_gating.js `gateCheckFields`. A field the operator edited or confirmed is
 * never overwritten by a later read.
 */

import { createReviewRowView, flashCopied } from "./review_row.js";
import { REVIEW_FIELDS } from "./field_definitions.js";
import { formatRowAsTabSeparatedLine, formatRowsAsTabSeparatedText, writeTextToClipboard } from "./clipboard_rows.js";
import { magnifierRegionForField } from "./field_magnifier.js";
import { composeDisplayedCrop, createCanvasFromRgbaPixels, drawCanvasScaledToWidth, shouldBlurTopBand } from "./crop_rendering.js";
import { applyGatedFieldState, copyValueForField, createUnreadFieldRecord, displayTextForValue, rotateBoxHalfTurn } from "./review_field_states.js";
import { createFieldCallbacks, describeRowForDebug, renderField } from "./review_field_editing.js";
import { gateCheckFields } from "../fields/field_gating.js";
import { formatIsoDateAsShortMonthDay, parseDateAssumingYear } from "../fields/date_parsing.js";

function createRowState() {
  return {
    uprightCropCanvas: null,
    displayedCropCanvas: null,
    upsideDownProbability: 0,
    rotatedHalfTurn: false,
    micrShown: false,
    rawReads: null,
    rawReadsRotatedHalfTurn: false, // the orientation the stored reads' boxes are in
    readRequestGeneration: 0, // bumps on every Rotate so a stale re-read is ignored
    fields: Object.fromEntries(REVIEW_FIELDS.map(({ key }) => [key, createUnreadFieldRecord()])),
    done: false,
    view: null,
  };
}

export class ReviewGrid {
  /**
   * `options`: `{ rowsContainerElement, headingElement, emailDateInputElement, toast, settings,
   * batchHistory, onOpenLightbox(index), onRowsChanged(), rereadCheckFields(index, rotatedHalfTurn) }`;
   * `rereadCheckFields` resolves to `{ rawReads }` (pipeline_client.js).
   */
  constructor(options) {
    Object.assign(this, options);
    this.rows = [];
    this.emailDateIso = null;
    this.magnifiedField = null; // { index, fieldKey } while a magnifier is open
    this.emailDateInputElement.addEventListener("input", () => this.setEmailDateText(this.emailDateInputElement.value));
  }

  /** Builds `checkCount` empty rows (crops and reads arrive one by one). */
  start(checkCount) {
    this.clear();
    this.headingElement.textContent = `${checkCount} ${checkCount === 1 ? "check" : "checks"}`;
    for (let index = 0; index < checkCount; index += 1) {
      const row = createRowState();
      row.view = createReviewRowView(index + 1, {
        onCropClick: () => this.onOpenLightbox(index),
        onRotate: () => this.rotateCheck(index),
        onToggleMicr: () => this.toggleMicrShown(index),
        onCopyRow: (button) => this.copyRow(index, button),
        onDoneChange: (isDone) => this.setDone(index, isDone),
        fieldCallbacks: (fieldKey) => createFieldCallbacks(this, index, fieldKey),
      });
      this.rows.push(row);
      this.rowsContainerElement.append(row.view.rowElement);
      REVIEW_FIELDS.forEach(({ key }) => renderField(this, index, key));
    }
    this.onRowsChanged();
  }

  /** Receives one crop from the pipeline worker. */
  setCrop(index, { width, height, rgbaBuffer, upsideDownProbability }) {
    const row = this.rows[index];
    row.uprightCropCanvas = createCanvasFromRgbaPixels(width, height, rgbaBuffer);
    row.upsideDownProbability = upsideDownProbability;
    this.renderCrop(index);
  }

  renderCrop(index) {
    const row = this.rows[index];
    if (!row.uprightCropCanvas) return;
    if (row.displayedCropCanvas) row.displayedCropCanvas.width = 0;
    row.displayedCropCanvas = composeDisplayedCrop(row.uprightCropCanvas, {
      rotatedHalfTurn: row.rotatedHalfTurn,
      micrBlurred: !row.micrShown,
      blurTopBandToo: shouldBlurTopBand(row.upsideDownProbability, row.rotatedHalfTurn),
    });
    row.view.showCrop((canvas, displayWidth) => drawCanvasScaledToWidth(canvas, row.displayedCropCanvas, displayWidth));
    if (this.magnifiedField && this.magnifiedField.index === index) this.openMagnifier(index, this.magnifiedField.fieldKey);
  }

  /**
   * One check's raw reads (worker message, re-read after Rotate, or the test hook).
   * `readRotatedHalfTurn`: the orientation the reads' boxes are in (the worker's first
   * read is of the upright crop; a re-read is of the orientation the row asked for).
   */
  receiveFieldReads(index, rawReads, readRotatedHalfTurn = false) {
    const row = this.rows[index];
    if (!row) return;
    this.applyFieldReads(row, index, rawReads, readRotatedHalfTurn);
  }

  /**
   * A read streamed by the batch's own read chain (upright crop). Ignored once the operator
   * has rotated the row: its own re-read (both passes, in the new orientation) wins.
   */
  receiveStreamedFieldReads(index, rawReads) {
    const row = this.rows[index];
    if (!row || row.readRequestGeneration > 0) return;
    this.applyFieldReads(row, index, rawReads, false);
  }

  applyFieldReads(row, index, rawReads, readRotatedHalfTurn) {
    row.rawReads = rawReads;
    row.rawReadsRotatedHalfTurn = readRotatedHalfTurn;
    this.regateRow(index);
  }

  gatingContext() {
    return { knownPayerNames: this.batchHistory.getKnownPayerNames(), knownPayeeNames: this.settings.getPayeeNames() };
  }

  /** Re-applies the gate to a row's stored reads; touched fields keep the operator's text. */
  regateRow(index) {
    const row = this.rows[index];
    if (!row.rawReads) {
      this.updateDuplicateWarning(index); // the history may have changed ("Clear everything")
      return;
    }
    const gatedFields = gateCheckFields(row.rawReads, this.gatingContext());
    const needsBoxTurn = row.rawReadsRotatedHalfTurn !== row.rotatedHalfTurn && row.uprightCropCanvas;
    for (const { key } of REVIEW_FIELDS) {
      const gatedField = gatedFields[key];
      if (needsBoxTurn) gatedField.box = rotateBoxHalfTurn(gatedField.box, row.uprightCropCanvas.width, row.uprightCropCanvas.height);
      const previous = row.fields[key];
      const next = applyGatedFieldState(previous, gatedField, key, this.settings.getDateDisplayFormat());
      row.fields[key] = next;
      if (!next.touched && (next.text !== previous.text || next.snappedFrom !== previous.snappedFrom)) {
        row.view.fieldViews.get(key).setGatedText(next.text, next.snappedFrom);
      }
      renderField(this, index, key);
    }
    this.updateDuplicateWarning(index);
  }

  /** The known payers or the payee list changed: re-gate every row (untouched fields only change). */
  regateAllRows() {
    this.rows.forEach((row, index) => this.regateRow(index));
  }

  /** Rotates the crop 180 degrees and asks the worker to read it again that way up. */
  rotateCheck(index) {
    const row = this.rows[index];
    row.rotatedHalfTurn = !row.rotatedHalfTurn;
    if (row.uprightCropCanvas) {
      const { width, height } = row.uprightCropCanvas;
      for (const record of Object.values(row.fields)) record.box = rotateBoxHalfTurn(record.box, width, height);
    }
    this.renderCrop(index);
    for (const fieldView of row.view.fieldViews.values()) fieldView.clearUndoHistory();
    row.readRequestGeneration += 1;
    const generation = row.readRequestGeneration;
    const rotatedHalfTurn = row.rotatedHalfTurn;
    Promise.resolve()
      .then(() => this.rereadCheckFields(index, rotatedHalfTurn))
      .then((result) => {
        if (this.rows[index] !== row || row.readRequestGeneration !== generation) return; // rotated again, or a new batch
        this.receiveFieldReads(index, result.rawReads, rotatedHalfTurn);
      })
      .catch((error) => console.warn(`re-reading check ${index + 1} after rotating failed:`, error));
  }

  toggleMicrShown(index) {
    const row = this.rows[index];
    row.micrShown = !row.micrShown;
    row.view.setMicrShown(row.micrShown);
    this.renderCrop(index);
  }

  /** Opens (or redraws) the magnifier for a field of row `index`. */
  openMagnifier(index, fieldKey) {
    const row = this.rows[index];
    if (!row.displayedCropCanvas) return;
    if (this.magnifiedField && this.magnifiedField.index !== index) this.closeMagnifier();
    const { width, height } = row.displayedCropCanvas;
    const record = row.fields[fieldKey];
    row.view.showMagnifier(row.displayedCropCanvas, magnifierRegionForField(fieldKey, record.box, width, height), record.box);
    this.magnifiedField = { index, fieldKey };
  }

  closeMagnifier() {
    if (!this.magnifiedField) return;
    const row = this.rows[this.magnifiedField.index];
    if (row) row.view.hideMagnifier();
    this.magnifiedField = null;
  }

  setDone(index, isDone) {
    this.rows[index].done = isDone;
    this.rows[index].view.setDone(isDone);
    this.onRowsChanged();
  }

  /** `{ fieldKey: copy-ready value }` for one row. */
  rowCopyValues(index) {
    const { fields } = this.rows[index];
    return Object.fromEntries(REVIEW_FIELDS.map(({ key }) => [key, copyValueForField(key, fields[key].text)]));
  }

  copyField(index, fieldKey, button) {
    writeTextToClipboard(this.rowCopyValues(index)[fieldKey]).then(() => flashCopied(button));
  }

  copyRow(index, button) {
    const line = formatRowAsTabSeparatedLine(this.rowCopyValues(index), this.settings.getIncludedCopyColumnKeys());
    writeTextToClipboard(line).then(() => {
      flashCopied(button);
      this.setDone(index, true);
    });
  }

  /** Copies every row in count-step order and ticks them all done. */
  copyAllRows() {
    const records = this.rows.map((row, index) => this.rowCopyValues(index));
    const text = formatRowsAsTabSeparatedText(records, this.settings.getIncludedCopyColumnKeys());
    return writeTextToClipboard(text).then(() => {
      this.rows.forEach((row, index) => this.setDone(index, true));
      this.toast.show(`${this.rows.length} ${this.rows.length === 1 ? "row" : "rows"} copied`);
    });
  }

  /** "Seen before: batch on Sep 12." when this check number and payer were in a past batch. */
  updateDuplicateWarning(index) {
    const { checkNumber, payer } = this.rowCopyValues(index);
    const previousBatchDate = this.batchHistory.findPreviousBatchDate(checkNumber, payer);
    this.rows[index].view.setDuplicateWarning(previousBatchDate ? `Seen before: batch on ${formatIsoDateAsShortMonthDay(previousBatchDate)}.` : null);
  }

  /** The batch-level "Email date" box changed; offers it beside every blank date. */
  setEmailDateText(text) {
    this.emailDateIso = text.trim() ? parseDateAssumingYear(text, new Date().getFullYear()) : null; // "Sep 12" = this year
    this.rows.forEach((row, index) => renderField(this, index, "date"));
  }

  /** The date display setting changed: re-show every date that parses. */
  refreshDateDisplay() {
    const displayFormat = this.settings.getDateDisplayFormat();
    this.rows.forEach((row, index) => {
      const record = row.fields.date;
      record.text = displayTextForValue("date", copyValueForField("date", record.text), displayFormat);
      renderField(this, index, "date");
    });
  }

  /** Enter: focus the next unsure or blank field in the batch, across rows. */
  focusNextReviewField(currentInput) {
    const tabStops = this.rows.flatMap(({ view }) => [...view.fieldViews.values()].map(({ inputElement }) => inputElement))
      .filter((input) => input === currentInput || input.tabIndex >= 0);
    const nextInput = tabStops[tabStops.indexOf(currentInput) + 1];
    if (nextInput) nextInput.focus();
  }

  /** Rows the operator has not copied or ticked. */
  countUnfinishedRows() {
    return this.rows.filter(({ done }) => !done).length;
  }

  /** For Finish batch: each row's copy values (payer, check number, date, ...). */
  collectRowCopyValues() {
    return this.rows.map((row, index) => this.rowCopyValues(index));
  }

  /** For the lightbox: how many checks, and each one's displayed canvas. */
  getCheckCount() {
    return this.rows.length;
  }

  getDisplayedCropCanvas(index) {
    return this.rows[index].displayedCropCanvas;
  }

  /** For window.__checkTranscriberDebug: each row's field values and states (text only). */
  describeRowsForDebug() {
    return this.rows.map((row, index) => describeRowForDebug(this, index));
  }

  /** Drops every row and releases every crop's pixels. */
  clear() {
    this.closeMagnifier();
    for (const row of this.rows) {
      if (row.uprightCropCanvas) row.uprightCropCanvas.width = 0;
      if (row.displayedCropCanvas) row.displayedCropCanvas.width = 0;
      row.view.cropCanvas.width = 0;
      row.view.magnifierCanvas.width = 0;
    }
    this.rows = [];
    this.emailDateIso = null;
    this.emailDateInputElement.value = "";
    this.rowsContainerElement.replaceChildren();
  }
}
