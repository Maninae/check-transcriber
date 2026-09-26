/**
 * The review grid (spec 4.3 with milestone-3 scope: crops and hand-typed fields, no
 * automatic reads yet). Owns every row's state; review_row.js only builds DOM.
 *
 * Row state: the upright crop from the worker (full 1600 px resolution), the
 * classifier's upside-down probability (for the unsure-orientation blur), the
 * operator's half-turn rotation, whether the MICR band is shown, the typed values and
 * the done flag. Rows keep the count step's numbering, so copied rows line up with
 * the numbers on the photo.
 */

import { createReviewRowView, flashCopied } from "./review_row.js";
import { createEmptyFieldValues } from "./field_definitions.js";
import { formatRowAsTabSeparatedLine, formatRowsAsTabSeparatedText, writeTextToClipboard } from "./clipboard_rows.js";
import {
  composeDisplayedCrop,
  createCanvasFromRgbaPixels,
  drawCanvasScaledToWidth,
  isOrientationUnsure,
} from "./crop_rendering.js";

export class ReviewGrid {
  /** `options`: `{ rowsContainerElement, headingElement, toast, onOpenLightbox(index), onRowsChanged() }`. */
  constructor({ rowsContainerElement, headingElement, toast, onOpenLightbox, onRowsChanged }) {
    this.rowsContainerElement = rowsContainerElement;
    this.headingElement = headingElement;
    this.toast = toast;
    this.onOpenLightbox = onOpenLightbox;
    this.onRowsChanged = onRowsChanged;
    this.rows = [];
  }

  /** Builds `checkCount` empty rows (crops arrive one by one via `setCrop`). */
  start(checkCount) {
    this.clear();
    this.headingElement.textContent = `${checkCount} ${checkCount === 1 ? "check" : "checks"}`;
    for (let index = 0; index < checkCount; index += 1) {
      const row = {
        uprightCropCanvas: null,
        displayedCropCanvas: null,
        upsideDownProbability: 0,
        rotatedHalfTurn: false,
        micrShown: false,
        fieldValues: createEmptyFieldValues(),
        done: false,
        view: null,
      };
      row.view = createReviewRowView(index + 1, this.createRowCallbacks(index));
      this.rows.push(row);
      this.rowsContainerElement.append(row.view.rowElement);
    }
    this.onRowsChanged();
  }

  createRowCallbacks(index) {
    return {
      onCropClick: () => this.onOpenLightbox(index),
      onRotate: () => this.rotateCheck(index),
      onToggleMicr: () => this.toggleMicrShown(index),
      onFieldInput: (fieldKey, value) => { this.rows[index].fieldValues[fieldKey] = value; },
      onCopyField: (fieldKey, button) => this.copyField(index, fieldKey, button),
      onCopyRow: (button) => this.copyRow(index, button),
      onDoneChange: (isDone) => this.setDone(index, isDone),
      onEnterInField: (input) => this.focusNextField(input),
    };
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
      blurTopBandToo: isOrientationUnsure(row.upsideDownProbability),
    });
    row.view.showCrop((canvas, displayWidth) => drawCanvasScaledToWidth(canvas, row.displayedCropCanvas, displayWidth));
  }

  /** Rotates the crop 180 degrees (a later milestone also re-reads the fields here). */
  rotateCheck(index) {
    this.rows[index].rotatedHalfTurn = !this.rows[index].rotatedHalfTurn;
    this.renderCrop(index);
  }

  toggleMicrShown(index) {
    const row = this.rows[index];
    row.micrShown = !row.micrShown;
    row.view.setMicrShown(row.micrShown);
    this.renderCrop(index);
  }

  setDone(index, isDone) {
    this.rows[index].done = isDone;
    this.rows[index].view.setDone(isDone);
    this.onRowsChanged();
  }

  copyField(index, fieldKey, button) {
    writeTextToClipboard(this.rows[index].fieldValues[fieldKey].trim()).then(() => flashCopied(button));
  }

  copyRow(index, button) {
    writeTextToClipboard(formatRowAsTabSeparatedLine(this.rows[index].fieldValues)).then(() => {
      flashCopied(button);
      this.setDone(index, true);
    });
  }

  /** Copies every row in count-step order and ticks them all done. */
  copyAllRows() {
    const text = formatRowsAsTabSeparatedText(this.rows.map(({ fieldValues }) => fieldValues));
    return writeTextToClipboard(text).then(() => {
      this.rows.forEach((row, index) => this.setDone(index, true));
      this.toast.show(`${this.rows.length} ${this.rows.length === 1 ? "row" : "rows"} copied`);
    });
  }

  /** Enter moves to the next field in the batch, across rows. */
  focusNextField(currentInput) {
    const allInputs = this.rows.flatMap(({ view }) => [...view.fieldInputs.values()]);
    const nextInput = allInputs[allInputs.indexOf(currentInput) + 1];
    if (nextInput) nextInput.focus();
  }

  /** Rows the operator has not copied or ticked. */
  countUnfinishedRows() {
    return this.rows.filter(({ done }) => !done).length;
  }

  /** For the lightbox: how many checks, and each one's displayed canvas and caption. */
  getCheckCount() {
    return this.rows.length;
  }

  getDisplayedCropCanvas(index) {
    return this.rows[index].displayedCropCanvas;
  }

  /** Drops every row and releases every crop's pixels. */
  clear() {
    for (const row of this.rows) {
      if (row.uprightCropCanvas) row.uprightCropCanvas.width = 0;
      if (row.displayedCropCanvas) row.displayedCropCanvas.width = 0;
      row.view.cropCanvas.width = 0;
    }
    this.rows = [];
    this.rowsContainerElement.replaceChildren();
  }
}
