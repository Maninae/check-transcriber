/**
 * The DOM for one review-grid row (spec 4.3): the upright crop on the left with its
 * number, rotate and "show bottom line" (MICR) controls; the hand-typed fields with a copy
 * button each; a Copy row button and the done checkbox on the right.
 *
 * Only fields take Tab focus (every button is tabindex -1), so Tab walks field to field
 * down the whole batch without the mouse. This module builds and updates DOM only; the
 * row's state lives in review_grid.js, which passes callbacks in.
 */

import { REVIEW_FIELDS } from "./field_definitions.js";
import { attachFieldUndoHistory } from "./field_undo.js";

const CROP_DISPLAY_WIDTH_PX = 500;
const COPIED_FLASH_MS = 1000;
const COPY_ICON = "⧉"; // two joined squares, the usual copy glyph

function createElement(tagName, className, textContent) {
  const element = document.createElement(tagName);
  if (className) element.className = className;
  if (textContent !== undefined) element.textContent = textContent;
  return element;
}

function createButton(className, text, onClick) {
  const button = createElement("button", className, text);
  button.type = "button";
  button.tabIndex = -1;
  button.addEventListener("click", onClick);
  return button;
}

/** Shows "copied" feedback on a button for a second. */
export function flashCopied(button) {
  const originalText = button.textContent;
  const originalTitle = button.title;
  button.classList.add("copy-icon-button--copied");
  button.textContent = button.classList.contains("copy-icon-button") ? "✓" : "Copied";
  button.title = "Copied";
  setTimeout(() => {
    button.classList.remove("copy-icon-button--copied");
    button.textContent = originalText;
    button.title = originalTitle;
  }, COPIED_FLASH_MS);
}

/**
 * Builds one row. `callbacks`: onCropClick, onRotate, onToggleMicr, onFieldInput(key, value),
 * onCopyField(key, button), onCopyRow(button), onDoneChange(isDone), onEnterInField(input).
 */
export function createReviewRowView(checkNumber, callbacks) {
  const rowElement = createElement("article", "review-row");
  rowElement.dataset.checkNumber = String(checkNumber);

  const cropColumn = createElement("div", "review-crop-column");
  const cropPlaceholder = createElement("div", "review-crop-placeholder", "Straightening…");
  const cropCanvas = createElement("canvas", "review-crop-canvas");
  cropCanvas.hidden = true;
  cropCanvas.addEventListener("click", callbacks.onCropClick);
  const numberBadge = createElement("span", "review-row-number", `Check ${checkNumber}`);
  const cropActions = createElement("div", "review-crop-actions");
  const rotateButton = createButton("small-link-button", "Rotate", callbacks.onRotate);
  const micrToggleButton = createButton("small-link-button", "Show bottom line", callbacks.onToggleMicr);
  micrToggleButton.title = "The bottom line holds bank numbers and is blurred by default";
  // The number sits under the crop, never on it: the payer name is printed top-left.
  cropActions.append(numberBadge, rotateButton, micrToggleButton);
  cropColumn.append(cropPlaceholder, cropCanvas, cropActions);

  const fieldsGrid = createElement("div", "review-fields");
  const fieldInputs = new Map();
  for (const { key, label } of REVIEW_FIELDS) {
    const inputId = `field-${checkNumber}-${key}`;
    const labelElement = createElement("label", "review-field-label", label);
    labelElement.htmlFor = inputId;
    const input = createElement("input", "review-field-input");
    input.id = inputId;
    input.type = "text";
    input.autocomplete = "off";
    input.spellcheck = false;
    input.dataset.fieldKey = key;
    input.addEventListener("input", () => callbacks.onFieldInput(key, input.value));
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        callbacks.onEnterInField(input);
      }
    });
    attachFieldUndoHistory(input, (restoredValue) => callbacks.onFieldInput(key, restoredValue));
    const copyButton = createButton("copy-icon-button", COPY_ICON, () => callbacks.onCopyField(key, copyButton));
    copyButton.title = `Copy ${label.toLowerCase()}`;
    copyButton.setAttribute("aria-label", `Copy ${label.toLowerCase()} for check ${checkNumber}`);
    fieldsGrid.append(labelElement, input, copyButton);
    fieldInputs.set(key, input);
  }

  const sideColumn = createElement("div", "review-row-side");
  const copyRowButton = createButton("quiet-button", "Copy row", () => callbacks.onCopyRow(copyRowButton));
  const doneLabel = createElement("label", "review-done-label");
  const doneCheckbox = createElement("input");
  doneCheckbox.type = "checkbox";
  doneCheckbox.tabIndex = -1;
  doneCheckbox.addEventListener("change", () => callbacks.onDoneChange(doneCheckbox.checked));
  doneLabel.append(doneCheckbox, document.createTextNode("Done"));
  sideColumn.append(copyRowButton, doneLabel);

  rowElement.append(cropColumn, fieldsGrid, sideColumn);

  return {
    rowElement,
    cropCanvas,
    fieldInputs,
    showCrop(drawIntoCanvas) {
      drawIntoCanvas(cropCanvas, CROP_DISPLAY_WIDTH_PX);
      cropPlaceholder.hidden = true;
      cropCanvas.hidden = false;
    },
    setDone(isDone) {
      doneCheckbox.checked = isDone;
      rowElement.classList.toggle("review-row--done", isDone);
    },
    setMicrShown(isShown) {
      micrToggleButton.textContent = isShown ? "Blur bottom line" : "Show bottom line";
    },
  };
}
