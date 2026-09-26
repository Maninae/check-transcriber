/**
 * The DOM for one review-grid row (spec 4.3): the upright crop on the left with its number,
 * rotate and "show bottom line" (MICR) controls, the box outline of the focused field and
 * the inline magnifier under the crop; the field lines (review_field_view.js) with the
 * duplicate warning under them; Copy row and the done checkbox on the right.
 *
 * Only fields take Tab focus (every button is tabindex -1). This module builds and updates
 * DOM only; the row's state lives in review_grid.js, which passes callbacks in.
 */

import { REVIEW_FIELDS } from "./field_definitions.js";
import { createReviewFieldView, createUntabbableButton } from "./review_field_view.js";
import { drawMagnifiedRegion, positionBoxOutline } from "./field_magnifier.js";

export const CROP_DISPLAY_WIDTH_PX = 500;
const COPIED_FLASH_MS = 1000;

function createElement(tagName, className, textContent) {
  const element = document.createElement(tagName);
  if (className) element.className = className;
  if (textContent !== undefined) element.textContent = textContent;
  return element;
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

function createCropColumn(checkNumber, callbacks) {
  const cropColumn = createElement("div", "review-crop-column");
  const cropPlaceholder = createElement("div", "review-crop-placeholder", "Straightening…");
  const cropFrame = createElement("div", "review-crop-frame");
  cropFrame.hidden = true;
  const cropCanvas = createElement("canvas", "review-crop-canvas");
  cropCanvas.addEventListener("click", callbacks.onCropClick);
  const boxOutline = createElement("div", "review-crop-box-outline");
  boxOutline.hidden = true;
  cropFrame.append(cropCanvas, boxOutline);
  const magnifierCanvas = createElement("canvas", "review-magnifier");
  magnifierCanvas.hidden = true;
  magnifierCanvas.setAttribute("aria-hidden", "true");
  const numberBadge = createElement("span", "review-row-number", `Check ${checkNumber}`);
  const cropActions = createElement("div", "review-crop-actions");
  const rotateButton = createUntabbableButton("small-link-button", "Rotate", callbacks.onRotate);
  const micrToggleButton = createUntabbableButton("small-link-button", "Show bottom line", callbacks.onToggleMicr);
  micrToggleButton.title = "The bottom line holds bank numbers and is blurred by default";
  // The number sits under the crop, never on it: the payer name is printed top-left.
  cropActions.append(numberBadge, rotateButton, micrToggleButton);
  cropColumn.append(cropPlaceholder, cropFrame, magnifierCanvas, cropActions);
  return { cropColumn, cropPlaceholder, cropFrame, cropCanvas, boxOutline, magnifierCanvas, micrToggleButton };
}

/**
 * Builds one row. `callbacks`: onCropClick, onRotate, onToggleMicr, onCopyRow(button),
 * onDoneChange(isDone), and `fieldCallbacks(fieldKey)` returning the callbacks of
 * review_field_view.js for that field.
 */
export function createReviewRowView(checkNumber, callbacks) {
  const rowElement = createElement("article", "review-row");
  rowElement.dataset.checkNumber = String(checkNumber);
  const crop = createCropColumn(checkNumber, callbacks);

  const fieldsColumn = createElement("div", "review-fields-column");
  const fieldsGrid = createElement("div", "review-fields");
  const fieldViews = new Map();
  for (const field of REVIEW_FIELDS) {
    const fieldView = createReviewFieldView(checkNumber, field, callbacks.fieldCallbacks(field.key));
    fieldsGrid.append(fieldView.labelElement, fieldView.cellElement, fieldView.copyButton);
    fieldViews.set(field.key, fieldView);
  }
  const duplicateWarning = createElement("p", "review-duplicate-warning");
  duplicateWarning.hidden = true;
  fieldsColumn.append(fieldsGrid, duplicateWarning);

  const sideColumn = createElement("div", "review-row-side");
  const copyRowButton = createUntabbableButton("quiet-button", "Copy row", () => callbacks.onCopyRow(copyRowButton));
  const doneLabel = createElement("label", "review-done-label");
  const doneCheckbox = createElement("input");
  doneCheckbox.type = "checkbox";
  doneCheckbox.tabIndex = -1;
  doneCheckbox.addEventListener("change", () => callbacks.onDoneChange(doneCheckbox.checked));
  doneLabel.append(doneCheckbox, document.createTextNode("Done"));
  sideColumn.append(copyRowButton, doneLabel);

  rowElement.append(crop.cropColumn, fieldsColumn, sideColumn);

  return {
    rowElement,
    cropCanvas: crop.cropCanvas,
    magnifierCanvas: crop.magnifierCanvas,
    fieldViews,
    showCrop(drawIntoCanvas) {
      drawIntoCanvas(crop.cropCanvas, CROP_DISPLAY_WIDTH_PX);
      crop.cropPlaceholder.hidden = true;
      crop.cropFrame.hidden = false;
    },
    setDone(isDone) {
      doneCheckbox.checked = isDone;
      rowElement.classList.toggle("review-row--done", isDone);
    },
    setMicrShown(isShown) {
      crop.micrToggleButton.textContent = isShown ? "Blur bottom line" : "Show bottom line";
    },
    /** `text` like "Seen before: batch on Sep 12.", or null to hide. */
    setDuplicateWarning(text) {
      duplicateWarning.textContent = text || "";
      duplicateWarning.hidden = !text;
    },
    /** Outlines `box` on the crop and magnifies `region` of the displayed crop under it. */
    showMagnifier(displayedCropCanvas, region, box) {
      positionBoxOutline(crop.boxOutline, box, displayedCropCanvas.width, CROP_DISPLAY_WIDTH_PX);
      drawMagnifiedRegion(crop.magnifierCanvas, displayedCropCanvas, region, CROP_DISPLAY_WIDTH_PX);
      crop.magnifierCanvas.hidden = false;
    },
    hideMagnifier() {
      crop.boxOutline.hidden = true;
      crop.magnifierCanvas.hidden = true;
      crop.magnifierCanvas.width = 0; // its pixels are a copy of the crop; drop them
    },
    isMagnifierOpen() {
      return !crop.magnifierCanvas.hidden;
    },
  };
}
