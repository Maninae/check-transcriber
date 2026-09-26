/**
 * The DOM for one field line in a review row: label, input, copy button, and under the
 * input the amount's disagreement note and (for the date) the one-click email-date
 * suggestion. Renders a field record from review_field_states.js; owns no state.
 *
 * Rendering (spec 4.3 "Field states"), readable without a legend:
 * - confident / confirmed / not read yet: plain editable text.
 * - unsure: the value on a soft amber ground; a snapped value says what was read in its tooltip.
 * - blank: empty, same amber ground, placeholder "read from the check".
 * Tab stops: unsure and blank fields only (confident ones get tabIndex -1, still clickable).
 */

import { attachFieldUndoHistory } from "./field_undo.js";
import { attachPayerAutocomplete } from "./payer_autocomplete.js";
import { isHighlighted, isReviewTabStop, REVIEW_STATES } from "./review_field_states.js";

const BLANK_FIELD_PLACEHOLDER = "read from the check";
const SVG_NAMESPACE = "http://www.w3.org/2000/svg";
// Two overlapping sheets (copy) and a tick (copied); CSS shows one at a time.
const COPY_ICON_PATH = "M9 9h9v10H9zM6 15H5V5h10v1";
const COPIED_ICON_PATH = "M5 12.5l4.5 4.5L19 7.5";
const FIELD_ARRIVE_ANIMATION_MS = 500;

function createIcon(className, pathData) {
  const svg = document.createElementNS(SVG_NAMESPACE, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("class", className);
  svg.setAttribute("aria-hidden", "true");
  const path = document.createElementNS(SVG_NAMESPACE, "path");
  path.setAttribute("d", pathData);
  svg.append(path);
  return svg;
}

function createElement(tagName, className, textContent) {
  const element = document.createElement(tagName);
  if (className) element.className = className;
  if (textContent !== undefined) element.textContent = textContent;
  return element;
}

/** Tooltip for a field: what its colour means, and what was read when a name was tidied. */
function titleForRecord(record, highlighted) {
  if (record.reviewState === REVIEW_STATES.UNREAD) return "Being read…";
  if (!highlighted) return "";
  if (record.snappedFrom) return `Read as "${record.snappedFrom}"`; // exact text: test_review_flow.py
  if (record.reviewState === REVIEW_STATES.BLANK) return "Not read confidently. Type it from the check; click the field to see that part up close.";
  return "Read, but not confidently. Please check it against the check.";
}

/** A button that never takes Tab focus (Tab walks fields only). */
export function createUntabbableButton(className, text, onClick) {
  const button = createElement("button", className, text);
  button.type = "button";
  button.tabIndex = -1;
  button.addEventListener("click", onClick);
  return button;
}

/**
 * Builds one field line. `callbacks`: onInput(text), onEnter(), onTabForward(), onCommit(),
 * onFocus(), onBlur(), onEscape(), onCopy(button), onUseEmailDate(); for the payer also
 * suggestPayerNames(text) and onPickPayer(name).
 * Returns the elements plus `render(record, { emailDateSuggestion })`,
 * `setGatedText(text, snappedFromText)` and `setTextUndoably(text)`.
 */
export function createReviewFieldView(checkNumber, { key, label }, callbacks) {
  const inputId = `field-${checkNumber}-${key}`;
  const labelElement = createElement("label", "review-field-label", label);
  labelElement.htmlFor = inputId;
  const cellElement = createElement("div", "review-field-cell");
  const lineElement = createElement("div", "review-field-line");
  const input = createElement("input", "review-field-input");
  input.id = inputId;
  input.type = "text";
  input.autocomplete = "off";
  input.spellcheck = false;
  input.dataset.fieldKey = key;
  // Only the date line offers the batch's email date (spec 4.3 date handling).
  const emailDateButton = key === "date" ? createUntabbableButton("email-date-suggestion", "", callbacks.onUseEmailDate) : null;
  const noteElement = createElement("p", "review-field-note");
  noteElement.hidden = true;
  lineElement.append(input);
  if (emailDateButton) {
    emailDateButton.hidden = true;
    lineElement.append(emailDateButton);
  }
  cellElement.append(lineElement, noteElement);

  // Autocomplete first: it swallows Enter/Escape when it uses them (see its docstring).
  if (key === "payer") {
    attachPayerAutocomplete(input, lineElement, { suggestNames: callbacks.suggestPayerNames, onPick: callbacks.onPickPayer });
  }
  const undoHistory = attachFieldUndoHistory(input, (restoredText) => callbacks.onInput(restoredText));
  input.addEventListener("input", () => callbacks.onInput(input.value));
  input.addEventListener("change", () => callbacks.onCommit());
  input.addEventListener("focus", () => callbacks.onFocus());
  input.addEventListener("click", () => callbacks.onFocus()); // reopens a magnifier closed with Escape
  input.addEventListener("blur", () => callbacks.onBlur());
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      callbacks.onEnter();
    } else if (event.key === "Tab" && !event.shiftKey) {
      callbacks.onTabForward(); // no preventDefault: the browser moves focus to the next stop
    } else if (event.key === "Escape") {
      callbacks.onEscape();
    }
  });

  const copyButton = createUntabbableButton("copy-icon-button", "", () => callbacks.onCopy(copyButton));
  copyButton.append(createIcon("copy-icon-copy", COPY_ICON_PATH), createIcon("copy-icon-done", COPIED_ICON_PATH));
  copyButton.title = `Copy ${label.toLowerCase()}`;
  copyButton.setAttribute("aria-label", `Copy ${label.toLowerCase()} for check ${checkNumber}`);

  return {
    fieldKey: key,
    labelElement,
    cellElement,
    copyButton,
    inputElement: input,

    render(record, { emailDateSuggestion = null } = {}) {
      if (input.value !== record.text) input.value = record.text;
      // First value from the reader: fade it in so the change from skeleton is noticed.
      if (input.dataset.reviewState === "unread" && record.reviewState !== "unread") {
        input.classList.add("review-field-input--arrived");
        setTimeout(() => input.classList.remove("review-field-input--arrived"), FIELD_ARRIVE_ANIMATION_MS);
      }
      input.dataset.touched = String(Boolean(record.touched));
      const highlighted = isHighlighted(record);
      input.classList.toggle("review-field-input--flagged", highlighted);
      input.dataset.reviewState = record.reviewState;
      input.placeholder = record.reviewState === REVIEW_STATES.BLANK ? BLANK_FIELD_PLACEHOLDER : "";
      input.tabIndex = isReviewTabStop(record) ? 0 : -1;
      input.title = titleForRecord(record, highlighted);
      noteElement.textContent = record.note || "";
      noteElement.hidden = !record.note;
      if (!emailDateButton) return;
      emailDateButton.hidden = !emailDateSuggestion;
      if (emailDateSuggestion) {
        emailDateButton.textContent = `Use ${emailDateSuggestion}`;
        emailDateButton.title = "Use the email's date";
      }
    },

    /** A value from the reader. After a snap, Ctrl+Z goes back to what was actually read. */
    clearUndoHistory() {
      undoHistory.clearHistory();
    },
    setGatedText(text, snappedFromText) {
      if (snappedFromText && snappedFromText !== text) {
        input.value = snappedFromText;
        undoHistory.setValue(text);
      } else {
        input.value = text;
      }
    },

    /** A value set for the operator (autocomplete pick, email date) that Ctrl+Z can undo. */
    setTextUndoably(text) {
      undoHistory.setValue(text);
    },
  };
}
