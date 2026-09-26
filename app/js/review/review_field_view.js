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
const COPY_ICON = "⧉"; // two joined squares, the usual copy glyph

function createElement(tagName, className, textContent) {
  const element = document.createElement(tagName);
  if (className) element.className = className;
  if (textContent !== undefined) element.textContent = textContent;
  return element;
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

  const copyButton = createUntabbableButton("copy-icon-button", COPY_ICON, () => callbacks.onCopy(copyButton));
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
      const highlighted = isHighlighted(record);
      input.classList.toggle("review-field-input--flagged", highlighted);
      input.dataset.reviewState = record.reviewState;
      input.placeholder = record.reviewState === REVIEW_STATES.BLANK ? BLANK_FIELD_PLACEHOLDER : "";
      input.tabIndex = isReviewTabStop(record) ? 0 : -1;
      input.title = highlighted && record.snappedFrom ? `Read as "${record.snappedFrom}"` : "";
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
