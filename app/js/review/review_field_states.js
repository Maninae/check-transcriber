/**
 * One review-grid field's state, and the rules that move it (spec 4.3 "Field states" and
 * "Keyboard flow"). Pure: no DOM, no storage, so the Node unit tests cover it directly.
 *
 * A field record is `{ text, gatedState, reviewState, box, note, snappedFrom, touched }`:
 * - `text`: what the input shows. Dates show in the operator's display format; the copy
 *   value (`copyValueForField`) is always ISO, amounts always two decimals.
 * - `gatedState`: what field_gating.js said (confident / unsure / blank), or "unread"
 *   until the reads arrive. It fixes the Tab stops for the batch: unsure and blank fields
 *   (and every field of a row not read yet) are stops; confident ones are skipped.
 * - `reviewState`: what the row shows now. Unsure turns "confirmed" when the operator tabs
 *   past it, presses Enter or edits it; a blank field stays blank (amber) while empty.
 * - `touched`: the operator edited or confirmed it, so a re-read (after Rotate, or when the
 *   known-names lists change) must not overwrite it.
 */

import { FIELD_STATES } from "../fields/field_gating.js";
import { formatIsoDateForDisplay, parseDateToIso } from "../fields/date_parsing.js";
import { normalizeTypedAmount } from "../fields/money_parsing.js";

export const REVIEW_STATES = Object.freeze({
  UNREAD: "unread",
  CONFIDENT: FIELD_STATES.CONFIDENT,
  UNSURE: FIELD_STATES.UNSURE,
  BLANK: FIELD_STATES.BLANK,
  CONFIRMED: "confirmed",
});

export function createUnreadFieldRecord() {
  return { text: "", gatedState: REVIEW_STATES.UNREAD, reviewState: REVIEW_STATES.UNREAD, box: null, note: null, snappedFrom: null, touched: false };
}

/** The text an input shows for a copy-ready value (dates follow the display setting). */
export function displayTextForValue(fieldKey, value, dateDisplayFormat) {
  if (fieldKey !== "date" || !value) return value;
  const isoDate = parseDateToIso(value);
  return isoDate ? formatIsoDateForDisplay(isoDate, dateDisplayFormat) : value;
}

/** What gets copied: ISO date (or the raw text when it does not parse), two-decimal amount. */
export function copyValueForField(fieldKey, text) {
  const trimmedText = text.replace(/\s+/g, " ").trim();
  if (!trimmedText) return "";
  if (fieldKey === "date") return parseDateToIso(trimmedText) || trimmedText;
  if (fieldKey === "amount") return normalizeTypedAmount(trimmedText);
  return trimmedText;
}

/** The input text after the operator commits a field (Enter, Tab, leaving it): normalized in place. */
export function committedTextForField(fieldKey, text, dateDisplayFormat) {
  return displayTextForValue(fieldKey, copyValueForField(fieldKey, text), dateDisplayFormat);
}

/** Whether Tab and Enter stop at this field. */
export function isReviewTabStop(record) {
  return record.gatedState !== REVIEW_STATES.CONFIDENT;
}

/** Whether the field wears the amber "look here" highlight. */
export function isHighlighted(record) {
  return record.reviewState === REVIEW_STATES.UNSURE || record.reviewState === REVIEW_STATES.BLANK;
}

/** Whether focusing the field outlines its box and opens the magnifier. */
export function wantsMagnifier(record) {
  return record.gatedState === REVIEW_STATES.UNSURE || record.gatedState === REVIEW_STATES.BLANK;
}

/**
 * A new gate result for this field. An untouched field takes it whole; a touched one keeps
 * the operator's text and state and only takes the new box (still the right crop region).
 */
export function applyGatedFieldState(record, gatedField, fieldKey, dateDisplayFormat) {
  if (record.touched) return { ...record, box: gatedField.box };
  return {
    text: displayTextForValue(fieldKey, gatedField.value, dateDisplayFormat),
    gatedState: gatedField.state,
    reviewState: gatedField.state,
    box: gatedField.box,
    note: gatedField.note,
    snappedFrom: gatedField.snappedFrom,
    touched: false,
  };
}

/** The operator typed, undid, picked a suggestion or took the email date. */
export function markEdited(record, text) {
  let reviewState = REVIEW_STATES.CONFIRMED;
  if (!text.trim()) {
    const wasFlagged = record.gatedState === REVIEW_STATES.UNSURE || record.gatedState === REVIEW_STATES.BLANK;
    reviewState = wasFlagged ? REVIEW_STATES.BLANK : record.reviewState === REVIEW_STATES.UNREAD ? REVIEW_STATES.UNREAD : REVIEW_STATES.CONFIRMED;
  }
  return { ...record, text, reviewState, note: null, touched: true };
}

/** Tab past or Enter: an unsure (or filled blank) field is confirmed; an empty one stays blank. */
export function markConfirmed(record) {
  if (!isHighlighted(record) || !record.text.trim()) return record;
  return { ...record, reviewState: REVIEW_STATES.CONFIRMED, touched: true };
}

/** A box `[x0, y0, x1, y1]` in a `width` x `height` crop, after a half-turn of that crop. */
export function rotateBoxHalfTurn(box, width, height) {
  if (!box) return null;
  const [x0, y0, x1, y1] = box;
  return [width - x1, height - y1, width - x0, height - y0];
}
