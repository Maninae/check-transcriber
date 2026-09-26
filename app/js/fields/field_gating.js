/**
 * Stage 7 of the pipeline (spec section 5): turns the worker's raw field reads for one
 * check into what the review grid shows, one state per field:
 *   confident -> plain text;  unsure -> amber, value shown;  blank -> amber, empty.
 *
 * Rules (experiments/field_reading/README.md section 5, followed exactly):
 * - check number: CRNN, filled when confident and all digits.
 * - payer (printed): CRNN confidence gate, then the autocomplete snap to confirmed names
 *   (a snap is always unsure). Handwritten payer: blank (unsure via the opt-in reader).
 * - amount: filled ONLY when the courtesy box and the legal line parse to the same money
 *   value; otherwise the courtesy value is shown unsure, with a note when the legal line
 *   parsed to something else. The legal line itself is never shown.
 * - payee: snapped to the configured co-op list when it matches, else the raw read; always
 *   unsure (a sanity check, not copied by default).
 * - date (printed): confidence gate plus the plausible window; handwritten: blank.
 * - memo (printed): at best unsure; handwritten: blank.
 * With the opt-in handwriting reader, every handwritten field (check number, legal line and
 * payee included) is read by it and gated the same way, but any value resting on one of its
 * reads is capped at unsure (field_gating_config.js explains why). With the reader off, a
 * handwritten legal line counts as unread, so it neither confirms nor contradicts the amount.
 *
 * Pure: no DOM, no storage. Runs on the main thread (and in Node for the unit tests), so
 * the grid can re-gate when the known-names lists change without re-reading pixels.
 */

import {
  AMOUNT_DISAGREEMENT_NOTE,
  FIELD_CONFIDENCE_THRESHOLDS,
  HANDWRITING_READ_UNSURE_MIN_CONFIDENCE,
  HANDWRITTEN_PROBABILITY_THRESHOLD,
  PAYEE_SNAP_MIN_SCORE,
  PAYER_SNAP_EDITS_PER_CHARACTER,
  PAYER_SNAP_MAXIMUM_EDITS,
  PLAUSIBLE_DATE_WINDOW_DAYS,
} from "./field_gating_config.js";
import { formatCentsAsAmount, parseAmountNumericToCents, parseAmountWordsToCents } from "./money_parsing.js";
import { isIsoDateWithinWindow, parseDateToIso, todayIso } from "./date_parsing.js";
import { bestWeightedRatioMatch, levenshteinDistance, normalizeFreeText } from "./fuzzy_matching.js";

export const FIELD_STATES = Object.freeze({ CONFIDENT: "confident", UNSURE: "unsure", BLANK: "blank" });

// Which reader produced a raw read (set by the worker, pipeline/fields/).
export const READERS = Object.freeze({ PRINTED: "crnn_general", AMOUNT: "crnn_amount", HANDWRITING: "trocr" });

const CHECK_NUMBER_PREFIX_PATTERN = /^(?:no\.?|#|nº)\s*/;

/** experiments' `parse_check_number_digits`, minus the leading-zero strip: digits as printed, or null. */
export function parseCheckNumberDigits(checkNumberText) {
  const compactText = checkNumberText.trim().toLowerCase().replace(CHECK_NUMBER_PREFIX_PATTERN, "").replaceAll(" ", "");
  return /^[0-9]+$/.test(compactText) ? compactText : null;
}

/** Check number as compared for duplicates: digits without leading zeros. */
export function normalizeCheckNumberForComparison(checkNumberText) {
  const digits = parseCheckNumberDigits(checkNumberText);
  return digits === null ? null : digits.replace(/^0+/, "") || "0";
}

function makeFieldState(state, value, rawRead, extra = {}) {
  return { state, value: state === FIELD_STATES.BLANK ? "" : value, box: rawRead ? rawRead.box : null, note: null, snappedFrom: null, ...extra };
}

function isHandwritten(rawRead) {
  return Boolean(rawRead) && rawRead.handwrittenProbability > HANDWRITTEN_PROBABILITY_THRESHOLD;
}

function hasText(rawRead) {
  return Boolean(rawRead) && Boolean(rawRead.text) && rawRead.text.trim().length > 0;
}

/**
 * The confident / unsure / blank call for a read against a field's two thresholds.
 * Handwriting-reader reads are capped at unsure (see field_gating_config.js).
 */
function stateFromConfidence(rawRead, thresholds) {
  if (!hasText(rawRead)) return FIELD_STATES.BLANK;
  if (rawRead.reader === READERS.HANDWRITING) {
    return rawRead.confidence >= HANDWRITING_READ_UNSURE_MIN_CONFIDENCE ? FIELD_STATES.UNSURE : FIELD_STATES.BLANK;
  }
  if (rawRead.confidence >= thresholds.filled) return FIELD_STATES.CONFIDENT;
  if (rawRead.confidence >= thresholds.unsure) return FIELD_STATES.UNSURE;
  return FIELD_STATES.BLANK;
}

/** A handwritten field with no handwriting read is blank by design (README section 5). */
function isUnreadHandwriting(rawRead) {
  return isHandwritten(rawRead) && rawRead.reader !== READERS.HANDWRITING;
}

function gateCheckNumber(rawRead) {
  const digits = hasText(rawRead) ? parseCheckNumberDigits(rawRead.text) : null;
  if (digits === null || isUnreadHandwriting(rawRead)) return makeFieldState(FIELD_STATES.BLANK, "", rawRead);
  return makeFieldState(stateFromConfidence(rawRead, FIELD_CONFIDENCE_THRESHOLDS.check_number), digits, rawRead);
}

/**
 * The known payer name an OCR read should snap to, or null: exact (normalized) matches keep
 * the read's own state; near matches within the edit budget snap and turn unsure.
 */
export function findPayerSnap(readText, knownPayerNames) {
  const normalizedRead = normalizeFreeText(readText);
  if (!normalizedRead) return null;
  const allowedEdits = Math.min(PAYER_SNAP_MAXIMUM_EDITS, Math.max(1, Math.floor(normalizedRead.length * PAYER_SNAP_EDITS_PER_CHARACTER)));
  let best = null;
  for (const knownName of knownPayerNames) {
    const distance = levenshteinDistance(normalizedRead, normalizeFreeText(knownName));
    if (distance <= allowedEdits && (best === null || distance < best.distance)) best = { knownName, distance };
  }
  return best;
}

function gatePayer(rawRead, knownPayerNames) {
  if (isUnreadHandwriting(rawRead)) return makeFieldState(FIELD_STATES.BLANK, "", rawRead);
  const state = stateFromConfidence(rawRead, FIELD_CONFIDENCE_THRESHOLDS.payer_name);
  if (state === FIELD_STATES.BLANK) return makeFieldState(state, "", rawRead);
  const readText = rawRead.text.trim();
  const snap = findPayerSnap(readText, knownPayerNames);
  if (!snap) return makeFieldState(state, readText, rawRead);
  if (snap.distance === 0) return makeFieldState(state, snap.knownName, rawRead);
  return makeFieldState(FIELD_STATES.UNSURE, snap.knownName, rawRead, { snappedFrom: readText });
}

/** A read the style classifier called handwritten but only the default readers saw: treated as not read. */
function readOrNullWhenUnreadHandwriting(rawRead) {
  return isUnreadHandwriting(rawRead) ? null : rawRead;
}

/**
 * Spec 4.3 amount handling + the README agreement rule (see module docstring). A handwritten
 * legal line the handwriting reader did not read counts as absent (no agreement, no note).
 * Any amount that rests on a handwriting-reader read is capped at unsure.
 */
function gateAmount(numericRead, wordsReadOrUnread) {
  const wordsRead = readOrNullWhenUnreadHandwriting(wordsReadOrUnread);
  const numericCents = hasText(numericRead) ? parseAmountNumericToCents(numericRead.text) : null;
  const wordsCents = hasText(wordsRead) ? parseAmountWordsToCents(wordsRead.text) : null;
  const usesHandwritingReader = [numericRead, wordsRead].some((read) => read && read.reader === READERS.HANDWRITING);
  if (numericCents !== null && numericCents === wordsCents) {
    const state = usesHandwritingReader ? FIELD_STATES.UNSURE : FIELD_STATES.CONFIDENT;
    return makeFieldState(state, formatCentsAsAmount(numericCents), numericRead);
  }
  if (numericCents !== null) {
    const note = wordsCents !== null ? AMOUNT_DISAGREEMENT_NOTE : null;
    return makeFieldState(FIELD_STATES.UNSURE, formatCentsAsAmount(numericCents), numericRead, { note });
  }
  if (wordsCents !== null) {
    // The courtesy box did not parse but the legal line did (amount_cross_check.py recovers it the same way).
    return makeFieldState(FIELD_STATES.UNSURE, formatCentsAsAmount(wordsCents), numericRead);
  }
  return makeFieldState(FIELD_STATES.BLANK, "", numericRead);
}

/**
 * Payee: snapped to the co-op list when it matches (WRatio >= 60), otherwise the raw read;
 * always unsure. Blank only when there is nothing to show (no read, or unread handwriting).
 */
function gatePayee(rawRead, knownPayeeNames) {
  if (!hasText(rawRead) || isUnreadHandwriting(rawRead)) return makeFieldState(FIELD_STATES.BLANK, "", rawRead);
  const readText = rawRead.text.trim();
  const match = knownPayeeNames.length ? bestWeightedRatioMatch(readText, knownPayeeNames) : null;
  if (!match || match.score < PAYEE_SNAP_MIN_SCORE) return makeFieldState(FIELD_STATES.UNSURE, readText, rawRead);
  return makeFieldState(FIELD_STATES.UNSURE, match.choice, rawRead, { snappedFrom: readText });
}

function gateDate(rawRead, referenceIso) {
  if (isUnreadHandwriting(rawRead)) return makeFieldState(FIELD_STATES.BLANK, "", rawRead);
  const isoDate = hasText(rawRead) ? parseDateToIso(rawRead.text) : null;
  if (isoDate === null || !isIsoDateWithinWindow(isoDate, referenceIso, PLAUSIBLE_DATE_WINDOW_DAYS)) {
    return makeFieldState(FIELD_STATES.BLANK, "", rawRead);
  }
  return makeFieldState(stateFromConfidence(rawRead, FIELD_CONFIDENCE_THRESHOLDS.date), isoDate, rawRead);
}

function gateMemo(rawRead) {
  if (isUnreadHandwriting(rawRead)) return makeFieldState(FIELD_STATES.BLANK, "", rawRead);
  const state = stateFromConfidence(rawRead, FIELD_CONFIDENCE_THRESHOLDS.memo);
  return makeFieldState(state, state === FIELD_STATES.BLANK ? "" : rawRead.text.trim(), rawRead);
}

/**
 * Gates one check's reads.
 *
 * Args:
 *   rawReads: `{ payer_name, payee, amount_numeric, amount_words, date, memo, check_number }`,
 *     each `{ text, confidence, handwrittenProbability, reader, box } | null` (null = no box found).
 *   context: `{ knownPayerNames: string[], knownPayeeNames: string[], todayIso?: string }`.
 * Returns:
 *   `{ payer, amount, date, checkNumber, memo, payee }`, each
 *   `{ state, value, box, note, snappedFrom }`. `value` is copy-ready: amount "1234.56",
 *   date ISO, check number digits. `box` is the crop region ([x0, y0, x1, y1] in upright crop
 *   pixels) the value came from, or null; `snappedFrom` is the raw read when a snap replaced it.
 */
export function gateCheckFields(rawReads, context) {
  const referenceIso = context.todayIso || todayIso();
  return {
    payer: gatePayer(rawReads.payer_name, context.knownPayerNames || []),
    amount: gateAmount(rawReads.amount_numeric, rawReads.amount_words),
    date: gateDate(rawReads.date, referenceIso),
    checkNumber: gateCheckNumber(rawReads.check_number),
    memo: gateMemo(rawReads.memo),
    payee: gatePayee(rawReads.payee, context.knownPayeeNames || []),
  };
}
