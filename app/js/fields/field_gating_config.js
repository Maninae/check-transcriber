/**
 * Every threshold the field gate uses, in one place.
 *
 * PROVISIONAL. These come from synthetic validation checks (experiments/field_reading,
 * val-frozen thresholds of the shipped CRNN + segnet reader, reported in
 * field-reading/reports/final/final_comparison.json). The field-reading README shows such
 * thresholds do not transfer (95% on val gave 88-94% on eval), so the gold set of real
 * phone photos must re-tune every number here before anyone trusts the confident state.
 *
 * Two thresholds per field implement spec section 5 stage 7: at or above the "filled"
 * threshold a read is shown confident; at or above the "unsure" threshold it is shown with
 * the amber highlight; below that it is left blank. Filled = the val threshold for 98%
 * accuracy, unsure = the one for 95%.
 */

export const FIELD_CONFIDENCE_THRESHOLDS = Object.freeze({
  // CRNN mean-character confidence; val 98% and 95% (identical: val accuracy never dips).
  check_number: { filled: 0.5392, unsure: 0.5392 },
  payer_name: { filled: 0.9895, unsure: 0.9691 },
  date: { filled: 0.9821, unsure: 0.9548 },
  // Printed memo is never shown confident (README section 5: "printed: unsure").
  memo: { filled: Infinity, unsure: 0.9603 },
});

// Style classifier: p(handwritten) above this routes the field to the handwriting path.
export const HANDWRITTEN_PROBABILITY_THRESHOLD = 0.5;

// TrOCR (opt-in "read handwriting") reads are never shown confident: no real handwriting
// set exists to tune a confident threshold on (synthetic handwriting measures font recall).
// Reads at or above this sequence confidence are shown unsure, the rest blank.
export const HANDWRITING_READ_UNSURE_MIN_CONFIDENCE = 0.6;

// Payee: rapidfuzz WRatio against the configured co-op list; payee_snap.py's SNAP_MIN_SCORE.
export const PAYEE_SNAP_MIN_SCORE = 60;

// Payer autocomplete snap (spec 4.3): an OCR read within this many edits of a known name,
// scaled with length (1 edit per 8 characters, at least 1, at most 3), snaps to it.
export const PAYER_SNAP_EDITS_PER_CHARACTER = 1 / 8;
export const PAYER_SNAP_MAXIMUM_EDITS = 3;

// Spec 5.7: a date must fall within a year either side of today.
export const PLAUSIBLE_DATE_WINDOW_DAYS = 366;

// Note shown under an unsure amount when the two amount reads disagree (spec 4.3).
export const AMOUNT_DISAGREEMENT_NOTE = "the written amount reads differently";
