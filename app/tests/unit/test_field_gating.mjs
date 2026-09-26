/**
 * Node unit tests for the field gate (js/fields/): money normalization, the amount agreement
 * rule, date parsing and the plausible window, the payee snap, the payer autocomplete snap,
 * check numbers, handwritten routing and the confident / unsure / blank thresholds.
 *
 *     node app/tests/unit/test_field_gating.mjs
 *
 * Hand-written cases only; app/tests/parity/run_gating_parity.mjs checks the same parsers and
 * the WRatio port against the Python originals on thousands of real eval strings.
 * Prints one PASS/FAIL line per check and exits non-zero on any failure.
 */

import { formatCentsAsAmount, normalizeTypedAmount, parseAmountNumericToCents, parseAmountWordsToCents } from "../../js/fields/money_parsing.js";
import { formatIsoDateAsShortMonthDay, formatIsoDateForDisplay, isIsoDateWithinWindow, parseDateAssumingYear, parseDateToIso } from "../../js/fields/date_parsing.js";
import { levenshteinDistance, normalizeFreeText, weightedRatio } from "../../js/fields/fuzzy_matching.js";
import { findPayerSnap, gateCheckFields, normalizeCheckNumberForComparison, parseCheckNumberDigits } from "../../js/fields/field_gating.js";
import { AMOUNT_DISAGREEMENT_NOTE, FIELD_CONFIDENCE_THRESHOLDS } from "../../js/fields/field_gating_config.js";

const failures = [];
function check(description, condition) {
  console.log(`[${condition ? "PASS" : "FAIL"}] ${description}`);
  if (!condition) failures.push(description);
}

const TODAY = "2026-09-26";
const printedRead = (text, confidence = 0.999, box = [10, 10, 100, 40]) => ({ text, confidence, handwrittenProbability: 0.05, reader: "crnn_general", box });
const handwrittenRead = (text, confidence = 0.999) => ({ ...printedRead(text, confidence), handwrittenProbability: 0.9 });
const trocrRead = (text, confidence) => ({ ...handwrittenRead(text, confidence), reader: "trocr" });
const gate = (rawReads, context = {}) => gateCheckFields({
  payer_name: null, payee: null, amount_numeric: null, amount_words: null, date: null, memo: null, check_number: null, ...rawReads,
}, { knownPayerNames: [], knownPayeeNames: [], todayIso: TODAY, ...context });

// Money: every courtesy convention the parser knows, two decimals, no symbol.
const courtesyCases = [["$***1,245.76", 124576], ["2,489.xx", 248900], ["2,489.-", 248900], ["2,475 00/100", 247500],
  ["694 xx/100", 69400], ["12 no/100", 1200], ["1234 =", 123400], ["681", 68100], ["$12.5", null], ["about 12", null]];
for (const [text, cents] of courtesyCases) check(`courtesy "${text}" -> ${cents}`, parseAmountNumericToCents(text) === cents);
const wordsCases = [["***FOUR HUNDRED FIFTY-THREE AND 00/100***", 45300], ["Fifteen hundred and 07/100 dollars", 150007],
  ["two thousand forty one & xx/100", 204100], ["one hundred twenty 5 cents", 12005], ["five twenty and 00/100", null],
  ["twenty twenty", null], ["one hundred 10/100 20/100", null]];
for (const [text, cents] of wordsCases) check(`legal line "${text}" -> ${cents}`, parseAmountWordsToCents(text) === cents);
check("cents format as two decimals, no symbol", formatCentsAsAmount(124576) === "1245.76" && formatCentsAsAmount(5) === "0.05");
check("typed amounts normalize ($1,250 -> 1250.00), unparseable text is kept", normalizeTypedAmount("$1,250") === "1250.00" && normalizeTypedAmount("n/a") === "n/a");

// Amount agreement (README section 5): confident only when both reads agree.
let gated = gate({ amount_numeric: printedRead("$***453.00", 0.6), amount_words: printedRead("FOUR HUNDRED FIFTY-THREE AND 00/100", 0.5) });
check("agreeing amount reads are confident whatever their confidence", gated.amount.state === "confident" && gated.amount.value === "453.00");
gated = gate({ amount_numeric: printedRead("$***453.00"), amount_words: printedRead("FOUR HUNDRED FIFTY-FOUR AND 00/100") });
check("disagreeing reads: courtesy value, unsure, with the note", gated.amount.state === "unsure" && gated.amount.value === "453.00" && gated.amount.note === AMOUNT_DISAGREEMENT_NOTE);
gated = gate({ amount_numeric: printedRead("$***453.00"), amount_words: printedRead("FOUR HUNDRD FIFTY") });
check("legal line unreadable: courtesy value unsure, no note", gated.amount.state === "unsure" && gated.amount.note === null);
gated = gate({ amount_numeric: printedRead("$**4S3"), amount_words: printedRead("FOUR HUNDRED FIFTY-THREE AND 00/100") });
check("courtesy unreadable, legal line readable: legal value, unsure", gated.amount.state === "unsure" && gated.amount.value === "453.00");
gated = gate({ amount_numeric: printedRead("$***453.00"), amount_words: handwrittenRead("FOUR HUNDRED FIFTY-FOUR AND 00/100") });
check("handwritten legal line without the handwriting reader counts as unread: unsure, no note", gated.amount.state === "unsure" && gated.amount.note === null);
gated = gate({ amount_numeric: printedRead("$***453.00"), amount_words: handwrittenRead("FOUR HUNDRED FIFTY-THREE AND 00/100") });
check("an unread handwritten legal line cannot confirm the amount", gated.amount.state === "unsure");
gated = gate({ amount_numeric: trocrRead("453.00", 0.9), amount_words: trocrRead("four hundred fifty-three and 00/100", 0.9) });
check("an agreeing amount that rests on handwriting-reader reads is capped at unsure", gated.amount.state === "unsure" && gated.amount.value === "453.00");
check("a handwriting-reader check number is capped at unsure", gate({ check_number: trocrRead("2683", 0.99) }).checkNumber.state === "unsure");
gated = gate({ amount_numeric: null, amount_words: null });
check("no amount reads: blank", gated.amount.state === "blank" && gated.amount.value === "");

// Dates: forms, ISO, window, display.
const dateCases = [["02/17/2025", "2025-02-17"], ["6/28/25", "2025-06-28"], ["8.12.26", "2026-08-12"], ["2025-06-29", "2025-06-29"],
  ["May 19, 2026", "2026-05-19"], ["Aug 3rd 2026", "2026-08-03"], ["Sept 5 2026", "2026-09-05"], ["3 August 2026", "2026-08-03"],
  ["2/30/2026", null], ["13/01/2026", null], ["Smarch 3 2026", null]];
for (const [text, iso] of dateCases) check(`date "${text}" -> ${iso}`, parseDateToIso(text) === iso);
check("window: a year either side of today", isIsoDateWithinWindow("2025-09-26", TODAY, 366) && !isIsoDateWithinWindow("2025-09-01", TODAY, 366));
check("display M/D/YYYY and ISO", formatIsoDateForDisplay("2026-09-04", "m/d/yyyy") === "9/4/2026" && formatIsoDateForDisplay("2026-09-04", "iso") === "2026-09-04");
check("a year-less email date takes the given year", parseDateAssumingYear("9/12", 2026) === "2026-09-12" && parseDateAssumingYear("Sep 12", 2026) === "2026-09-12"
  && parseDateAssumingYear("12 Sept", 2026) === "2026-09-12" && parseDateAssumingYear("9/12/2025", 2026) === "2025-09-12" && parseDateAssumingYear("soon", 2026) === null);
check("short month-day for the duplicate warning", formatIsoDateAsShortMonthDay("2026-09-12") === "Sep 12");
const dateThresholds = FIELD_CONFIDENCE_THRESHOLDS.date;
check("printed date above the filled threshold: confident ISO", gate({ date: printedRead("9/4/2026", dateThresholds.filled) }).date.state === "confident"
  && gate({ date: printedRead("9/4/2026", dateThresholds.filled) }).date.value === "2026-09-04");
check("printed date between thresholds: unsure", gate({ date: printedRead("9/4/2026", (dateThresholds.filled + dateThresholds.unsure) / 2) }).date.state === "unsure");
check("printed date below the unsure threshold: blank", gate({ date: printedRead("9/4/2026", dateThresholds.unsure - 0.01) }).date.state === "blank");
check("date outside the window: blank", gate({ date: printedRead("9/4/2024") }).date.state === "blank");
check("handwritten date without the handwriting reader: blank", gate({ date: handwrittenRead("9/4/2026") }).date.state === "blank");
check("handwritten date read by the handwriting reader: unsure at most", gate({ date: trocrRead("9/4/2026", 0.99) }).date.state === "unsure"
  && gate({ date: trocrRead("9/4/2026", 0.3) }).date.state === "blank");

// Check number: digits only, confident above its gate.
check("check number digits (prefixes and spaces dropped)", parseCheckNumberDigits("No. 2921") === "2921" && parseCheckNumberDigits("# 29 21") === "2921" && parseCheckNumberDigits("29A1") === null);
check("check number comparison strips leading zeros", normalizeCheckNumberForComparison("0012345") === "12345");
check("confident check number", gate({ check_number: printedRead("2683", 0.9) }).checkNumber.state === "confident");
check("low-confidence check number: blank", gate({ check_number: printedRead("2683", 0.3) }).checkNumber.state === "blank");

// Payer: gate, then the autocomplete snap (always unsure when it changes the text).
const payerThresholds = FIELD_CONFIDENCE_THRESHOLDS.payer_name;
check("confident printed payer", gate({ payer_name: printedRead("Lauren Copeland", payerThresholds.filled) }).payer.state === "confident");
gated = gate({ payer_name: printedRead("Lauren Copelard", payerThresholds.filled) }, { knownPayerNames: ["Lauren Copeland", "Marcus Bell"] });
check("a read one edit from a known name snaps to it, unsure, raw read kept", gated.payer.value === "Lauren Copeland" && gated.payer.state === "unsure" && gated.payer.snappedFrom === "Lauren Copelard");
gated = gate({ payer_name: printedRead("LAUREN COPELAND", payerThresholds.filled) }, { knownPayerNames: ["Lauren Copeland"] });
check("an exact (normalized) match takes the known spelling and keeps the confident state", gated.payer.value === "Lauren Copeland" && gated.payer.state === "confident");
check("a far read does not snap", findPayerSnap("Marcus Beldingham", ["Marcus Bell"]) === null);
check("the snap budget scales with length (short names: one edit)", findPayerSnap("Bob", ["Rob"]) !== null && findPayerSnap("Bob", ["Ron"]) === null);
check("handwritten payer without the handwriting reader: blank", gate({ payer_name: handwrittenRead("Lauren Copeland") }).payer.state === "blank");
check("Levenshtein distance", levenshteinDistance("kitten", "sitting") === 3 && levenshteinDistance("", "abc") === 3);
check("free-text normalization matches the harness", normalizeFreeText("  **Hello   World.  ") === "hello world");

// Payee: snapped to the co-op list, always unsure; blank without a list or a match.
const coops = ["Blue Heron Commons Land Trust", "Greenwillow Housing Cooperative", "Marrowstone Co-op Homes"];
gated = gate({ payee: printedRead("Blue Heron CLT", 0.8) }, { knownPayeeNames: coops });
check("abbreviated payee snaps to the co-op, unsure", gated.payee.value === "Blue Heron Commons Land Trust" && gated.payee.state === "unsure");
gated = gate({ payee: printedRead("Blue Heron CLT") });
check("payee without a configured list: the raw read, unsure", gated.payee.state === "unsure" && gated.payee.value === "Blue Heron CLT");
gated = gate({ payee: printedRead("Pacific Gas and Electric") }, { knownPayeeNames: coops });
check("payee below the snap score: the raw read, unsure", gated.payee.state === "unsure" && gated.payee.value === "Pacific Gas and Electric" && gated.payee.snappedFrom === null);
check("handwritten payee without the handwriting reader: blank", gate({ payee: handwrittenRead("Blue Heron CLT") }, { knownPayeeNames: coops }).payee.state === "blank");
check("WRatio: identical 100, disjoint low", weightedRatio("abc", "abc") === 100 && weightedRatio("abc", "xyz") < 10);

// Memo: printed memo is never confident.
const memoThresholds = FIELD_CONFIDENCE_THRESHOLDS.memo;
check("printed memo at full confidence is still unsure", gate({ memo: printedRead("Sept rent", 0.9999) }).memo.state === "unsure");
check("low-confidence memo: blank", gate({ memo: printedRead("Sept rent", memoThresholds.unsure - 0.01) }).memo.state === "blank");
check("handwritten memo without the handwriting reader: blank", gate({ memo: handwrittenRead("Sept rent") }).memo.state === "blank");

// Boxes travel with every state (the magnifier needs them even when blank).
check("a blank field keeps its box for the magnifier", JSON.stringify(gate({ memo: handwrittenRead("x") }).memo.box) === JSON.stringify([10, 10, 100, 40]));

if (failures.length) {
  console.log(`\n${failures.length} check(s) failed`);
  process.exit(1);
}
console.log("\nAll field-gating checks passed.");
