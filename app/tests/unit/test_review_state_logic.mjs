/**
 * Node unit tests for the review grid's pure logic: the namespaced local store, the payer
 * names and check history (autocomplete, duplicate lookup), the settings model, and the
 * field-state rules (touched fields survive re-reads, confirm on Tab, copy values).
 *
 *     node app/tests/unit/test_review_state_logic.mjs
 *
 * Prints one PASS/FAIL line per check and exits non-zero on any failure.
 */

import { LocalStore } from "../../js/storage/local_store.js";
import { BatchHistory } from "../../js/storage/batch_history.js";
import { AppSettings, SETTING_NAMES } from "../../js/settings/app_settings.js";
import {
  applyGatedFieldState,
  copyValueForField,
  committedTextForField,
  createUnreadFieldRecord,
  isHighlighted,
  isReviewTabStop,
  markConfirmed,
  markEdited,
  REVIEW_STATES,
  rotateBoxHalfTurn,
} from "../../js/review/review_field_states.js";
import { magnifierRegionForField } from "../../js/review/field_magnifier.js";

const failures = [];
function check(description, condition) {
  console.log(`[${condition ? "PASS" : "FAIL"}] ${description}`);
  if (!condition) failures.push(description);
}

/** A Map-backed stand-in for window.localStorage. */
class MemoryStorage {
  constructor() { this.items = new Map(); }
  get length() { return this.items.size; }
  key(index) { return [...this.items.keys()][index] ?? null; }
  getItem(key) { return this.items.has(key) ? this.items.get(key) : null; }
  setItem(key, value) { this.items.set(key, String(value)); }
  removeItem(key) { this.items.delete(key); }
}

// Local store
const memoryStorage = new MemoryStorage();
memoryStorage.setItem("someOtherApp.setting", "keep me");
const localStore = new LocalStore(memoryStorage);
localStore.writeJson("probe", { a: [1, "two"] });
check("values live under the checkTranscriber.v1. prefix", memoryStorage.getItem("checkTranscriber.v1.probe") === '{"a":[1,"two"]}');
let refusedPixels = false;
try { localStore.writeJson("pixels", { data: new Uint8ClampedArray(4) }); } catch { refusedPixels = true; }
check("a typed array (pixel buffer) is refused", refusedPixels);
memoryStorage.setItem("checkTranscriber.v1.broken", "{not json");
check("corrupt JSON reads as the default", localStore.readJson("broken", "fallback") === "fallback");

// Payer names and history
const batchHistory = new BatchHistory(localStore);
check("new names are added once", batchHistory.addKnownPayerNames(["Lauren Copeland", "lauren copeland ", "", "Marcus  Bell"]) === 2);
check("names keep their spelling, whitespace collapsed", JSON.stringify(batchHistory.getKnownPayerNames()) === '["Lauren Copeland","Marcus Bell"]');
check("one letter offers nothing", batchHistory.suggestPayerNames("L").length === 0);
check("two letters offer matches", JSON.stringify(batchHistory.suggestPayerNames("la")) === '["Lauren Copeland"]');
check("a later word matches too", JSON.stringify(batchHistory.suggestPayerNames("bel")) === '["Marcus Bell"]');
batchHistory.recordConfirmedChecks([
  { checkNumber: "0412", payer: "Lauren Copeland", date: "2026-09-01" },
  { checkNumber: "", payer: "Nobody", date: "" },
], "2026-09-12");
check("a check without a number is not remembered", batchHistory.getCheckHistory().length === 1);
check("duplicate found ignoring leading zeros and case", batchHistory.findPreviousBatchDate("412", "LAUREN COPELAND") === "2026-09-12");
check("same number, other payer: no duplicate", batchHistory.findPreviousBatchDate("412", "Marcus Bell") === null);
batchHistory.removeKnownPayerName("marcus bell");
check("removing a name", JSON.stringify(batchHistory.getKnownPayerNames()) === '["Lauren Copeland"]');

// Settings
const settings = new AppSettings(localStore);
const changes = [];
settings.onChange((name) => changes.push(name));
check("default copy order is Date, Payer, Amount, Check number, Memo", JSON.stringify(settings.getIncludedCopyColumnKeys()) === '["date","payer","amount","checkNumber","memo"]');
check("handwriting reader is off by default", settings.isHandwritingReaderEnabled() === false);
settings.setCopyColumns([{ key: "amount", included: true }, { key: "payer", included: false }]);
check("a saved order is merged with the defaults", JSON.stringify(settings.getIncludedCopyColumnKeys()) === '["amount","date","checkNumber","memo"]');
check("settings survive a reload", JSON.stringify(new AppSettings(localStore).getIncludedCopyColumnKeys()) === '["amount","date","checkNumber","memo"]');
settings.addPayeeName("Oak Street Co-op");
check("payee list saved and listeners told", new AppSettings(localStore).getPayeeNames()[0] === "Oak Street Co-op" && changes.includes(SETTING_NAMES.PAYEE_NAMES));
const removedCount = localStore.removeEveryOwnedKey();
check(`Clear everything removes only this app's keys (${removedCount})`, memoryStorage.length === 1 && memoryStorage.getItem("someOtherApp.setting") === "keep me");

// Field states
const unsureGate = { state: "unsure", value: "Lauren Copeland", box: [10, 20, 110, 60], note: null, snappedFrom: "Lauren Copelnad" };
let record = applyGatedFieldState(createUnreadFieldRecord(), unsureGate, "payer", "iso");
check("an unsure read is highlighted and a Tab stop", isHighlighted(record) && isReviewTabStop(record));
record = markConfirmed(record);
check("Tab past confirms it (no highlight, touched)", record.reviewState === REVIEW_STATES.CONFIRMED && record.touched && !isHighlighted(record));
check("still a Tab stop after confirming (Shift+Tab can come back)", isReviewTabStop(record));
const reread = applyGatedFieldState(record, { ...unsureGate, value: "Someone Else", box: [1, 2, 3, 4] }, "payer", "iso");
check("a re-read keeps a confirmed field's text, takes the new box", reread.text === "Lauren Copeland" && reread.box[0] === 1);
let blank = applyGatedFieldState(createUnreadFieldRecord(), { state: "blank", value: "", box: null, note: null, snappedFrom: null }, "date", "iso");
check("confirming an empty blank field leaves it blank", markConfirmed(blank).reviewState === REVIEW_STATES.BLANK);
blank = markEdited(blank, "9/4/26");
check("typing into a blank field confirms it", blank.reviewState === REVIEW_STATES.CONFIRMED && blank.touched);
check("clearing it again brings the blank highlight back", markEdited(blank, "").reviewState === REVIEW_STATES.BLANK);
const confident = applyGatedFieldState(createUnreadFieldRecord(), { state: "confident", value: "2026-09-04", box: null, note: null, snappedFrom: null }, "date", "m/d/yyyy");
check("a confident date shows in the display format and is skipped by Tab", confident.text === "9/4/2026" && !isReviewTabStop(confident));
check("copy value of a displayed date is ISO", copyValueForField("date", "9/4/2026") === "2026-09-04");
check("an unparseable typed date is copied as typed", copyValueForField("date", "early Sept") === "early Sept");
check("typed amounts commit to two decimals", committedTextForField("amount", "$1,250", "iso") === "1250.00");
check("a half-turn maps a box to the opposite corner", JSON.stringify(rotateBoxHalfTurn([10, 20, 110, 60], 1600, 700)) === "[1490,640,1590,680]");
check("the fallback magnifier region for amount", JSON.stringify(magnifierRegionForField("amount", null, 1600, 700)) === JSON.stringify([1235, 267, 1547, 375]));

if (failures.length) {
  console.log(`${failures.length} check(s) failed`);
  process.exit(1);
}
console.log("All checks passed.");
