/**
 * What the page remembers from past batches (spec 4.3 autocomplete + duplicate warning,
 * spec 4.6 Finish batch): the payer names the operator confirmed, and one small text entry
 * per confirmed check `{ checkNumber, payer, date, batchDate }`.
 *
 * Text only, in localStorage through `LocalStore`. Names compare with the field reader's
 * `normalizeFreeText` and check numbers with `normalizeCheckNumberForComparison` (leading
 * zeros ignored), the same rules the gate uses, so "0412" and "412" are one check.
 *
 * - Only non-empty values are remembered; a check without a number is not a history entry.
 * - Both lists are capped so years of batches stay small (oldest drop first).
 */

import { normalizeFreeText } from "../fields/fuzzy_matching.js";
import { normalizeCheckNumberForComparison } from "../fields/field_gating.js";

const KNOWN_PAYER_NAMES_KEY = "knownPayerNames";
const CHECK_HISTORY_KEY = "checkHistory";
const MAXIMUM_KNOWN_PAYER_NAMES = 1000;
const MAXIMUM_CHECK_HISTORY_ENTRIES = 5000;
const MINIMUM_AUTOCOMPLETE_PREFIX_LENGTH = 2;
const MAXIMUM_AUTOCOMPLETE_SUGGESTIONS = 6;

function isNonEmptyString(value) {
  return typeof value === "string" && value.trim().length > 0;
}

export class BatchHistory {
  /** `localStore`: a storage/local_store.js `LocalStore`. */
  constructor(localStore) {
    this.localStore = localStore;
    this.cachedCheckHistory = null; // the duplicate check runs on every keystroke
  }

  /** Drops the cached copy (after "Clear everything" wiped storage underneath). */
  forgetCachedCopies() {
    this.cachedCheckHistory = null;
  }

  /** Confirmed payer names, oldest first. */
  getKnownPayerNames() {
    const names = this.localStore.readJson(KNOWN_PAYER_NAMES_KEY, []);
    return Array.isArray(names) ? names.filter(isNonEmptyString) : [];
  }

  /** Adds names not already known (compared normalized); returns how many were new. */
  addKnownPayerNames(names) {
    const knownNames = this.getKnownPayerNames();
    const knownNormalized = new Set(knownNames.map(normalizeFreeText));
    let addedCount = 0;
    for (const name of names.filter(isNonEmptyString)) {
      const trimmedName = name.trim().replace(/\s+/g, " ");
      const normalizedName = normalizeFreeText(trimmedName);
      if (!normalizedName || knownNormalized.has(normalizedName)) continue;
      knownNormalized.add(normalizedName);
      knownNames.push(trimmedName);
      addedCount += 1;
    }
    this.localStore.writeJson(KNOWN_PAYER_NAMES_KEY, knownNames.slice(-MAXIMUM_KNOWN_PAYER_NAMES));
    return addedCount;
  }

  removeKnownPayerName(name) {
    const normalizedName = normalizeFreeText(name);
    this.localStore.writeJson(KNOWN_PAYER_NAMES_KEY, this.getKnownPayerNames().filter((known) => normalizeFreeText(known) !== normalizedName));
  }

  clearKnownPayerNames() {
    this.localStore.writeJson(KNOWN_PAYER_NAMES_KEY, []);
  }

  /**
   * Known names matching what the operator has typed so far: after two letters, names
   * whose start, or the start of any word, matches the typed text. Exact matches are left
   * out (nothing to complete).
   */
  suggestPayerNames(typedText) {
    const typedNormalized = normalizeFreeText(typedText);
    if (typedNormalized.length < MINIMUM_AUTOCOMPLETE_PREFIX_LENGTH) return [];
    const startMatches = [];
    const wordMatches = [];
    for (const name of this.getKnownPayerNames()) {
      const normalizedName = normalizeFreeText(name);
      if (normalizedName === typedNormalized) continue;
      if (normalizedName.startsWith(typedNormalized)) startMatches.push(name);
      else if (normalizedName.split(" ").some((word) => word.startsWith(typedNormalized))) wordMatches.push(name);
    }
    return [...startMatches, ...wordMatches].slice(0, MAXIMUM_AUTOCOMPLETE_SUGGESTIONS);
  }

  getCheckHistory() {
    if (this.cachedCheckHistory === null) {
      const entries = this.localStore.readJson(CHECK_HISTORY_KEY, []);
      this.cachedCheckHistory = Array.isArray(entries) ? entries.filter((entry) => entry && isNonEmptyString(entry.checkNumber)) : [];
    }
    return this.cachedCheckHistory;
  }

  /** Appends `[{ checkNumber, payer, date }]` from one finished batch dated `batchDateIso`. */
  recordConfirmedChecks(checkRecords, batchDateIso) {
    const newEntries = [];
    for (const { checkNumber, payer, date } of checkRecords) {
      if (!isNonEmptyString(checkNumber)) continue;
      const entry = { checkNumber: checkNumber.trim(), batchDate: batchDateIso };
      if (isNonEmptyString(payer)) entry.payer = payer.trim();
      if (isNonEmptyString(date)) entry.date = date.trim();
      newEntries.push(entry);
    }
    const history = [...this.getCheckHistory(), ...newEntries].slice(-MAXIMUM_CHECK_HISTORY_ENTRIES);
    this.localStore.writeJson(CHECK_HISTORY_KEY, history);
    this.cachedCheckHistory = history;
    return newEntries.length;
  }

  /**
   * The most recent past batch date in which this check number was seen from this payer,
   * or null. Both must be present: a check number alone repeats across payers.
   */
  findPreviousBatchDate(checkNumber, payer) {
    const wantedNumber = isNonEmptyString(checkNumber) ? normalizeCheckNumberForComparison(checkNumber) : null;
    const wantedPayer = isNonEmptyString(payer) ? normalizeFreeText(payer) : "";
    if (wantedNumber === null || !wantedPayer) return null;
    let latestBatchDate = null;
    for (const entry of this.getCheckHistory()) {
      if (!isNonEmptyString(entry.payer) || normalizeFreeText(entry.payer) !== wantedPayer) continue;
      if (normalizeCheckNumberForComparison(entry.checkNumber) !== wantedNumber) continue;
      if (latestBatchDate === null || entry.batchDate > latestBatchDate) latestBatchDate = entry.batchDate;
    }
    return latestBatchDate;
  }
}
