/**
 * The operator's settings (spec 4.3 date display, 4.5 copy columns, the payee list, the
 * opt-in handwriting reader), remembered in localStorage through `LocalStore`.
 *
 * One object, one listener list: every setter saves and then tells the listeners which
 * setting changed, so the review grid can re-render dates, re-order copies or re-gate rows
 * without knowing where the change came from (settings panel, "Clear everything").
 *
 * Copy columns are `[{ key, included }]` in paste order. Stored lists are merged with the
 * defaults on load, so a field added in a later version shows up (at the end) and a
 * removed one disappears, without anyone's saved order being lost.
 */

import { DATE_DISPLAY_FORMATS } from "../fields/date_parsing.js";
import { DEFAULT_COPY_COLUMNS } from "../review/field_definitions.js";
import { normalizeFreeText } from "../fields/fuzzy_matching.js";

const SETTINGS_KEY = "settings";
const PAYEE_NAMES_KEY = "payeeNames";

export const SETTING_NAMES = Object.freeze({
  DATE_DISPLAY_FORMAT: "dateDisplayFormat",
  COPY_COLUMNS: "copyColumns",
  PAYEE_NAMES: "payeeNames",
  HANDWRITING_READER_ENABLED: "handwritingReaderEnabled",
  EVERYTHING: "everything", // "Clear everything" reset every setting at once
});

function mergeCopyColumnsWithDefaults(storedColumns) {
  const knownKeys = new Set(DEFAULT_COPY_COLUMNS.map(({ key }) => key));
  const merged = [];
  const seenKeys = new Set();
  for (const column of Array.isArray(storedColumns) ? storedColumns : []) {
    if (!column || !knownKeys.has(column.key) || seenKeys.has(column.key)) continue;
    merged.push({ key: column.key, included: Boolean(column.included) });
    seenKeys.add(column.key);
  }
  for (const column of DEFAULT_COPY_COLUMNS) {
    if (!seenKeys.has(column.key)) merged.push({ ...column });
  }
  return merged;
}

function defaultSettingsRecord() {
  return {
    dateDisplayFormat: DATE_DISPLAY_FORMATS.ISO,
    copyColumns: DEFAULT_COPY_COLUMNS.map((column) => ({ ...column })),
    handwritingReaderEnabled: false,
  };
}

export class AppSettings {
  /** `localStore`: a storage/local_store.js `LocalStore`. */
  constructor(localStore) {
    this.localStore = localStore;
    this.listeners = [];
    this.loadFromStorage();
  }

  loadFromStorage() {
    const stored = this.localStore.readJson(SETTINGS_KEY, {}) || {};
    const defaults = defaultSettingsRecord();
    this.record = {
      dateDisplayFormat: Object.values(DATE_DISPLAY_FORMATS).includes(stored.dateDisplayFormat) ? stored.dateDisplayFormat : defaults.dateDisplayFormat,
      copyColumns: mergeCopyColumnsWithDefaults(stored.copyColumns),
      handwritingReaderEnabled: stored.handwritingReaderEnabled === true,
    };
    const storedPayees = this.localStore.readJson(PAYEE_NAMES_KEY, []);
    this.payeeNames = Array.isArray(storedPayees) ? storedPayees.filter((name) => typeof name === "string" && name.trim()) : [];
  }

  /** `listener(settingName)` after every change. */
  onChange(listener) {
    this.listeners.push(listener);
  }

  saveAndNotify(settingName) {
    this.localStore.writeJson(SETTINGS_KEY, this.record);
    this.listeners.forEach((listener) => listener(settingName));
  }

  getDateDisplayFormat() {
    return this.record.dateDisplayFormat;
  }

  setDateDisplayFormat(displayFormat) {
    this.record.dateDisplayFormat = displayFormat;
    this.saveAndNotify(SETTING_NAMES.DATE_DISPLAY_FORMAT);
  }

  /** `[{ key, included }]` in paste order (a copy; mutate through `setCopyColumns`). */
  getCopyColumns() {
    return this.record.copyColumns.map((column) => ({ ...column }));
  }

  /** Field keys to copy, in order. */
  getIncludedCopyColumnKeys() {
    return this.record.copyColumns.filter(({ included }) => included).map(({ key }) => key);
  }

  setCopyColumns(copyColumns) {
    this.record.copyColumns = mergeCopyColumnsWithDefaults(copyColumns);
    this.saveAndNotify(SETTING_NAMES.COPY_COLUMNS);
  }

  isHandwritingReaderEnabled() {
    return this.record.handwritingReaderEnabled;
  }

  setHandwritingReaderEnabled(enabled) {
    this.record.handwritingReaderEnabled = Boolean(enabled);
    this.saveAndNotify(SETTING_NAMES.HANDWRITING_READER_ENABLED);
  }

  /** The co-op payee list the payee snap uses (default empty). */
  getPayeeNames() {
    return [...this.payeeNames];
  }

  /** Adds a payee unless already listed (compared normalized); returns whether it was added. */
  addPayeeName(name) {
    const trimmedName = name.trim().replace(/\s+/g, " ");
    if (!trimmedName || this.payeeNames.some((known) => normalizeFreeText(known) === normalizeFreeText(trimmedName))) return false;
    this.payeeNames.push(trimmedName);
    this.localStore.writeJson(PAYEE_NAMES_KEY, this.payeeNames);
    this.listeners.forEach((listener) => listener(SETTING_NAMES.PAYEE_NAMES));
    return true;
  }

  removePayeeName(name) {
    this.payeeNames = this.payeeNames.filter((known) => known !== name);
    this.localStore.writeJson(PAYEE_NAMES_KEY, this.payeeNames);
    this.listeners.forEach((listener) => listener(SETTING_NAMES.PAYEE_NAMES));
  }

  /** After "Clear everything" removed the stored copies: back to defaults in memory too. */
  resetToDefaultsAfterStorageCleared() {
    this.record = defaultSettingsRecord();
    this.payeeNames = [];
    this.listeners.forEach((listener) => listener(SETTING_NAMES.EVERYTHING));
  }
}
