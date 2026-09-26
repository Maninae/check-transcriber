/**
 * Check dates to ISO `YYYY-MM-DD`, and back out for display. Parsing is a line-for-line port
 * of experiments/field_reading/metrics/date_parsing.py (US month-first for all-numeric dates,
 * two-digit years mean 20YY); change the Python first, then mirror it here.
 *
 * Forms: `02/17/2025`, `6/28/25`, `8.12.26`, `2025-06-29`, `May 19, 2026`, `Aug 3rd 2026`,
 * `3 August 2026`. Impossible calendar dates (`2/30/2026`) give null.
 */

const TWO_DIGIT_YEAR_CENTURY = 2000;
const FULL_MONTH_NAMES = [
  "january", "february", "march", "april", "may", "june",
  "july", "august", "september", "october", "november", "december",
];
const MILLISECONDS_PER_DAY = 86400000;

const YEAR_FIRST_NUMERIC_PATTERN = /^(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})$/;
const MONTH_FIRST_NUMERIC_PATTERN = /^(\d{1,2})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{2}|\d{4})$/;
const DAY_WITH_ORDINAL = String.raw`(\d{1,2})(?:st|nd|rd|th)?`;
const MONTH_WORD = String.raw`([a-z]{3,9})\.?`;
const MONTH_NAME_FIRST_PATTERN = new RegExp(`^${MONTH_WORD}\\s*${DAY_WITH_ORDINAL}\\s*,?\\s*(\\d{4}|\\d{2})$`);
const DAY_FIRST_MONTH_NAME_PATTERN = new RegExp(`^${DAY_WITH_ORDINAL}\\s+${MONTH_WORD}\\s*,?\\s*(\\d{4}|\\d{2})$`);

/** `aug`, `august`, `sept` -> 8, 8, 9 (any 3+ letter prefix of a month name), else null. */
export function monthWordToNumber(monthWord) {
  if (monthWord.length < 3) return null;
  const monthIndex = FULL_MONTH_NAMES.findIndex((fullName) => fullName.startsWith(monthWord));
  return monthIndex < 0 ? null : monthIndex + 1;
}

function expandYear(yearText) {
  const yearValue = parseInt(yearText, 10);
  return yearText.length === 2 ? yearValue + TWO_DIGIT_YEAR_CENTURY : yearValue;
}

function isoDateOrNull(yearValue, monthValue, dayValue) {
  if (monthValue === null || yearValue < 1 || yearValue > 9999 || monthValue < 1 || monthValue > 12 || dayValue < 1) return null;
  const date = new Date(Date.UTC(yearValue, monthValue - 1, dayValue));
  date.setUTCFullYear(yearValue); // Date.UTC maps years 0-99 to 1900-1999
  if (date.getUTCMonth() !== monthValue - 1 || date.getUTCDate() !== dayValue) return null;
  return `${String(yearValue).padStart(4, "0")}-${String(monthValue).padStart(2, "0")}-${String(dayValue).padStart(2, "0")}`;
}

/** Date text as written on a check -> `YYYY-MM-DD`, or null. */
export function parseDateToIso(dateText) {
  const compactText = dateText.toLowerCase().replaceAll("*", " ").split(/\s+/).filter(Boolean).join(" ")
    .replace(/^[ ,]+|[ ,]+$/g, "");
  let match = YEAR_FIRST_NUMERIC_PATTERN.exec(compactText);
  if (match) return isoDateOrNull(parseInt(match[1], 10), parseInt(match[2], 10), parseInt(match[3], 10));
  match = MONTH_FIRST_NUMERIC_PATTERN.exec(compactText);
  if (match) return isoDateOrNull(expandYear(match[3]), parseInt(match[1], 10), parseInt(match[2], 10));
  match = MONTH_NAME_FIRST_PATTERN.exec(compactText);
  if (match) return isoDateOrNull(expandYear(match[3]), monthWordToNumber(match[1]), parseInt(match[2], 10));
  match = DAY_FIRST_MONTH_NAME_PATTERN.exec(compactText);
  if (match) return isoDateOrNull(expandYear(match[3]), monthWordToNumber(match[2]), parseInt(match[1], 10));
  return null;
}

/** Local calendar date of `now` as ISO (the operator's today, not UTC's). */
export function todayIso(now = new Date()) {
  return `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}-${String(now.getDate()).padStart(2, "0")}`;
}

/** Whether an ISO date lies within `windowDays` of the ISO `referenceIso` (spec 5.7: a year either side of today). */
export function isIsoDateWithinWindow(isoDate, referenceIso, windowDays) {
  const differenceDays = Math.abs(Date.parse(`${isoDate}T00:00:00Z`) - Date.parse(`${referenceIso}T00:00:00Z`)) / MILLISECONDS_PER_DAY;
  return differenceDays <= windowDays;
}

export const DATE_DISPLAY_FORMATS = Object.freeze({ ISO: "iso", MONTH_DAY_YEAR: "m/d/yyyy" });

/** ISO -> the operator's display format (`2026-09-04` -> `9/4/2026` for m/d/yyyy). */
export function formatIsoDateForDisplay(isoDate, displayFormat) {
  if (displayFormat !== DATE_DISPLAY_FORMATS.MONTH_DAY_YEAR) return isoDate;
  const [year, month, day] = isoDate.split("-").map((part) => parseInt(part, 10));
  return `${month}/${day}/${year}`;
}

/** Short human date for "Seen before: batch on Sep 12." */
export function formatIsoDateAsShortMonthDay(isoDate) {
  const [, month, day] = isoDate.split("-").map((part) => parseInt(part, 10));
  const shortMonth = FULL_MONTH_NAMES[month - 1].slice(0, 3);
  return `${shortMonth[0].toUpperCase()}${shortMonth.slice(1)} ${day}`;
}
