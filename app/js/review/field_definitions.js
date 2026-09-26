/**
 * The fields beside each crop, and the default column order used when copying rows.
 *
 * Display order follows the check's layout (spec 4.3), payee last: it is a sanity check
 * (who the check is written to), shown but not copied by default. The copy default is the
 * spec 4.5 order (Date, Payer, Amount, Check number, Memo); the settings panel lets the
 * operator reorder and include or drop columns (settings/app_settings.js keeps that).
 */

export const REVIEW_FIELDS = Object.freeze([
  { key: "payer", label: "Payer" },
  { key: "amount", label: "Amount" },
  { key: "date", label: "Date" },
  { key: "checkNumber", label: "Check number" },
  { key: "memo", label: "Memo" },
  { key: "payee", label: "Payee" },
]);

export const FIELD_LABELS = Object.freeze(Object.fromEntries(REVIEW_FIELDS.map(({ key, label }) => [key, label])));

export const DEFAULT_COPY_COLUMNS = Object.freeze([
  Object.freeze({ key: "date", included: true }),
  Object.freeze({ key: "payer", included: true }),
  Object.freeze({ key: "amount", included: true }),
  Object.freeze({ key: "checkNumber", included: true }),
  Object.freeze({ key: "memo", included: true }),
  Object.freeze({ key: "payee", included: false }),
]);

export const DEFAULT_COPY_COLUMN_ORDER = Object.freeze(DEFAULT_COPY_COLUMNS.filter(({ included }) => included).map(({ key }) => key));

/**
 * Where each field usually sits on a personal check, as fractions of the upright crop
 * (x0, y0, x1, y1): the 'personal' layout search regions from experiments/field_reading's
 * layout priors. The inline magnifier falls back to these when a field has no read box.
 */
export const FALLBACK_FIELD_REGIONS = Object.freeze({
  payer: [0.034, 0.055, 0.73, 0.201],
  payee: [0.152, 0.364, 0.784, 0.573],
  amount: [0.772, 0.381, 0.967, 0.535],
  date: [0.636, 0.215, 0.96, 0.385],
  memo: [0.1, 0.71, 0.507, 0.882],
  checkNumber: [0.891, 0.067, 0.966, 0.169],
});
