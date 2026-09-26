/**
 * The hand-typed fields beside each crop, and the column order used when copying rows.
 *
 * Display order follows the check's layout (spec 4.3). Copy order is the spec 4.5
 * default (Date, Payer, Amount, Check number, Memo); the settings panel that lets the
 * operator reorder or drop columns is a later milestone, so this constant is the stub.
 */

export const REVIEW_FIELDS = Object.freeze([
  { key: "payer", label: "Payer" },
  { key: "amount", label: "Amount" },
  { key: "date", label: "Date" },
  { key: "checkNumber", label: "Check number" },
  { key: "memo", label: "Memo" },
]);

export const DEFAULT_COPY_COLUMN_ORDER = Object.freeze(["date", "payer", "amount", "checkNumber", "memo"]);

/** An empty `{ fieldKey: "" }` record for a new row. */
export function createEmptyFieldValues() {
  return Object.fromEntries(REVIEW_FIELDS.map(({ key }) => [key, ""]));
}
