/**
 * The JS field-gate parsers and WRatio port vs the Python originals, on real eval strings.
 *
 *     node app/tests/parity/run_gating_parity.mjs   (after dump_gating_reference.py)
 *
 * Exits non-zero on any disagreement; prints the first few.
 */

import fs from "node:fs";
import { parseAmountNumericToCents, parseAmountWordsToCents } from "../../js/fields/money_parsing.js";
import { parseDateToIso } from "../../js/fields/date_parsing.js";
import { normalizeFreeText, weightedRatio } from "../../js/fields/fuzzy_matching.js";

const REFERENCE_PATH = "/tmp/check-transcriber-m4/gating_reference.json";
const SCORE_TOLERANCE = 1e-9;
const reference = JSON.parse(fs.readFileSync(REFERENCE_PATH, "utf8"));

let failed = false;
function compare(name, cases, compute, equal = (a, b) => a === b) {
  const mismatches = cases.filter(([input, expected, extra]) => !equal(compute(input, expected, extra), extra === undefined ? expected : extra));
  console.log(`[${mismatches.length ? "FAIL" : "PASS"}] ${name}: ${cases.length - mismatches.length}/${cases.length} identical`);
  mismatches.slice(0, 5).forEach((mismatch) => console.log("   ", JSON.stringify(mismatch)));
  failed ||= mismatches.length > 0;
}

compare("courtesy amount parsing", reference.courtesy, (text) => parseAmountNumericToCents(text));
compare("legal line parsing", reference.legal, (text) => parseAmountWordsToCents(text));
compare("date parsing", reference.date, (text) => parseDateToIso(text));
compare("free-text normalization", reference.free_text, (text) => normalizeFreeText(text));
compare("WRatio (payee snap scorer)", reference.wratio, (read, name) => weightedRatio(read, name), (a, b) => Math.abs(a - b) < SCORE_TOLERANCE);
process.exit(failed ? 1 : 0);
