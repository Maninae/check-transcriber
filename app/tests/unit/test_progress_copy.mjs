/**
 * Node unit tests for js/progress_copy.js: the worker's stage text mapped to operator copy,
 * the progress fraction, and the review summary line.
 *
 *     node app/tests/unit/test_progress_copy.mjs
 *
 * Also guards the product rule that the UI never says "AI", "model" or "OCR".
 * Prints one PASS/FAIL line per check and exits non-zero on any failure.
 */

import { STAGE_COPY, describeReviewSummary, describeWorkerStage, parseCheckOfTotal, reviewProgressFraction } from "../../js/progress_copy.js";

const failures = [];
function check(description, condition) {
  console.log(`[${condition ? "PASS" : "FAIL"}] ${description}`);
  if (!condition) failures.push(description);
}

const WORKER_STAGE_TEXTS = [
  "Finding checks",
  "Tightening the outlines",
  "Straightening check 3 of 6",
  "Reading check 3 of 6",
  "Reading handwriting on check 2 of 6",
  "Something new the worker says",
];
const FORBIDDEN_WORDS = /\b(ai|model|models|ocr)\b/i;

check("finding checks reads as 'Looking for checks…'", describeWorkerStage("Finding checks").headline === "Looking for checks…");
check("straightening keeps the check counter", describeWorkerStage("Straightening check 3 of 6").headline === "Straightening check 3 of 6…");
check("printed reads say what is read and which check",
  describeWorkerStage("Reading check 3 of 6").headline === "Reading the printed fields…" && describeWorkerStage("Reading check 3 of 6").detail === "Check 3 of 6");
check("handwriting warns it takes a moment", describeWorkerStage("Reading handwriting on check 2 of 6").headline === "Reading handwriting… (this takes a moment)");
check("unknown worker text passes through with an ellipsis", describeWorkerStage("Something new").headline === "Something new…");
check("every stage has a slow hint", WORKER_STAGE_TEXTS.every((text) => describeWorkerStage(text).slowHint));

const allCopy = [...WORKER_STAGE_TEXTS.map(describeWorkerStage), ...Object.values(STAGE_COPY)]
  .flatMap(({ headline, detail, slowHint }) => [headline, detail, slowHint]);
check("no stage copy says AI, model or OCR", allCopy.every((text) => !FORBIDDEN_WORDS.test(text)));

check("parses 'check N of M'", JSON.stringify(parseCheckOfTotal("Reading check 4 of 7")) === JSON.stringify({ current: 4, total: 7 }));
check("no counter -> null", parseCheckOfTotal("Finding checks") === null);

check("fraction is 0 with nothing done", reviewProgressFraction({ cropsShown: 0, printedReadsShown: 0, checkCount: 6 }) === 0);
check("fraction is half once every check is straightened", reviewProgressFraction({ cropsShown: 6, printedReadsShown: 0, checkCount: 6 }) === 0.5);
check("fraction is 1 once every check is read", reviewProgressFraction({ cropsShown: 6, printedReadsShown: 6, checkCount: 6 }) === 1);
check("no checks -> 0, not NaN", reviewProgressFraction({ cropsShown: 0, printedReadsShown: 0, checkCount: 0 }) === 0);

const summary = (overrides) => describeReviewSummary({ checkCount: 6, unreadFieldCount: 0, flaggedFieldCount: 0, handwritingPending: false, ...overrides });
check("unread fields -> Reading…", summary({ unreadFieldCount: 4 }).text === "Reading…" && summary({ unreadFieldCount: 4 }).tone === "working");
check("amber fields are counted", summary({ flaggedFieldCount: 3 }).text === "All 6 checks read. 3 fields need a look (amber).");
check("one amber field is singular", summary({ flaggedFieldCount: 1 }).text === "All 6 checks read. 1 field needs a look (amber).");
check("nothing flagged says so", summary({}).tone === "done");
check("handwriting still running is 'working'", summary({ handwritingPending: true, flaggedFieldCount: 2 }).tone === "working");
check("one check is singular", describeReviewSummary({ checkCount: 1, unreadFieldCount: 0, flaggedFieldCount: 0 }).text.startsWith("All 1 check read."));

if (failures.length) {
  console.log(`\n${failures.length} check(s) failed`);
  process.exit(1);
}
console.log("\nAll checks passed.");
