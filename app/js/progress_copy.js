/**
 * Plain-words copy for what the page is doing, shown in the progress panel
 * (progress_panel.js). Pure text in, text out, so it is Node-testable.
 *
 * The pipeline worker reports its stage in its own words ("Finding checks",
 * "Straightening check 3 of 6", "Reading check 3 of 6", "Reading handwriting on check
 * 2 of 6"). This module turns each into what the operator sees:
 * - `headline`: the stage in plain words ("Straightening check 3 of 6…").
 * - `detail`: one quieter line of context, or "".
 * - `slowHint`: what to say once the stage has taken a while ("Still working…").
 * Unknown worker text passes through unchanged, so a new worker stage never shows blank.
 * The UI never says "AI", "model" or "OCR" (app/CLAUDE.md product invariants).
 */

const CHECK_OF_TOTAL_PATTERN = /check (\d+) of (\d+)/i;

export const STAGE_COPY = Object.freeze({
  opening: { headline: "Opening the photo…", detail: "", slowHint: "Still opening. Large photos take a few seconds." },
  waitingForTools: {
    headline: "Getting ready…",
    detail: "Your photo is safe here and starts the moment the check-reading tools are loaded.",
    slowHint: "Still loading the check-reading tools. This only takes long the first time.",
  },
  findingChecks: { headline: "Looking for checks…", detail: "", slowHint: "Still looking. Large photos take a little longer." },
  outliningChecks: { headline: "Outlining each check…", detail: "", slowHint: "Still working. Large photos take a little longer." },
});

/** Operator copy for one worker progress message. */
export function describeWorkerStage(workerText) {
  const text = String(workerText || "").trim();
  if (/^finding checks/i.test(text)) return STAGE_COPY.findingChecks;
  if (/^tightening the outlines/i.test(text)) return STAGE_COPY.outliningChecks;
  const match = text.match(CHECK_OF_TOTAL_PATTERN);
  if (match && /^straightening/i.test(text)) {
    return { headline: `Straightening check ${match[1]} of ${match[2]}…`, detail: "Turning each check upright and trimming the background.", slowHint: "Still working, large photo." };
  }
  if (match && /^reading handwriting/i.test(text)) {
    return {
      headline: "Reading handwriting… (this takes a moment)",
      detail: `Check ${match[1]} of ${match[2]}. You can start reviewing the checks above while this runs.`,
      slowHint: "Still reading handwriting. This is the slowest part.",
    };
  }
  if (match && /^reading/i.test(text)) {
    return { headline: "Reading the printed fields…", detail: `Check ${match[1]} of ${match[2]}`, slowHint: "Still reading. Almost there." };
  }
  return { headline: text.endsWith("…") ? text : `${text}…`, detail: "", slowHint: "Still working…" };
}

/** `{ current, total }` from worker text like "Reading check 3 of 6", or null. */
export function parseCheckOfTotal(workerText) {
  const match = String(workerText || "").match(CHECK_OF_TOTAL_PATTERN);
  return match ? { current: Number(match[1]), total: Number(match[2]) } : null;
}

/**
 * Fraction done (0..1) of Continue's work: every check straightened, then every check
 * read. The handwriting pass, when on, is reported by its own check counter instead.
 */
export function reviewProgressFraction({ cropsShown, printedReadsShown, checkCount }) {
  if (!checkCount) return 0;
  return Math.min(1, (cropsShown + printedReadsShown) / (2 * checkCount));
}

/** "6 checks" / "1 check". */
export function pluralizeChecks(count) {
  return `${count} ${count === 1 ? "check" : "checks"}`;
}

/**
 * The line beside the review heading once the reads are in. `counts`:
 * `{ checkCount, unreadFieldCount, flaggedFieldCount, handwritingPending }`.
 */
export function describeReviewSummary({ checkCount, unreadFieldCount, flaggedFieldCount, handwritingPending }) {
  if (unreadFieldCount > 0) return { tone: "working", text: "Reading…" };
  const base = `All ${pluralizeChecks(checkCount)} read.`;
  if (handwritingPending) return { tone: "working", text: `${base} Handwriting still coming in.` };
  if (flaggedFieldCount === 0) return { tone: "done", text: `${base} Everything looks confident. Copy the rows when ready.` };
  const fields = `${flaggedFieldCount} ${flaggedFieldCount === 1 ? "field needs" : "fields need"}`;
  return { tone: "attention", text: `${base} ${fields} a look (amber).` };
}
