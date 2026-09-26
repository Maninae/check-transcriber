/**
 * The count step's one-line header and the note under it (spec 4.2 and the zero-found
 * and too-small cases in spec sections 5 and 6). Pure text from the current quads.
 */

import { measureQuadSideLengths } from "../pipeline/quadrilateral_math.js";

// Spec 6 says "about 400 px". Checks ~370 px wide in the synthetic eval photos still
// rectify to legible crops (payer, date, amount all readable), so the warning fires only
// clearly below that, where crops stop being readable.
const MINIMUM_READABLE_CHECK_WIDTH_PX = 300;

function pluralizeChecks(count) {
  return `${count} ${count === 1 ? "check" : "checks"}`;
}

/** Returns `{ heading, note }` for `quads` (`[{ corners, confident }]`). */
export function describeCountHeader(quads, wasEdited) {
  if (quads.length === 0) {
    return {
      heading: wasEdited ? "No checks outlined yet" : "I couldn't find any checks",
      note: wasEdited
        ? "Click Add a check, then drag a box around each check on the photo."
        : "Try a photo on a plainer background with all four corners of each check showing, "
          + "or add them by hand: drag a box around each one on the photo below.",
    };
  }
  const heading = wasEdited ? pluralizeChecks(quads.length) : `Found ${pluralizeChecks(quads.length)}`;
  const longSides = quads.map(({ corners }) => measureQuadSideLengths(corners).longSide).sort((a, b) => a - b);
  if (longSides[Math.floor(longSides.length / 2)] < MINIMUM_READABLE_CHECK_WIDTH_PX) {
    return { heading, note: "This photo is too small to read; ask for the original attachment." };
  }
  if (quads.every(({ confident }) => confident)) {
    return { heading, note: "Every outline looks solid. Count the checks on the photo; if they match, press Continue." };
  }
  return {
    heading,
    note: "Give the amber outlines a second look, then press Continue.",
  };
}
