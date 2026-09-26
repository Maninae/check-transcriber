/**
 * String similarity for the two snaps in the review grid:
 * - `weightedRatio`: a port of rapidfuzz 3.x `fuzz.WRatio` (no processor, case-sensitive),
 *   the scorer experiments/field_reading/field_gating/payee_snap.py measured the payee snap
 *   with (95% precision at score >= 60). Mirrors rapidfuzz's pure-Python fallback
 *   (`fuzz_py.py`) including its window-skipping in partial_ratio, so scores match exactly.
 * - `levenshteinDistance`: plain edit distance, for the payer autocomplete snap (spec 4.3,
 *   "within a small edit distance of a known name").
 */

const UNBASE_SCALE = 0.95;

/** Longest common subsequence length (Indel similarity is 2 x LCS). */
function longestCommonSubsequenceLength(first, second) {
  if (first.length === 0 || second.length === 0) return 0;
  let previousRow = new Array(second.length + 1).fill(0);
  let currentRow = new Array(second.length + 1).fill(0);
  for (let i = 1; i <= first.length; i += 1) {
    for (let j = 1; j <= second.length; j += 1) {
      currentRow[j] = first[i - 1] === second[j - 1] ? previousRow[j - 1] + 1 : Math.max(previousRow[j], currentRow[j - 1]);
    }
    [previousRow, currentRow] = [currentRow, previousRow];
  }
  return previousRow[second.length];
}

/** Indel normalized similarity in [0, 1] (rapidfuzz `Indel.normalized_similarity`). */
function indelNormalizedSimilarity(first, second) {
  const lengthSum = first.length + second.length;
  if (lengthSum === 0) return 1;
  const distance = lengthSum - 2 * longestCommonSubsequenceLength(first, second);
  return 1 - distance / lengthSum;
}

/** rapidfuzz `fuzz.ratio`, 0-100. Strings are compared as arrays of code points. */
export function simpleRatio(first, second) {
  return indelNormalizedSimilarity([...first], [...second]) * 100;
}

/** `_partial_ratio_impl`: best window of `longer` against all of `shorter` (both arrays), 0-100. */
function partialRatioImplementation(shorter, longer) {
  const shorterCharacters = new Set(shorter);
  const shorterLength = shorter.length;
  const longerLength = longer.length;
  let bestScore = 0;
  for (let i = 1; i < shorterLength; i += 1) {
    if (!shorterCharacters.has(longer[i - 1])) continue;
    const score = indelNormalizedSimilarity(shorter, longer.slice(0, i));
    if (score > bestScore) {
      bestScore = score;
      if (bestScore === 1) return 100;
    }
  }
  for (let i = 0; i < longerLength - shorterLength; i += 1) {
    if (!shorterCharacters.has(longer[i + shorterLength - 1])) continue;
    const score = indelNormalizedSimilarity(shorter, longer.slice(i, i + shorterLength));
    if (score > bestScore) {
      bestScore = score;
      if (bestScore === 1) return 100;
    }
  }
  for (let i = Math.max(0, longerLength - shorterLength); i < longerLength; i += 1) {
    if (!shorterCharacters.has(longer[i])) continue;
    const score = indelNormalizedSimilarity(shorter, longer.slice(i));
    if (score > bestScore) {
      bestScore = score;
      if (bestScore === 1) return 100;
    }
  }
  return bestScore * 100;
}

/** rapidfuzz `fuzz.partial_ratio`, 0-100. */
export function partialRatio(first, second) {
  const firstCharacters = [...first];
  const secondCharacters = [...second];
  if (firstCharacters.length === 0 && secondCharacters.length === 0) return 100;
  const firstIsShorter = firstCharacters.length <= secondCharacters.length;
  const shorter = firstIsShorter ? firstCharacters : secondCharacters;
  const longer = firstIsShorter ? secondCharacters : firstCharacters;
  let score = partialRatioImplementation(shorter, longer);
  if (score !== 100 && firstCharacters.length === secondCharacters.length) {
    score = Math.max(score, partialRatioImplementation(longer, shorter));
  }
  return score;
}

/** Python's str.split(): whitespace-separated tokens. */
function splitTokens(text) {
  return text.split(/\s+/).filter(Boolean);
}

function sortedJoined(tokens) {
  return [...tokens].sort((a, b) => (a < b ? -1 : a > b ? 1 : 0)).join(" ");
}

function normalizedDistanceScore(distance, lengthSum) {
  return lengthSum ? 100 - (100 * distance) / lengthSum : 100;
}

function tokenSortRatio(first, second) {
  return simpleRatio(sortedJoined(splitTokens(first)), sortedJoined(splitTokens(second)));
}

function tokenSetRatio(first, second) {
  const tokensA = new Set(splitTokens(first));
  const tokensB = new Set(splitTokens(second));
  if (tokensA.size === 0 || tokensB.size === 0) return 0;
  const intersection = [...tokensA].filter((token) => tokensB.has(token));
  const differenceAB = [...tokensA].filter((token) => !tokensB.has(token));
  const differenceBA = [...tokensB].filter((token) => !tokensA.has(token));
  if (intersection.length && (!differenceAB.length || !differenceBA.length)) return 100;
  const differenceABJoined = sortedJoined(differenceAB);
  const differenceBAJoined = sortedJoined(differenceBA);
  const abLength = [...differenceABJoined].length;
  const baLength = [...differenceBAJoined].length;
  const sectionLength = [...intersection.join(" ")].length;
  const hasSection = sectionLength !== 0 ? 1 : 0;
  const sectionABLength = sectionLength + hasSection + abLength;
  const sectionBALength = sectionLength + hasSection + baLength;
  const indelDistance = abLength + baLength - 2 * longestCommonSubsequenceLength([...differenceABJoined], [...differenceBAJoined]);
  const result = normalizedDistanceScore(indelDistance, sectionABLength + sectionBALength);
  if (!sectionLength) return result;
  const sectionABRatio = normalizedDistanceScore(hasSection + abLength, sectionLength + sectionABLength);
  const sectionBARatio = normalizedDistanceScore(hasSection + baLength, sectionLength + sectionBALength);
  return Math.max(result, sectionABRatio, sectionBARatio);
}

function tokenRatio(first, second) {
  return Math.max(tokenSetRatio(first, second), tokenSortRatio(first, second));
}

function partialTokenRatio(first, second) {
  const tokenListA = splitTokens(first);
  const tokenListB = splitTokens(second);
  const tokensA = new Set(tokenListA);
  const tokensB = new Set(tokenListB);
  if ([...tokensA].some((token) => tokensB.has(token))) return 100;
  const differenceAB = [...tokensA].filter((token) => !tokensB.has(token));
  const differenceBA = [...tokensB].filter((token) => !tokensA.has(token));
  const result = partialRatio(sortedJoined(tokenListA), sortedJoined(tokenListB));
  if (tokenListA.length === differenceAB.length && tokenListB.length === differenceBA.length) return result;
  return Math.max(result, partialRatio(sortedJoined(differenceAB), sortedJoined(differenceBA)));
}

/** rapidfuzz `fuzz.WRatio(first, second)`, 0-100 (see module docstring). */
export function weightedRatio(first, second) {
  if (!first || !second) return 0;
  const firstLength = [...first].length;
  const secondLength = [...second].length;
  const lengthRatio = firstLength > secondLength ? firstLength / secondLength : secondLength / firstLength;
  let endRatio = simpleRatio(first, second);
  if (lengthRatio < 1.5) return Math.max(endRatio, tokenRatio(first, second) * UNBASE_SCALE);
  const partialScale = lengthRatio <= 8.0 ? 0.9 : 0.6;
  endRatio = Math.max(endRatio, partialRatio(first, second) * partialScale);
  return Math.max(endRatio, partialTokenRatio(first, second) * UNBASE_SCALE * partialScale);
}

/**
 * rapidfuzz `process.extractOne(query, choices, scorer=WRatio)`: the best-scoring choice
 * (first one wins ties), or null for an empty query or list. Returns `{ choice, score }`.
 */
export function bestWeightedRatioMatch(query, choices) {
  let best = null;
  for (const choice of choices) {
    const score = weightedRatio(query, choice);
    if (best === null || score > best.score) best = { choice, score };
  }
  return best;
}

/** Levenshtein edit distance (unit costs). */
export function levenshteinDistance(first, second) {
  const firstCharacters = [...first];
  const secondCharacters = [...second];
  let previousRow = Array.from({ length: secondCharacters.length + 1 }, (unused, index) => index);
  for (let i = 1; i <= firstCharacters.length; i += 1) {
    const currentRow = [i];
    for (let j = 1; j <= secondCharacters.length; j += 1) {
      const substitutionCost = firstCharacters[i - 1] === secondCharacters[j - 1] ? 0 : 1;
      currentRow.push(Math.min(previousRow[j] + 1, currentRow[j - 1] + 1, previousRow[j - 1] + substitutionCost));
    }
    previousRow = currentRow;
  }
  return previousRow[secondCharacters.length];
}

const EDGE_PUNCTUATION_OR_SPACE = /^[\s!-/:-@[-`{-~]+|[\s!-/:-@[-`{-~]+$/g;

/**
 * experiments' `normalize_free_text`: casefold, collapse whitespace, strip ASCII punctuation
 * and spaces at both ends. Used to compare names (duplicates, autocomplete).
 */
export function normalizeFreeText(text) {
  return text.toLowerCase().split(/\s+/).filter(Boolean).join(" ").replace(EDGE_PUNCTUATION_OR_SPACE, "");
}
