/**
 * Check amounts to integer cents: the courtesy box ("$***1,245.76", "694 xx/100") and the
 * written legal line ("FOUR HUNDRED FIFTY-THREE AND 00/100"). A line-for-line port of
 * experiments/field_reading/metrics/money_amount_parsing.py, which defines what the field
 * reader was scored against; change the Python first, then mirror it here.
 *
 * Both parsers return null for anything they do not fully understand, so a half-read
 * amount never agrees with anything by accident.
 */

const CENTS_PER_DOLLAR = 100;

const DOT_CENTS_PATTERN = /^(\d+)\.(\d{2})$/;
const DOT_ZERO_CENTS_PATTERN = /^(\d+)\.(?:xx|-+)$/;
const FRACTION_CENTS_PATTERN = /^(\d+)\s*(\d{2}|xx|no)\s*\/\s*100$/;
const WHOLE_DOLLARS_PATTERN = /^(\d+)\s*=*$/;
const COURTESY_EDGE_NOISE_CHARACTERS = "$* \t";

const WORDS_FRACTION_PATTERN = /(\d{1,2}|xx|no)\s*\/\s*100/g;
const WORDS_CENTS_PATTERN = /(\d{1,2})\s*cents?\b/g;
const WORDS_FILLER_TOKENS = new Set(["and", "dollars", "dollar", "only", "exactly"]);
const WORDS_PUNCTUATION_TOKENS = new Set(["*", ".", ",", "/", ":", ";"]);
const UNIT_WORD_VALUES = {
  zero: 0,
  one: 1,
  two: 2,
  three: 3,
  four: 4,
  five: 5,
  six: 6,
  seven: 7,
  eight: 8,
  nine: 9,
};
const TEEN_WORD_VALUES = {
  ten: 10,
  eleven: 11,
  twelve: 12,
  thirteen: 13,
  fourteen: 14,
  fifteen: 15,
  sixteen: 16,
  seventeen: 17,
  eighteen: 18,
  nineteen: 19,
};
const TENS_WORD_VALUES = {
  twenty: 20,
  thirty: 30,
  forty: 40,
  fifty: 50,
  sixty: 60,
  seventy: 70,
  eighty: 80,
  ninety: 90,
};
const SCALE_WORD_VALUES = {
  thousand: 1000,
  million: 1000000,
};

const has = (table, key) => Object.prototype.hasOwnProperty.call(table, key);

/** Python's str.strip(characters). */
function stripCharacters(text, characters) {
  let start = 0;
  let end = text.length;
  while (start < end && characters.includes(text[start])) start += 1;
  while (end > start && characters.includes(text[end - 1])) end -= 1;
  return text.slice(start, end);
}

/** `07` -> 7; `xx`, `no` and dash runs mean zero cents. */
function fractionTokenToCents(fractionToken) {
  return /^\d+$/.test(fractionToken) ? parseInt(fractionToken, 10) : 0;
}

/** Courtesy-box text -> integer cents, or null. */
export function parseAmountNumericToCents(courtesyText) {
  let compactText = stripCharacters(courtesyText, COURTESY_EDGE_NOISE_CHARACTERS).replaceAll(",", "").toLowerCase();
  compactText = compactText.replaceAll("*", "").replaceAll("$", "").trim();
  let match = DOT_CENTS_PATTERN.exec(compactText);
  if (match) return parseInt(match[1], 10) * CENTS_PER_DOLLAR + parseInt(match[2], 10);
  match = DOT_ZERO_CENTS_PATTERN.exec(compactText);
  if (match) return parseInt(match[1], 10) * CENTS_PER_DOLLAR;
  match = FRACTION_CENTS_PATTERN.exec(compactText);
  if (match) return parseInt(match[1], 10) * CENTS_PER_DOLLAR + fractionTokenToCents(match[2]);
  match = WHOLE_DOLLARS_PATTERN.exec(compactText);
  if (match) return parseInt(match[1], 10) * CENTS_PER_DOLLAR;
  return null;
}

/**
 * ["two", "thousand", "forty", "one"] -> 2041, or null for a malformed sequence.
 * The previous word's kind is tracked so "five twenty" or "twenty twenty" are rejected, not summed.
 */
export function numberWordsToInteger(numberWords) {
  if (numberWords.length === 0) return null;
  let totalValue = 0;
  let groupValue = 0;
  let previousKind = null;
  for (const word of numberWords) {
    if (has(UNIT_WORD_VALUES, word) && [null, "tens", "hundred"].includes(previousKind)) {
      groupValue += UNIT_WORD_VALUES[word];
      previousKind = "unit";
    } else if (has(TEEN_WORD_VALUES, word) && [null, "hundred"].includes(previousKind)) {
      groupValue += TEEN_WORD_VALUES[word];
      previousKind = "teen";
    } else if (has(TENS_WORD_VALUES, word) && [null, "hundred"].includes(previousKind)) {
      groupValue += TENS_WORD_VALUES[word];
      previousKind = "tens";
    } else if (word === "hundred" && ["unit", "teen"].includes(previousKind) && groupValue < 100) {
      groupValue *= 100;
      previousKind = "hundred";
    } else if (has(SCALE_WORD_VALUES, word) && previousKind !== null) {
      totalValue += groupValue * SCALE_WORD_VALUES[word];
      groupValue = 0;
      previousKind = null;
    } else {
      return null;
    }
  }
  return totalValue + groupValue;
}

/** Legal-line text -> integer cents, or null. */
export function parseAmountWordsToCents(legalLineText) {
  let loweredText = legalLineText.toLowerCase().replaceAll("&", " and ").replaceAll("-", " ");
  let centsValue = 0;
  const fractionMatches = [...loweredText.matchAll(WORDS_FRACTION_PATTERN)].map((match) => match[1]);
  const centsMatches = [...loweredText.matchAll(WORDS_CENTS_PATTERN)].map((match) => match[1]);
  const matches = fractionMatches.length ? fractionMatches : centsMatches;
  if (matches.length > 1) return null;
  if (matches.length === 1) {
    centsValue = fractionTokenToCents(matches[0]);
    loweredText = loweredText.replace(WORDS_FRACTION_PATTERN, " ").replace(WORDS_CENTS_PATTERN, " ");
  }
  const wordTokens = loweredText.match(/[a-z]+|\S/g) || [];
  const numberWords = wordTokens.filter((token) => !WORDS_FILLER_TOKENS.has(token) && !WORDS_PUNCTUATION_TOKENS.has(token));
  const dollarValue = numberWordsToInteger(numberWords);
  if (dollarValue === null) return null;
  return dollarValue * CENTS_PER_DOLLAR + centsValue;
}

/** Two decimals, no symbol, no thousands separator: what pastes into a numeric column. */
export function formatCentsAsAmount(cents) {
  const dollars = Math.floor(cents / CENTS_PER_DOLLAR);
  const remainder = cents % CENTS_PER_DOLLAR;
  return `${dollars}.${String(remainder).padStart(2, "0")}`;
}

/** An operator-typed amount normalized for copying ("$1,250" -> "1250.00"); the text itself if it does not parse. */
export function normalizeTypedAmount(typedText) {
  const cents = parseAmountNumericToCents(typedText);
  return cents === null ? typedText.trim() : formatCentsAsAmount(cents);
}
