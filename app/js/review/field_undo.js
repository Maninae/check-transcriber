/**
 * Ctrl+Z per field (spec 4.3): undoes the last edit in the focused field only.
 *
 * The browser's own input undo is dropped whenever a value is set from script (the
 * payer autocomplete snap in a later milestone does exactly that), so each field keeps
 * its own small history. Typing bursts coalesce: a pause longer than
 * `TYPING_BURST_GAP_MS`, or a switch between typing and deleting, starts a new step, so
 * one Ctrl+Z removes a word-sized chunk rather than one character.
 */

const TYPING_BURST_GAP_MS = 700;
const MAXIMUM_UNDO_STEPS = 50;

function isUndoShortcut(event) {
  return (event.ctrlKey || event.metaKey) && !event.shiftKey && !event.altKey && event.key.toLowerCase() === "z";
}

function editKind(inputType) {
  return inputType && inputType.startsWith("delete") ? "delete" : "insert";
}

/**
 * Attaches the history to `inputElement`. `onValueRestored(value)` fires after an undo
 * so the caller can update its own state. Returns `{ setValue(value), clearHistory() }`;
 * `setValue` is for script edits that should themselves be undoable.
 */
export function attachFieldUndoHistory(inputElement, onValueRestored) {
  const previousValues = [];
  let lastEditTime = 0;
  let lastEditKind = null;

  function pushSnapshot(value) {
    previousValues.push(value);
    if (previousValues.length > MAXIMUM_UNDO_STEPS) previousValues.shift();
  }

  inputElement.addEventListener("beforeinput", (event) => {
    const now = performance.now();
    const kind = editKind(event.inputType);
    if (now - lastEditTime > TYPING_BURST_GAP_MS || kind !== lastEditKind) {
      pushSnapshot(inputElement.value);
    }
    lastEditTime = now;
    lastEditKind = kind;
  });

  inputElement.addEventListener("keydown", (event) => {
    if (!isUndoShortcut(event)) return;
    event.preventDefault();
    if (previousValues.length === 0) return;
    inputElement.value = previousValues.pop();
    lastEditKind = null;
    onValueRestored(inputElement.value);
  });

  return {
    /** Forgets every step (Rotate re-reads the field; undoing into the old orientation's read would mislead). */
    clearHistory() {
      previousValues.length = 0;
      lastEditKind = null;
    },
    setValue(value) {
      pushSnapshot(inputElement.value);
      lastEditKind = null;
      inputElement.value = value;
    },
  };
}
