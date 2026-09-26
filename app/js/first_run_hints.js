/**
 * Gentle first-run hints: never modal, never blocking, always dismissible.
 *
 * Each hint is a static `<details class="hint" data-hint-key="...">` in index.html with a
 * "Got it" button inside. For the operator's first few sessions at that moment the hint
 * opens by itself; once she presses "Got it" (or after MAX_AUTO_OPENS sessions) it stays
 * closed, leaving only its one-line summary as a quiet link she can reopen any time
 * (recognition over recall: the Gmail recipe is never lost, just folded away).
 *
 * What is remembered (text only, through storage/local_store.js, so "Clear everything
 * this page remembers" resets it): `{ [hintKey]: { opens, dismissed } }` under one key.
 * Nothing is written unless a hint actually auto-opens or is dismissed.
 */

const STORAGE_KEY = "firstRunHints";
const MAX_AUTO_OPENS = 3;

export const HINT_KEYS = Object.freeze({
  PASTE_FROM_GMAIL: "pasteFromGmail",
  FIX_OUTLINES: "fixOutlines",
  REVIEW_GUIDE: "reviewGuide",
});

export class FirstRunHints {
  /** `localStore`: storage/local_store.js LocalStore. `rootElement`: where the hint elements live. */
  constructor(localStore, rootElement = document) {
    this.localStore = localStore;
    this.hintElements = new Map(
      [...rootElement.querySelectorAll("details.hint[data-hint-key]")].map((element) => [element.dataset.hintKey, element]),
    );
    for (const [key, element] of this.hintElements) {
      const dismissButton = element.querySelector(".hint-dismiss");
      if (dismissButton) dismissButton.addEventListener("click", () => this.dismiss(key));
    }
  }

  readState() {
    const state = this.localStore.readJson(STORAGE_KEY, {});
    return state && typeof state === "object" ? state : {};
  }

  hintState(key) {
    const entry = this.readState()[key];
    return { opens: Number(entry?.opens) || 0, dismissed: Boolean(entry?.dismissed) };
  }

  writeHintState(key, entry) {
    this.localStore.writeJson(STORAGE_KEY, { ...this.readState(), [key]: entry });
  }

  /**
   * The moment for hint `key` has come (the empty page, a count step, a review grid):
   * show its summary line, and open it if she has not dismissed it or seen it enough.
   */
  present(key) {
    const element = this.hintElements.get(key);
    if (!element) return;
    element.hidden = false;
    const state = this.hintState(key);
    const shouldOpen = !state.dismissed && state.opens < MAX_AUTO_OPENS;
    element.open = shouldOpen;
    if (shouldOpen) this.writeHintState(key, { ...state, opens: state.opens + 1 });
  }

  /** Hides the hint entirely (its moment is over, e.g. the count step closed). */
  withdraw(key) {
    const element = this.hintElements.get(key);
    if (element) element.hidden = true;
  }

  /** "Got it": fold it away for good; the summary stays as a link. */
  dismiss(key) {
    const element = this.hintElements.get(key);
    if (element) element.open = false;
    this.writeHintState(key, { ...this.hintState(key), dismissed: true });
  }
}
