/**
 * The progress panel: the one moving "here is what I'm doing" card, from the moment a
 * photo lands until every check is read. It answers "is it stuck?" three ways at once:
 * - a small animated scan glyph (something is visibly alive),
 * - a bar: determinate when the work is countable (checks straightened and read),
 *   a gentle sliding segment when it is not (finding the checks),
 * - plain stage copy (progress_copy.js), plus a "still working" line once one stage has
 *   run longer than SLOW_STAGE_MS, so a big photo never looks frozen.
 *
 * The panel is sticky at the top of the viewport while visible (CSS), so after Continue
 * scrolls the page down to the grid it is still where she is looking. It can also end
 * in an error with one action button (Start over, Retry). Owns only its own DOM.
 * Motion is CSS-only and switched off under prefers-reduced-motion (styles/progress.css).
 */

const SLOW_STAGE_MS = 4000;
const SLOW_CHECK_INTERVAL_MS = 500;
const DONE_VISIBLE_MS = 3000; // long enough to read "All 6 checks read. 3 fields need a look (amber)."

export class ProgressPanel {
  /** `elements`: `{ panelElement, headlineElement, detailElement, barElement, trackElement, actionButton }`. */
  constructor(elements) {
    Object.assign(this, elements);
    this.stageStartedAt = 0;
    this.stageHeadline = "";
    this.slowHint = "";
    this.baseDetail = "";
    this.slowTimer = null;
    // Where the panel lives in index.html (under the drop zone); Continue moves it to the top
    // of the review step so it never covers the review heading.
    this.homeParent = this.panelElement.parentElement;
    this.homeNextSibling = this.panelElement.nextSibling;
    this.actionButton.addEventListener("click", () => this.onAction?.());
  }

  /** Moves the panel to the start of `containerElement` (sticky inside it from there). */
  placeAtStartOf(containerElement) {
    if (this.panelElement.parentElement !== containerElement || containerElement.firstElementChild !== this.panelElement) {
      containerElement.prepend(this.panelElement);
    }
  }

  /** Back under the drop zone, where every new photo's progress starts. */
  returnHome() {
    if (this.panelElement.parentElement !== this.homeParent) this.homeParent.insertBefore(this.panelElement, this.homeNextSibling);
  }

  isVisible() {
    return !this.panelElement.hidden;
  }

  /**
   * Shows a stage. `copy`: `{ headline, detail, slowHint }`. `fraction`: 0..1 for a
   * determinate bar, or null for the sliding "working" bar. The slow-stage clock restarts
   * only when the headline changes, so per-check detail updates do not reset it.
   */
  showStage(copy, fraction = null) {
    clearTimeout(this.doneTimer);
    this.clearAction();
    this.panelElement.hidden = false;
    this.panelElement.dataset.tone = "working";
    if (copy.headline !== this.stageHeadline) {
      this.stageHeadline = copy.headline;
      this.stageStartedAt = performance.now();
      this.headlineElement.textContent = copy.headline;
    }
    this.baseDetail = copy.detail || "";
    this.slowHint = copy.slowHint || "";
    this.renderDetail();
    this.setFraction(fraction);
    this.startSlowTimer();
  }

  setFraction(fraction) {
    const isDeterminate = typeof fraction === "number";
    this.trackElement.classList.toggle("progress-track--indeterminate", !isDeterminate);
    if (isDeterminate) {
      const percent = Math.round(Math.max(0.04, Math.min(1, fraction)) * 100);
      this.barElement.style.setProperty("width", `${percent}%`);
      this.trackElement.setAttribute("aria-valuenow", String(percent));
    } else {
      this.barElement.style.removeProperty("width");
      this.trackElement.removeAttribute("aria-valuenow");
    }
  }

  renderDetail() {
    const isSlow = this.slowHint && performance.now() - this.stageStartedAt > SLOW_STAGE_MS;
    const parts = [this.baseDetail, isSlow ? this.slowHint : ""].filter(Boolean);
    this.detailElement.textContent = parts.join(" · ");
    this.detailElement.hidden = parts.length === 0;
  }

  startSlowTimer() {
    if (this.slowTimer) return;
    this.slowTimer = setInterval(() => this.renderDetail(), SLOW_CHECK_INTERVAL_MS);
  }

  stopSlowTimer() {
    clearInterval(this.slowTimer);
    this.slowTimer = null;
  }

  /** The work finished: a full bar and `message` for a moment, then the panel folds away. */
  showDone(message, onFolded = null, visibleMs = DONE_VISIBLE_MS) {
    this.stopSlowTimer();
    this.clearAction();
    this.panelElement.hidden = false;
    this.panelElement.dataset.tone = "done";
    this.stageHeadline = "";
    this.headlineElement.textContent = message;
    this.detailElement.hidden = true;
    this.setFraction(1);
    clearTimeout(this.doneTimer);
    this.doneTimer = setTimeout(() => {
      if (this.panelElement.dataset.tone !== "done") return;
      this.hide();
      onFolded?.();
    }, visibleMs);
  }


  /** A failure she can act on: `message` plus one button (`actionLabel`, `onAction`). */
  showError(message, actionLabel = null, onAction = null) {
    this.stopSlowTimer();
    this.panelElement.hidden = false;
    this.panelElement.dataset.tone = "error";
    this.stageHeadline = "";
    this.headlineElement.textContent = message;
    this.detailElement.hidden = true;
    this.setFraction(null);
    this.clearAction();
    if (actionLabel) {
      this.actionButton.textContent = actionLabel;
      this.actionButton.hidden = false;
      this.onAction = onAction;
    }
  }

  clearAction() {
    this.actionButton.hidden = true;
    this.onAction = null;
  }

  hide() {
    clearTimeout(this.doneTimer);
    this.stopSlowTimer();
    this.clearAction();
    this.panelElement.hidden = true;
    this.stageHeadline = "";
  }
}
