/**
 * The one quiet line under the drop zone that reports whether the check-reading
 * tools are ready yet. Three states: loading (first visit only, in practice — later
 * visits are served from the service worker cache almost instantly), ready (the line
 * disappears; nothing left to say), and failed (a plain message plus a Retry button).
 *
 * This module only touches its own three DOM elements. It has no opinion about what
 * "ready" means upstream — `engine_loader.js` decides that and calls into here.
 */

export class EngineStatusLine {
  constructor({ containerElement, textElement, retryButtonElement }) {
    this.containerElement = containerElement;
    this.textElement = textElement;
    this.retryButtonElement = retryButtonElement;
    this.retryButtonElement.hidden = true;
  }

  showLoading(message) {
    this.containerElement.hidden = false;
    this.textElement.textContent = message;
    this.retryButtonElement.hidden = true;
  }

  /** Ready means nothing left to report — the line goes away so the drop zone stays the one thing on the page. */
  showReady() {
    this.containerElement.hidden = true;
  }

  showError(message, onRetry) {
    this.containerElement.hidden = false;
    this.textElement.textContent = message;
    this.retryButtonElement.hidden = false;
    this.retryButtonElement.onclick = onRetry;
  }
}
