/**
 * The one transient confirmation at the bottom of the screen ("6 rows copied", "Check 2
 * copied. Paste it into your tracker with Ctrl+V."). One toast at a time: a new message
 * replaces the old one and restarts the timer. The icon is CSS (styles/base.css), so the
 * element's text is exactly the message.
 *
 * `tone`: "success" (a tick, the default) or "info" (an "i", for something that did not work
 * but needs no alarm). Longer messages stay up longer, so a sentence can be read in full.
 */

const TOAST_MINIMUM_VISIBLE_MS = 1800;
const TOAST_MS_PER_CHARACTER = 45;
const TOAST_MAXIMUM_VISIBLE_MS = 5000;

export class Toast {
  constructor(toastElement) {
    this.toastElement = toastElement;
    this.hideTimer = null;
  }

  show(message, { tone = "success" } = {}) {
    this.toastElement.textContent = message;
    this.toastElement.dataset.tone = tone;
    // Re-show from hidden so the entrance animation plays again for a replacing message.
    this.toastElement.hidden = true;
    void this.toastElement.offsetWidth;
    this.toastElement.hidden = false;
    clearTimeout(this.hideTimer);
    const visibleMs = Math.min(TOAST_MAXIMUM_VISIBLE_MS, Math.max(TOAST_MINIMUM_VISIBLE_MS, message.length * TOAST_MS_PER_CHARACTER));
    this.hideTimer = setTimeout(() => { this.toastElement.hidden = true; }, visibleMs);
  }
}
