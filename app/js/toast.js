/**
 * The one transient confirmation line at the bottom of the screen ("6 rows copied").
 * One toast at a time; a new message replaces the old one and restarts the timer.
 */

const TOAST_VISIBLE_MS = 1800;

export class Toast {
  constructor(toastElement) {
    this.toastElement = toastElement;
    this.hideTimer = null;
  }

  show(message) {
    this.toastElement.textContent = message;
    this.toastElement.hidden = false;
    clearTimeout(this.hideTimer);
    this.hideTimer = setTimeout(() => { this.toastElement.hidden = true; }, TOAST_VISIBLE_MS);
  }
}
