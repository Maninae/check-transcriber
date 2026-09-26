/**
 * The lightbox (spec 4.4): one check crop, large, over a dimmed page. X or Escape closes
 * it and the page is exactly where it was (nothing scrolls, focus returns to where it
 * was); left/right arrows step through checks; wheel or pinch zooms around the pointer;
 * drag pans; Rotate turns the check (the same rotation as its row). It never holds an
 * input field, so nothing the operator is typing can hide behind it.
 *
 * The crop is drawn at its full 1600 px resolution and scaled with a CSS transform, so
 * zooming in stays sharp. `checkSource` is the review grid: `getCheckCount()`,
 * `getDisplayedCropCanvas(index)`, `rotateCheck(index)`.
 */

const MINIMUM_ZOOM = 1;
const MAXIMUM_ZOOM = 6;
const WHEEL_ZOOM_SENSITIVITY = 0.0015;

export class Lightbox {
  constructor({ lightboxElement, stageElement, canvasElement, captionElement, previousButton, nextButton, rotateButton, closeButton }, checkSource) {
    Object.assign(this, { lightboxElement, stageElement, canvasElement, captionElement });
    this.checkSource = checkSource;
    this.currentIndex = 0;
    this.focusBeforeOpening = null;
    this.activePointers = new Map(); // pointerId -> { x, y }
    this.resetView();
    closeButton.addEventListener("click", () => this.close());
    previousButton.addEventListener("click", () => this.step(-1));
    nextButton.addEventListener("click", () => this.step(1));
    rotateButton.addEventListener("click", () => this.rotateCurrent());
    lightboxElement.addEventListener("click", (event) => {
      if (event.target === lightboxElement) this.close(); // a click on the dimmed backdrop
    });
    document.addEventListener("keydown", (event) => this.handleKeydown(event));
    stageElement.addEventListener("wheel", (event) => this.handleWheel(event), { passive: false });
    stageElement.addEventListener("pointerdown", (event) => this.handlePointerDown(event));
    stageElement.addEventListener("pointermove", (event) => this.handlePointerMove(event));
    stageElement.addEventListener("pointerup", (event) => this.handlePointerUp(event));
    stageElement.addEventListener("pointercancel", (event) => this.handlePointerUp(event));
  }

  isOpen() {
    return !this.lightboxElement.hidden;
  }

  open(index) {
    this.focusBeforeOpening = document.activeElement;
    this.lightboxElement.hidden = false;
    this.show(index);
  }

  close() {
    if (!this.isOpen()) return;
    this.lightboxElement.hidden = true;
    this.canvasElement.width = 0; // drop the full-resolution copy
    if (this.focusBeforeOpening && this.focusBeforeOpening.focus) this.focusBeforeOpening.focus({ preventScroll: true });
  }

  show(index) {
    const checkCount = this.checkSource.getCheckCount();
    this.currentIndex = (index + checkCount) % checkCount;
    this.resetView();
    this.redraw();
    this.captionElement.textContent = `Check ${this.currentIndex + 1} of ${checkCount}`;
  }

  redraw() {
    const sourceCanvas = this.checkSource.getDisplayedCropCanvas(this.currentIndex);
    if (!sourceCanvas) return; // still being straightened
    this.canvasElement.width = sourceCanvas.width;
    this.canvasElement.height = sourceCanvas.height;
    this.canvasElement.getContext("2d").drawImage(sourceCanvas, 0, 0);
  }

  step(direction) {
    this.show(this.currentIndex + direction);
  }

  rotateCurrent() {
    this.checkSource.rotateCheck(this.currentIndex);
    this.redraw();
  }

  handleKeydown(event) {
    if (!this.isOpen()) return;
    const actions = { Escape: () => this.close(), ArrowLeft: () => this.step(-1), ArrowRight: () => this.step(1) };
    if (!actions[event.key]) return;
    event.preventDefault();
    actions[event.key]();
  }

  resetView() {
    this.zoom = MINIMUM_ZOOM;
    this.offsetX = 0;
    this.offsetY = 0;
    this.applyTransform();
  }

  applyTransform() {
    this.canvasElement.style.transform = `translate(${this.offsetX}px, ${this.offsetY}px) scale(${this.zoom})`;
  }

  /** Zooms to `newZoom` keeping the page point (clientX, clientY) fixed under the pointer. */
  zoomAround(newZoom, clientX, clientY) {
    const clampedZoom = Math.min(MAXIMUM_ZOOM, Math.max(MINIMUM_ZOOM, newZoom));
    const stageRect = this.stageElement.getBoundingClientRect();
    // The canvas scales about its centre, which sits at the stage centre plus the offset.
    const pointerX = clientX - (stageRect.left + stageRect.width / 2);
    const pointerY = clientY - (stageRect.top + stageRect.height / 2);
    const ratio = clampedZoom / this.zoom;
    this.offsetX = pointerX - (pointerX - this.offsetX) * ratio;
    this.offsetY = pointerY - (pointerY - this.offsetY) * ratio;
    this.zoom = clampedZoom;
    if (this.zoom === MINIMUM_ZOOM) {
      this.offsetX = 0;
      this.offsetY = 0;
    }
    this.applyTransform();
  }

  handleWheel(event) {
    event.preventDefault();
    this.zoomAround(this.zoom * Math.exp(-event.deltaY * WHEEL_ZOOM_SENSITIVITY), event.clientX, event.clientY);
  }

  handlePointerDown(event) {
    this.stageElement.setPointerCapture(event.pointerId);
    this.activePointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
    this.stageElement.classList.add("lightbox-stage--dragging");
  }

  handlePointerMove(event) {
    const previous = this.activePointers.get(event.pointerId);
    if (!previous) return;
    const current = { x: event.clientX, y: event.clientY };
    if (this.activePointers.size === 2) {
      const [otherId, other] = [...this.activePointers].find(([pointerId]) => pointerId !== event.pointerId);
      const previousSpread = Math.hypot(previous.x - other.x, previous.y - other.y);
      const currentSpread = Math.hypot(current.x - other.x, current.y - other.y);
      if (previousSpread > 0) {
        this.zoomAround(this.zoom * (currentSpread / previousSpread), (current.x + other.x) / 2, (current.y + other.y) / 2);
      }
      this.activePointers.set(otherId, other);
    } else if (this.zoom > MINIMUM_ZOOM) {
      this.offsetX += current.x - previous.x;
      this.offsetY += current.y - previous.y;
      this.applyTransform();
    }
    this.activePointers.set(event.pointerId, current);
  }

  handlePointerUp(event) {
    this.activePointers.delete(event.pointerId);
    if (this.activePointers.size === 0) this.stageElement.classList.remove("lightbox-stage--dragging");
  }
}
