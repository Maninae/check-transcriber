/**
 * The count step (spec 4.2): the photo fit to width with a numbered outline per check,
 * a "Found N checks" header, and cheap fixes before Continue:
 * - Add a check: press the button, drag a rough rectangle; the worker re-fits it to the
 *   check's edges (pipeline/drawn_rectangle_refit.js).
 * - Remove: click an outline, then its X or press Delete/Backspace.
 * - Adjust: drag any corner handle (quads, not boxes).
 * Numbers stay in reading order (re-sorted after every add or remove). The step is never
 * skipped; Continue hands the confirmed quads to the review step and locks this view
 * so the numbered photo stays above the grid for cross-checking.
 */

import { renderCountOverlay } from "./count_overlay.js";
import { describeCountHeader } from "./count_header_text.js";
import { sortIntoReadingOrder } from "../pipeline/reading_order.js";

const MINIMUM_DRAWN_SIZE_SCREEN_PX = 20;

export class CountStep {
  /** `dom`: section, heading, note, photo canvas, overlay svg, add and continue buttons.
   *  `callbacks`: `refitDrawnRectangle(drawnCorners, otherCornerSets) -> Promise<{ corners }>`, `onContinue(cornerSets)`. */
  constructor(dom, callbacks) {
    this.dom = dom;
    this.callbacks = callbacks;
    this.photoCanvas = null;
    this.quads = [];
    this.nextQuadId = 1;
    this.selectedQuadId = null;
    this.isDrawing = false;
    this.drawRectangle = null;
    this.cornerDrag = null; // { quadId, cornerIndex, pointerId }
    this.wasEdited = false;
    this.locked = false;
    dom.addCheckButton.addEventListener("click", () => this.setDrawingMode(!this.isDrawing));
    dom.continueButton.addEventListener("click", () => this.continueToReview());
    dom.overlayElement.addEventListener("pointerdown", (event) => this.handlePointerDown(event));
    dom.overlayElement.addEventListener("pointermove", (event) => this.handlePointerMove(event));
    dom.overlayElement.addEventListener("pointerup", (event) => this.handlePointerUp(event));
    document.addEventListener("keydown", (event) => this.handleKeydown(event));
    window.addEventListener("resize", () => { if (this.photoCanvas) this.render(); });
  }

  /** Shows the photo and the detector's quads (`[{ corners, confident }]`, reading order). */
  show(fullResCanvas, detections) {
    this.photoCanvas = fullResCanvas;
    this.quads = detections.map(({ corners, confident }) => ({ id: this.nextQuadId++, corners, confident, pending: false }));
    this.selectedQuadId = null;
    this.wasEdited = false;
    this.locked = false;
    this.dom.sectionElement.classList.remove("count-step--locked");
    this.dom.sectionElement.hidden = false;
    this.drawPhoto();
    this.setDrawingMode(this.quads.length === 0);
    this.render();
  }

  drawPhoto() {
    const frameWidth = this.dom.photoCanvas.parentElement.clientWidth;
    const targetWidth = Math.min(this.photoCanvas.width, Math.round(frameWidth * (window.devicePixelRatio || 1)));
    this.dom.photoCanvas.width = targetWidth;
    this.dom.photoCanvas.height = Math.round((targetWidth * this.photoCanvas.height) / this.photoCanvas.width);
    const context = this.dom.photoCanvas.getContext("2d");
    context.imageSmoothingQuality = "high";
    context.drawImage(this.photoCanvas, 0, 0, this.dom.photoCanvas.width, this.dom.photoCanvas.height);
  }

  render() {
    const displayScale = this.dom.overlayElement.getBoundingClientRect().width / this.photoCanvas.width || 1;
    renderCountOverlay(this.dom.overlayElement, {
      imageWidth: this.photoCanvas.width,
      imageHeight: this.photoCanvas.height,
      displayScale,
      quads: this.quads,
      selectedQuadId: this.selectedQuadId,
      drawRectangle: this.drawRectangle,
    });
    const header = describeCountHeader(this.quads, this.wasEdited);
    this.dom.headingElement.textContent = header.heading;
    this.dom.noteElement.textContent = header.note;
    this.dom.continueButton.disabled = this.quads.length === 0 || this.quads.some(({ pending }) => pending);
    this.callbacks.onCountChanged?.(this.quads.length);
  }

  setDrawingMode(isDrawing) {
    this.isDrawing = isDrawing;
    this.dom.addCheckButton.setAttribute("aria-pressed", String(isDrawing));
    this.dom.overlayElement.classList.toggle("count-overlay--drawing", isDrawing);
    if (isDrawing) this.selectedQuadId = null;
  }

  toPhotoPoint(event) {
    const rect = this.dom.overlayElement.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * this.photoCanvas.width;
    const y = ((event.clientY - rect.top) / rect.height) * this.photoCanvas.height;
    return [Math.min(Math.max(x, 0), this.photoCanvas.width - 1), Math.min(Math.max(y, 0), this.photoCanvas.height - 1)];
  }

  handlePointerDown(event) {
    if (this.locked) return;
    const target = event.target;
    const [x, y] = this.toPhotoPoint(event);
    if (target.closest("[data-remove-quad-id]")) {
      this.removeQuad(Number(target.closest("[data-remove-quad-id]").dataset.removeQuadId));
      return;
    }
    if (target.dataset.handleQuadId) {
      this.cornerDrag = { quadId: Number(target.dataset.handleQuadId), cornerIndex: Number(target.dataset.cornerIndex) };
      this.selectedQuadId = this.cornerDrag.quadId;
    } else if (this.isDrawing) {
      this.drawRectangle = { startX: x, startY: y, endX: x, endY: y };
    } else {
      this.selectedQuadId = target.dataset.quadId ? Number(target.dataset.quadId) : null;
    }
    this.dom.overlayElement.setPointerCapture(event.pointerId);
    this.render();
  }

  handlePointerMove(event) {
    if (this.cornerDrag) {
      const quad = this.quads.find(({ id }) => id === this.cornerDrag.quadId);
      quad.corners[this.cornerDrag.cornerIndex] = this.toPhotoPoint(event);
      this.render();
    } else if (this.drawRectangle) {
      [this.drawRectangle.endX, this.drawRectangle.endY] = this.toPhotoPoint(event);
      this.render();
    }
  }

  handlePointerUp() {
    if (this.cornerDrag) {
      const quad = this.quads.find(({ id }) => id === this.cornerDrag.quadId);
      quad.confident = true; // the operator has placed it by hand
      this.cornerDrag = null;
      this.wasEdited = true;
      this.render();
    } else if (this.drawRectangle) {
      const drawn = this.drawRectangle;
      this.drawRectangle = null;
      this.setDrawingMode(false);
      this.finishDrawnRectangle(drawn);
    }
  }

  finishDrawnRectangle({ startX, startY, endX, endY }) {
    const displayScale = this.dom.overlayElement.getBoundingClientRect().width / this.photoCanvas.width;
    const tooSmall = Math.min(Math.abs(endX - startX), Math.abs(endY - startY)) * displayScale < MINIMUM_DRAWN_SIZE_SCREEN_PX;
    if (tooSmall) {
      this.render();
      return;
    }
    const [left, right] = [Math.min(startX, endX), Math.max(startX, endX)];
    const [top, bottom] = [Math.min(startY, endY), Math.max(startY, endY)];
    const drawnCorners = [[left, top], [right, top], [right, bottom], [left, bottom]];
    const otherCornerSets = this.quads.map(({ corners }) => corners);
    const quad = { id: this.nextQuadId++, corners: drawnCorners, confident: true, pending: true };
    this.quads.push(quad);
    this.wasEdited = true;
    this.reorderQuads();
    this.callbacks.refitDrawnRectangle(drawnCorners, otherCornerSets).then(({ corners }) => {
      quad.corners = corners;
    }, (error) => {
      console.error("re-fitting the drawn check failed; keeping the rectangle as drawn:", error);
    }).finally(() => {
      quad.pending = false;
      this.reorderQuads();
    });
  }

  removeQuad(quadId) {
    this.quads = this.quads.filter(({ id }) => id !== quadId);
    this.selectedQuadId = null;
    this.wasEdited = true;
    this.reorderQuads();
  }

  reorderQuads() {
    this.quads = sortIntoReadingOrder(this.quads, ({ corners }) => corners);
    this.render();
  }

  handleKeydown(event) {
    if (this.locked || this.dom.sectionElement.hidden) return;
    const typingInField = event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement;
    if (typingInField) return;
    if ((event.key === "Delete" || event.key === "Backspace") && this.selectedQuadId !== null) {
      event.preventDefault();
      this.removeQuad(this.selectedQuadId);
    } else if (event.key === "Escape") {
      this.drawRectangle = null;
      this.setDrawingMode(false);
      this.selectedQuadId = null;
      this.render();
    }
  }

  continueToReview() {
    this.locked = true;
    this.selectedQuadId = null;
    this.setDrawingMode(false);
    this.dom.sectionElement.classList.add("count-step--locked");
    this.render();
    this.callbacks.onContinue(this.quads.map(({ corners }) => corners.map((corner) => [...corner])));
  }

  /** Current quads' corners, reading order (the debug hook and tests read this). */
  getCornerSets() {
    return this.quads.map(({ corners }) => corners.map((corner) => [...corner]));
  }

  /** Hides the step and drops every reference to the photo. */
  clear() {
    this.quads = [];
    this.photoCanvas = null;
    this.dom.photoCanvas.width = 0;
    this.dom.overlayElement.replaceChildren();
    this.dom.sectionElement.hidden = true;
    this.dom.sectionElement.classList.remove("count-step--locked");
  }
}
