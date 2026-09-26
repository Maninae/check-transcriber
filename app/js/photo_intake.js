/**
 * Photo intake: what happens between a file arriving (paste, drop, browse) and the batch
 * flow taking it, plus the photo strip at the top of the page (thumbnail, "Photo
 * received", Start over) and its inline "replace?" question.
 *
 * Order for every arriving file, so she never answers a question about a file that
 * would be refused anyway:
 * 1. Refuse HEIC and non-photos first. With no batch open the reason shows under the drop
 *    zone; with a batch open it is an info toast (visible at any scroll position) and the
 *    batch is untouched.
 * 2. If there are checks to lose, ask inline in the photo strip ("Replace these 6 checks
 *    with the new photo?"), holding the new file until she picks. Never window.confirm.
 * 3. "Opening the photo…" at once, decode, then hand the canvas to the batch flow.
 *
 * `file` is held only while the question is open, and never after: see image_decode.js.
 */

import { isLikelyHeicImage } from "./heic_detect.js";
import { decodeAndOrientImage } from "./image_decode.js";
import { STAGE_COPY } from "./progress_copy.js";
import { HINT_KEYS } from "./first_run_hints.js";

const ACCEPTED_IMAGE_MIME_TYPES = ["image/jpeg", "image/png", "image/webp"];
const THUMBNAIL_HEIGHT_CSS_PX = 44;
// Below this long side the checks are likely too small to read well (a Gmail thumbnail, not the photo).
const SMALL_PREVIEW_LONG_SIDE_PX = 1500;

export const INTAKE_MESSAGES = Object.freeze({
  heic: "This is an iPhone HEIC file. Open the attachment in Gmail and use Copy image instead.",
  unsupported: "That file isn't a photo this page can open. Use a JPEG, PNG or WebP photo. "
    + "From Gmail: open the photo, right-click it, choose Copy image, then press Ctrl+V here.",
  urlOnly: "That carried a link to the image, not the image itself. In Gmail, right-click the "
    + "photo and choose Copy image, then paste here (Ctrl+V) instead.",
  decodeFailed: "That photo couldn't be opened. It may be damaged or only partly downloaded. "
    + "Try copying it from Gmail again (right-click the photo, Copy image, then Ctrl+V here).",
  smallPreview: "This looks like a small preview, so the checks may be hard to read. "
    + "If you can, open the attachment in Gmail and copy the full photo.",
  keptBatch: "Your current checks are unchanged.",
});

function pluralizeChecks(count) {
  return `${count} ${count === 1 ? "check" : "checks"}`;
}

export class PhotoIntake {
  /**
   * `dom`: the elements from main.js (drop zone, prompt, preview strip, thumbnail, received
   * line and subline, hint box, replace bar, progress panel, recorded notice).
   * `services`: `{ batchFlow, hints, toast }`.
   */
  constructor(dom, { batchFlow, hints, toast }) {
    Object.assign(this, { dom, batchFlow, hints, toast });
    this.pendingDecision = null; // { onConfirm } while the replace bar is open
    dom.replaceConfirmYesButton.addEventListener("click", () => this.answerReplace(true));
    dom.replaceConfirmNoButton.addEventListener("click", () => this.answerReplace(false));
    dom.startOverButtonElement.addEventListener("click", () => this.requestStartOver());
    dom.hintDismissButton.addEventListener("click", () => this.clearHint());
  }

  showHint(message) {
    this.dom.hintLineElement.textContent = message;
    this.dom.hintLineElement.hidden = false;
    this.dom.hintContainerElement.hidden = false;
  }

  clearHint() {
    this.dom.hintLineElement.hidden = true;
    this.dom.hintLineElement.textContent = "";
    this.dom.hintContainerElement.hidden = true;
  }

  /** A refusal: under the drop zone when nothing is open, a toast when a batch is (she may be scrolled far down). */
  refuse(message) {
    if (this.batchFlow.hasOpenBatch()) {
      this.toast.show(`${message} ${INTAKE_MESSAGES.keptBatch}`, { tone: "info" });
    } else {
      this.showHint(message);
    }
  }

  /** A drop that carried no image bytes (a link from another tab, a PDF). */
  receiveNoUsableImage(reason) {
    this.refuse(reason === "url-reference-only" ? INTAKE_MESSAGES.urlOnly : INTAKE_MESSAGES.unsupported);
  }

  /** A File from any door. */
  async receiveFile(file) {
    const photoArrivedAt = performance.now();
    if (await isLikelyHeicImage(file)) {
      this.refuse(INTAKE_MESSAGES.heic);
      return;
    }
    if (!ACCEPTED_IMAGE_MIME_TYPES.includes(file.type)) {
      this.refuse(INTAKE_MESSAGES.unsupported);
      return;
    }
    const checkCount = this.batchFlow.countChecksAtStake();
    if (checkCount > 0) {
      // Timed from her answer: the pipeline's clock should not include her reading the question.
      this.askReplace(`Replace these ${pluralizeChecks(checkCount)} with the new photo?`, "Replace", () => this.openPhoto(file, performance.now()));
      return;
    }
    this.openPhoto(file, photoArrivedAt);
  }

  /** Start over: ask only when there are checks to lose. */
  requestStartOver() {
    const checkCount = this.batchFlow.countChecksAtStake();
    if (checkCount === 0) {
      this.resetToEmpty();
      return;
    }
    this.askReplace(`Start over? These ${pluralizeChecks(checkCount)} will be cleared.`, "Start over", () => this.resetToEmpty());
  }

  /** The inline question in the photo strip; scrolled into view, focus on the safe answer. */
  askReplace(question, confirmLabel, onConfirm) {
    this.pendingDecision = { onConfirm };
    this.dom.replaceConfirmText.textContent = question;
    this.dom.replaceConfirmYesButton.textContent = confirmLabel;
    this.dom.replaceConfirmElement.hidden = false;
    this.dom.dropZoneElement.scrollIntoView({ behavior: "auto", block: "start" });
    this.dom.replaceConfirmNoButton.focus({ preventScroll: true });
  }

  answerReplace(confirmed) {
    const decision = this.pendingDecision;
    this.pendingDecision = null; // drops the held file either way
    this.dom.replaceConfirmElement.hidden = true;
    if (confirmed && decision) decision.onConfirm();
  }

  async openPhoto(file, photoArrivedAt) {
    this.clearHint();
    if (this.batchFlow.hasOpenBatch()) this.batchFlow.clear();
    // Acknowledge at once: decoding a 12 MP photo takes a moment, and silence reads as "didn't work".
    this.hints.withdraw(HINT_KEYS.PASTE_FROM_GMAIL);
    this.dom.recordedNoticeElement.hidden = true;
    this.dom.progressPanel.returnHome();
    this.dom.progressPanel.showStage(STAGE_COPY.opening);
    let decoded;
    try {
      decoded = await decodeAndOrientImage(file);
    } catch (error) {
      console.error("decoding the photo failed:", error);
      this.dom.progressPanel.hide();
      this.showHint(INTAKE_MESSAGES.decodeFailed);
      this.hints.present(HINT_KEYS.PASTE_FROM_GMAIL);
      return;
    }
    // The working copy was milestone 1's detection placeholder; the detector makes its own.
    decoded.workingCanvas.width = 0;
    this.showPhotoReceived(decoded);
    this.batchFlow.startWithPhoto(decoded.fullResCanvas, photoArrivedAt);
  }

  drawThumbnail(fullResCanvas) {
    const canvas = this.dom.thumbnailCanvas;
    canvas.height = Math.round(THUMBNAIL_HEIGHT_CSS_PX * (window.devicePixelRatio || 1));
    canvas.width = Math.max(1, Math.round((canvas.height * fullResCanvas.width) / fullResCanvas.height));
    const context = canvas.getContext("2d");
    context.imageSmoothingQuality = "high";
    context.drawImage(fullResCanvas, 0, 0, canvas.width, canvas.height);
  }

  /** The drop zone shrinks to one strip: thumbnail, "Photo received", Start over. */
  showPhotoReceived(decoded) {
    this.drawThumbnail(decoded.fullResCanvas);
    const line = this.dom.photoReceivedLineElement;
    line.textContent = "Photo received";
    // The decoded size stays machine-readable (test_smoke.py checks EXIF rotation through it).
    line.dataset.photoWidth = String(decoded.width);
    line.dataset.photoHeight = String(decoded.height);
    const isSmallPreview = Math.max(decoded.width, decoded.height) < SMALL_PREVIEW_LONG_SIDE_PX;
    this.dom.photoReceivedSublineElement.textContent = isSmallPreview ? INTAKE_MESSAGES.smallPreview : "";
    this.dom.dropZonePromptElement.hidden = true;
    this.dom.previewContainerElement.hidden = false;
    this.dom.dropZoneElement.classList.add("drop-zone--compact");
  }

  /** Back to the empty drop zone (Start over, Finish batch, a failed batch's Start over). */
  resetToEmpty() {
    this.answerReplace(false);
    if (this.batchFlow.hasOpenBatch()) this.batchFlow.clear();
    this.toast.hide();
    this.dom.progressPanel.hide();
    this.dom.progressPanel.returnHome();
    this.dom.thumbnailCanvas.width = 0; // the thumbnail is photo pixels; drop them
    this.dom.previewContainerElement.hidden = true;
    this.dom.dropZonePromptElement.hidden = false;
    this.dom.dropZoneElement.classList.remove("drop-zone--compact");
    this.clearHint();
    this.hints.present(HINT_KEYS.PASTE_FROM_GMAIL);
  }
}
