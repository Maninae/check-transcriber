/**
 * Wires the page together: grabs the DOM, starts the engines, hooks up the three input
 * doors, and hands each decoded photo to the batch flow (batch_flow.js), which drives
 * the count step (count/) and the review grid (review/) through the pipeline worker
 * (pipeline/). Everything else is a leaf that only knows its own piece.
 */

import { initializeInputDoors } from "./input_doors.js";
import { isLikelyHeicImage } from "./heic_detect.js";
import { decodeAndOrientImage } from "./image_decode.js";
import { loadProcessingEngines, READINESS_MESSAGE, FAILURE_MESSAGE } from "./engine_loader.js";
import { EngineStatusLine } from "./status_line.js";
import { StepIndicator } from "./step_indicator.js";
import { Toast } from "./toast.js";
import { BatchFlow } from "./batch_flow.js";
import { CountStep } from "./count/count_step.js";
import { ReviewGrid } from "./review/review_grid.js";
import { Lightbox } from "./review/lightbox.js";

const ACCEPTED_IMAGE_MIME_TYPES = ["image/jpeg", "image/png", "image/webp"];

const HEIC_HINT_MESSAGE =
  "This is an iPhone HEIC file. Open the attachment in Gmail and use Copy image instead.";
const UNSUPPORTED_TYPE_HINT_MESSAGE =
  "That doesn't look like a photo this tool can read. Use a JPEG, PNG, or WebP image.";
const URL_ONLY_DROP_HINT_MESSAGE =
  "That carried a link to the image, not the image itself. In Gmail, right-click the " +
  "photo and choose Copy image, then paste here (Ctrl+V) instead.";
const START_OVER_CONFIRM_MESSAGE = "Start over with a new photo? The current batch will be cleared.";

function queryRequiredElement(id) {
  const element = document.getElementById(id);
  if (!element) {
    throw new Error(`expected element #${id} to exist in index.html`);
  }
  return element;
}

function initializeDom() {
  return {
    pageElement: document.querySelector(".page"),
    dropZoneElement: queryRequiredElement("drop-zone"),
    dropZonePromptElement: queryRequiredElement("drop-zone-prompt"),
    fileInputElement: queryRequiredElement("file-input"),
    previewContainerElement: queryRequiredElement("photo-preview-container"),
    photoReceivedLineElement: queryRequiredElement("photo-received-line"),
    startOverButtonElement: queryRequiredElement("start-over-button"),
    hintLineElement: queryRequiredElement("input-hint-line"),
    engineStatusLine: new EngineStatusLine({
      containerElement: queryRequiredElement("engine-status"),
      textElement: queryRequiredElement("engine-status-text"),
      retryButtonElement: queryRequiredElement("engine-status-retry"),
    }),
  };
}

function createBatchFlow(dom) {
  const toast = new Toast(queryRequiredElement("toast"));
  let batchFlow = null;
  const reviewGrid = new ReviewGrid({
    rowsContainerElement: queryRequiredElement("review-rows"),
    headingElement: queryRequiredElement("review-heading"),
    toast,
    onOpenLightbox: (index) => lightbox.open(index),
    onRowsChanged: () => {},
  });
  const lightbox = new Lightbox({
    lightboxElement: queryRequiredElement("lightbox"),
    stageElement: queryRequiredElement("lightbox-stage"),
    canvasElement: queryRequiredElement("lightbox-canvas"),
    captionElement: queryRequiredElement("lightbox-caption"),
    previousButton: queryRequiredElement("lightbox-previous"),
    nextButton: queryRequiredElement("lightbox-next"),
    rotateButton: queryRequiredElement("lightbox-rotate"),
    closeButton: queryRequiredElement("lightbox-close"),
  }, reviewGrid);
  const countStep = new CountStep({
    sectionElement: queryRequiredElement("count-step"),
    headingElement: queryRequiredElement("count-step-heading"),
    noteElement: queryRequiredElement("count-step-note"),
    photoCanvas: queryRequiredElement("count-photo-canvas"),
    overlayElement: queryRequiredElement("count-overlay"),
    addCheckButton: queryRequiredElement("add-check-button"),
    continueButton: queryRequiredElement("count-continue-button"),
  }, {
    refitDrawnRectangle: (drawnCorners, otherCornerSets) => batchFlow.pipelineClient.refitDrawnRectangle(drawnCorners, otherCornerSets),
    onContinue: (cornerSets) => batchFlow.continueToReview(cornerSets),
    onCountChanged: (count) => { if (batchFlow.step === "count") batchFlow.stepIndicator.show("count", count); },
  });
  batchFlow = new BatchFlow({
    countStep,
    reviewGrid,
    stepIndicator: new StepIndicator(queryRequiredElement("step-indicator")),
    progressLine: queryRequiredElement("pipeline-progress-line"),
    recordedLine: queryRequiredElement("batch-recorded-line"),
    reviewSection: queryRequiredElement("review-step"),
    pageElement: dom.pageElement,
  });
  queryRequiredElement("copy-all-top-button").addEventListener("click", () => reviewGrid.copyAllRows());
  queryRequiredElement("copy-all-bottom-button").addEventListener("click", () => reviewGrid.copyAllRows());
  queryRequiredElement("finish-batch-button").addEventListener("click", () => {
    batchFlow.finish();
    resetToEmptyDropZone(dom, batchFlow);
  });
  return batchFlow;
}

function showHint(dom, message) {
  dom.hintLineElement.textContent = message;
  dom.hintLineElement.hidden = false;
}

function clearHint(dom) {
  dom.hintLineElement.hidden = true;
  dom.hintLineElement.textContent = "";
}

/** Shrinks the drop zone to one line with the photo's size and Start over. */
function showPhotoReceived(dom, { width, height }) {
  dom.photoReceivedLineElement.textContent = `Photo received: ${width} x ${height}`;
  dom.dropZonePromptElement.hidden = true;
  dom.previewContainerElement.hidden = false;
  dom.dropZoneElement.classList.add("drop-zone--compact");
}

function resetToEmptyDropZone(dom, batchFlow) {
  if (batchFlow.hasOpenBatch()) batchFlow.clear();
  dom.previewContainerElement.hidden = true;
  dom.dropZonePromptElement.hidden = false;
  dom.dropZoneElement.classList.remove("drop-zone--compact");
  clearHint(dom);
}

/**
 * Handles a File from any of the three input doors: rejects HEIC and unsupported
 * types with an explanatory hint, otherwise decodes, orients, and starts a batch.
 * `file` is deliberately not retained past this function — see image_decode.js's
 * module docstring for why that matters.
 */
async function handleImageFile(dom, batchFlow, file) {
  const photoArrivedAt = performance.now();
  if (batchFlow.hasOpenBatch() && !window.confirm(START_OVER_CONFIRM_MESSAGE)) {
    return;
  }
  clearHint(dom);

  if (await isLikelyHeicImage(file)) {
    showHint(dom, HEIC_HINT_MESSAGE);
    return;
  }
  if (!ACCEPTED_IMAGE_MIME_TYPES.includes(file.type)) {
    showHint(dom, UNSUPPORTED_TYPE_HINT_MESSAGE);
    return;
  }

  if (batchFlow.hasOpenBatch()) batchFlow.clear();
  const decoded = await decodeAndOrientImage(file);
  // The working copy was milestone 1's detection placeholder; the detector makes its own.
  decoded.workingCanvas.width = 0;
  showPhotoReceived(dom, decoded);
  batchFlow.startWithPhoto(decoded.fullResCanvas, photoArrivedAt);
}

function handleNoUsableImage(dom, reason) {
  if (reason === "url-reference-only") {
    showHint(dom, URL_ONLY_DROP_HINT_MESSAGE);
  } else {
    showHint(dom, UNSUPPORTED_TYPE_HINT_MESSAGE);
  }
}

function initializeStartOverButton(dom, batchFlow) {
  dom.startOverButtonElement.addEventListener("click", () => {
    if (window.confirm(START_OVER_CONFIRM_MESSAGE)) {
      resetToEmptyDropZone(dom, batchFlow);
    }
  });
}

/**
 * Starts the engines, reporting state through `dom.engineStatusLine`. Retry re-runs this.
 * Deliberately callback-style, not `await`ed: see the "CALLBACK STYLE IS LOAD-BEARING"
 * note at the top of engine_loader.js.
 */
function startEngineLoading(dom, batchFlow) {
  dom.engineStatusLine.showLoading(READINESS_MESSAGE);
  loadProcessingEngines(
    (engines) => {
      console.log("OpenCV build info:\n" + engines.openCvBuildInfo);
      dom.engineStatusLine.showReady();
      batchFlow.setPipelineClient(engines.pipelineClient);
    },
    (error) => {
      console.error("engine loading failed:", error);
      dom.engineStatusLine.showError(FAILURE_MESSAGE, () => startEngineLoading(dom, batchFlow));
    },
  );
}

function registerServiceWorker() {
  if (!("serviceWorker" in navigator)) {
    return;
  }
  // Relative path/scope so this resolves correctly both at http://localhost/ and at
  // the GitHub Pages subpath https://maninae.github.io/check-transcriber/.
  navigator.serviceWorker.register("./sw.js").catch((error) => {
    console.warn("service worker registration failed:", error);
  });
}

function main() {
  const dom = initializeDom();
  const batchFlow = createBatchFlow(dom);
  // Read-only view for the Playwright tests (quads, step, timings; never pixels).
  window.__checkTranscriberDebug = { describe: () => batchFlow.describeForDebug() };

  initializeInputDoors({
    dropZoneElement: dom.dropZoneElement,
    dropZonePromptElement: dom.dropZonePromptElement,
    fileInputElement: dom.fileInputElement,
    onImageFile: (file) => handleImageFile(dom, batchFlow, file),
    onNoUsableImage: (reason) => handleNoUsableImage(dom, reason),
  });
  initializeStartOverButton(dom, batchFlow);
  registerServiceWorker();
  startEngineLoading(dom, batchFlow);
}

document.addEventListener("DOMContentLoaded", main);
