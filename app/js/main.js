/**
 * Wires the page together: grabs the DOM, starts the engines, hooks up the three input
 * doors, and hands each decoded photo to the batch flow (batch_flow.js), which drives
 * the count step (count/) and the review grid (review/) through the pipeline worker
 * (pipeline/). What the page remembers between visits (text only) lives in storage/ and
 * settings/. Everything else is a leaf that only knows its own piece.
 */

import { initializeInputDoors } from "./input_doors.js";
import { isLikelyHeicImage } from "./heic_detect.js";
import { decodeAndOrientImage } from "./image_decode.js";
import { loadProcessingEngines } from "./engine_loader.js";
import { EngineStatusLine } from "./status_line.js";
import { ProgressPanel } from "./progress_panel.js";
import { STAGE_COPY } from "./progress_copy.js";
import { FirstRunHints, HINT_KEYS } from "./first_run_hints.js";
import { TOTAL_DOWNLOAD_SIZE_MB } from "./cdn_config.js";
import { StepIndicator } from "./step_indicator.js";
import { Toast } from "./toast.js";
import { BatchFlow } from "./batch_flow.js";
import { CountStep } from "./count/count_step.js";
import { ReviewGrid } from "./review/review_grid.js";
import { Lightbox } from "./review/lightbox.js";
import { openBrowserLocalStore } from "./storage/local_store.js";
import { BatchHistory } from "./storage/batch_history.js";
import { AppSettings } from "./settings/app_settings.js";
import { createSettingsPanel } from "./settings/settings_panel_setup.js";

const ACCEPTED_IMAGE_MIME_TYPES = ["image/jpeg", "image/png", "image/webp"];

const HEIC_HINT_MESSAGE =
  "This is an iPhone HEIC file. Open the attachment in Gmail and use Copy image instead.";
const UNSUPPORTED_TYPE_HINT_MESSAGE =
  "That file isn't a photo this page can open. Use a JPEG, PNG or WebP photo. " +
  "From Gmail: open the photo, right-click it, choose Copy image, then press Ctrl+V here.";
const DECODE_FAILED_HINT_MESSAGE =
  "That photo couldn't be opened. It may be damaged or only partly downloaded. " +
  "Try copying it from Gmail again (right-click the photo, Copy image, then Ctrl+V here).";
// First visit (nothing cached yet) vs a later visit served from this computer's cache.
const FIRST_VISIT_READINESS_MESSAGE =
  `Getting ready for the first time (about ${TOTAL_DOWNLOAD_SIZE_MB} MB, once). You can paste your photo now.`;
const RETURN_VISIT_READINESS_MESSAGE = "Getting ready…";
const ENGINE_FAILURE_MESSAGE =
  "Couldn't load the check-reading tools. The first visit needs the internet; after that this page works offline. " +
  "Check the connection, then press Retry.";
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
    hintContainerElement: queryRequiredElement("input-hint"),
    hintLineElement: queryRequiredElement("input-hint-line"),
    thumbnailCanvas: queryRequiredElement("photo-thumbnail"),
    photoReceivedSublineElement: queryRequiredElement("photo-received-subline"),
    progressPanel: new ProgressPanel({
      panelElement: queryRequiredElement("progress-panel"),
      headlineElement: queryRequiredElement("progress-headline"),
      detailElement: queryRequiredElement("progress-detail"),
      barElement: queryRequiredElement("progress-bar"),
      trackElement: queryRequiredElement("progress-track"),
      actionButton: queryRequiredElement("progress-action"),
    }),
    engineStatusLine: new EngineStatusLine({
      containerElement: queryRequiredElement("engine-status"),
      textElement: queryRequiredElement("engine-status-text"),
      retryButtonElement: queryRequiredElement("engine-status-retry"),
    }),
    toast: new Toast(queryRequiredElement("toast")),
  };
}

function createBatchFlow(dom, { settings, batchHistory, hints }) {
  const toast = dom.toast;
  let batchFlow = null;
  const reviewGrid = new ReviewGrid({
    rowsContainerElement: queryRequiredElement("review-rows"),
    headingElement: queryRequiredElement("review-heading"),
    emailDateInputElement: queryRequiredElement("email-date-input"),
    toast,
    settings,
    batchHistory,
    onOpenLightbox: (index) => lightbox.open(index),
    onRowsChanged: () => {},
    rereadCheckFields: (index, rotatedHalfTurn) => batchFlow.pipelineClient.rereadCheckFields(index, rotatedHalfTurn),
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
    batchHistory,
    stepIndicator: new StepIndicator(queryRequiredElement("step-indicator")),
    progressPanel: dom.progressPanel,
    recordedNotice: queryRequiredElement("batch-recorded"),
    recordedLine: queryRequiredElement("batch-recorded-line"),
    reviewSummaryElement: queryRequiredElement("review-summary"),
    hints,
    toast,
    onStartOverRequested: () => resetToEmptyDropZone(dom, batchFlow, hints),
    reviewSection: queryRequiredElement("review-step"),
    pageElement: dom.pageElement,
  });
  queryRequiredElement("copy-all-top-button").addEventListener("click", () => reviewGrid.copyAllRows());
  queryRequiredElement("copy-all-bottom-button").addEventListener("click", () => reviewGrid.copyAllRows());
  queryRequiredElement("finish-batch-button").addEventListener("click", () => {
    batchFlow.finish();
    resetToEmptyDropZone(dom, batchFlow, hints);
    window.scrollTo({ top: 0, behavior: prefersReducedMotion() ? "auto" : "smooth" });
  });
  return batchFlow;
}

const THUMBNAIL_HEIGHT_CSS_PX = 44;

export function prefersReducedMotion() {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/** An error or guidance line under the drop zone, with its own dismiss button. */
function showHint(dom, message) {
  dom.hintLineElement.textContent = message;
  dom.hintLineElement.hidden = false;
  dom.hintContainerElement.hidden = false;
}

function clearHint(dom) {
  dom.hintLineElement.hidden = true;
  dom.hintLineElement.textContent = "";
  dom.hintContainerElement.hidden = true;
}

/** A small copy of the photo in the compact drop zone, so she sees it was taken in. */
function drawThumbnail(dom, fullResCanvas) {
  const heightPx = Math.round(THUMBNAIL_HEIGHT_CSS_PX * (window.devicePixelRatio || 1));
  dom.thumbnailCanvas.height = heightPx;
  dom.thumbnailCanvas.width = Math.max(1, Math.round((heightPx * fullResCanvas.width) / fullResCanvas.height));
  const context = dom.thumbnailCanvas.getContext("2d");
  context.imageSmoothingQuality = "high";
  context.drawImage(fullResCanvas, 0, 0, dom.thumbnailCanvas.width, dom.thumbnailCanvas.height);
}

/** Shrinks the drop zone to one line: thumbnail, "Photo received", Start over. */
function showPhotoReceived(dom, decoded) {
  drawThumbnail(dom, decoded.fullResCanvas);
  // The exact "Photo received: W x H" text is what test_smoke.py checks; the size also
  // tells her at a glance whether she pasted the full photo or a small preview.
  dom.photoReceivedLineElement.textContent = `Photo received: ${decoded.width} x ${decoded.height}`;
  dom.photoReceivedSublineElement.textContent = "Paste or drop another photo any time to replace it.";
  dom.dropZonePromptElement.hidden = true;
  dom.previewContainerElement.hidden = false;
  dom.dropZoneElement.classList.add("drop-zone--compact");
}

function resetToEmptyDropZone(dom, batchFlow, hints) {
  if (batchFlow.hasOpenBatch()) batchFlow.clear();
  dom.progressPanel.hide();
  dom.thumbnailCanvas.width = 0; // the thumbnail is photo pixels; drop them
  dom.previewContainerElement.hidden = true;
  dom.dropZonePromptElement.hidden = false;
  dom.dropZoneElement.classList.remove("drop-zone--compact");
  clearHint(dom);
  hints.present(HINT_KEYS.PASTE_FROM_GMAIL);
}

/**
 * Handles a File from any of the three input doors: rejects HEIC and unsupported
 * types with an explanatory hint, otherwise decodes, orients, and starts a batch.
 * `file` is deliberately not retained past this function — see image_decode.js's
 * module docstring for why that matters.
 */
async function handleImageFile(dom, batchFlow, file, hints) {
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
  // Acknowledge at once: decoding a 12 MP photo takes a moment, and silence reads as "didn't work".
  hints.withdraw(HINT_KEYS.PASTE_FROM_GMAIL);
  dom.recordedNoticeElement.hidden = true;
  dom.progressPanel.showStage(STAGE_COPY.opening);
  let decoded;
  try {
    decoded = await decodeAndOrientImage(file);
  } catch (error) {
    console.error("decoding the photo failed:", error);
    dom.progressPanel.hide();
    showHint(dom, DECODE_FAILED_HINT_MESSAGE);
    hints.present(HINT_KEYS.PASTE_FROM_GMAIL);
    return;
  }
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

function initializeStartOverButton(dom, batchFlow, hints) {
  dom.startOverButtonElement.addEventListener("click", () => {
    if (!batchFlow.hasOpenBatch() || window.confirm(START_OVER_CONFIRM_MESSAGE)) {
      resetToEmptyDropZone(dom, batchFlow, hints);
    }
  });
}

/**
 * Starts the engines, reporting state through `dom.engineStatusLine`. Retry re-runs this.
 * Deliberately callback-style, not `await`ed: see the "CALLBACK STYLE IS LOAD-BEARING"
 * note at the top of engine_loader.js.
 */
function startEngineLoading(dom, batchFlow, onPipelineClientReady) {
  const isServedFromCache = Boolean(navigator.serviceWorker && navigator.serviceWorker.controller);
  dom.engineStatusLine.showLoading(isServedFromCache ? RETURN_VISIT_READINESS_MESSAGE : FIRST_VISIT_READINESS_MESSAGE);
  loadProcessingEngines(
    (engines) => {
      console.log("OpenCV build info:\n" + engines.openCvBuildInfo);
      dom.engineStatusLine.showReady();
      batchFlow.setPipelineClient(engines.pipelineClient);
      onPipelineClientReady(engines.pipelineClient);
    },
    (error) => {
      console.error("engine loading failed:", error);
      dom.engineStatusLine.showError(ENGINE_FAILURE_MESSAGE, () => startEngineLoading(dom, batchFlow, onPipelineClientReady));
      batchFlow.onEnginesFailed();
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
  const localStore = openBrowserLocalStore();
  const batchHistory = new BatchHistory(localStore);
  const settings = new AppSettings(localStore);
  const hints = new FirstRunHints(localStore);
  dom.recordedNoticeElement = queryRequiredElement("batch-recorded");
  const batchFlow = createBatchFlow(dom, { settings, batchHistory, hints });
  // The PipelineClient is a plain class (not a thenable), so resolving a Promise with it is safe.
  let resolvePipelineClient;
  const pipelineClientReady = new Promise((resolve) => { resolvePipelineClient = resolve; });
  const settingsPanel = createSettingsPanel({
    settings,
    batchHistory,
    localStore,
    reviewGrid: batchFlow.reviewGrid,
    whenPipelineClientReady: () => pipelineClientReady,
  });
  // Finish batch adds names; keep an open panel's list current.
  queryRequiredElement("finish-batch-button").addEventListener("click", () => settingsPanel.renderAll());
  // For the Playwright tests: a text-only view (quads, step, timings, field values and
  // states; never pixels), and field reads fed through the same path as the worker's.
  window.__checkTranscriberDebug = {
    describe: () => batchFlow.describeForDebug(),
    setFieldReads: (checkIndex, rawReads) => batchFlow.reviewGrid.receiveFieldReads(checkIndex, rawReads),
  };

  initializeInputDoors({
    dropZoneElement: dom.dropZoneElement,
    dropZonePromptElement: dom.dropZonePromptElement,
    fileInputElement: dom.fileInputElement,
    dropOverlayElement: queryRequiredElement("drop-overlay"),
    onImageFile: (file) => handleImageFile(dom, batchFlow, file, hints),
    onNoUsableImage: (reason) => handleNoUsableImage(dom, reason),
  });
  queryRequiredElement("input-hint-dismiss").addEventListener("click", () => clearHint(dom));
  initializeStartOverButton(dom, batchFlow, hints);
  hints.present(HINT_KEYS.PASTE_FROM_GMAIL);
  registerServiceWorker();
  startEngineLoading(dom, batchFlow, (pipelineClient) => {
    resolvePipelineClient(pipelineClient);
    settingsPanel.restoreHandwritingReaderAtStartup();
  });
}

document.addEventListener("DOMContentLoaded", main);
