/**
 * Wires the page together: grabs the DOM, starts the engines, hooks up the three input
 * doors to photo intake (photo_intake.js), which hands each decoded photo to the batch flow (batch_flow.js), which drives
 * the count step (count/) and the review grid (review/) through the pipeline worker
 * (pipeline/). What the page remembers between visits (text only) lives in storage/ and
 * settings/. Everything else is a leaf that only knows its own piece.
 */

import { initializeInputDoors } from "./input_doors.js";
import { loadProcessingEngines } from "./engine_loader.js";
import { EngineStatusLine } from "./status_line.js";
import { ProgressPanel } from "./progress_panel.js";
import { PhotoIntake } from "./photo_intake.js";
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

// First visit (nothing cached yet) vs a later visit served from this computer's cache.
const FIRST_VISIT_READINESS_MESSAGE =
  `Getting ready for the first time (about ${TOTAL_DOWNLOAD_SIZE_MB} MB, once). You can paste your photo now.`;
const RETURN_VISIT_READINESS_MESSAGE = "Getting ready…";
const ENGINE_FAILURE_MESSAGE =
  "Couldn't load the check-reading tools. The first visit needs the internet; after that this page works offline. " +
  "Check the connection, then press Retry.";
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
    hintDismissButton: queryRequiredElement("input-hint-dismiss"),
    replaceConfirmElement: queryRequiredElement("replace-confirm"),
    replaceConfirmText: queryRequiredElement("replace-confirm-text"),
    replaceConfirmYesButton: queryRequiredElement("replace-confirm-yes"),
    replaceConfirmNoButton: queryRequiredElement("replace-confirm-no"),
    recordedNoticeElement: queryRequiredElement("batch-recorded"),
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

function createBatchFlow(dom, { settings, batchHistory, hints, onStartOverRequested }) {
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
    onStartOverRequested,
    reviewSection: queryRequiredElement("review-step"),
    pageElement: dom.pageElement,
  });
  queryRequiredElement("copy-all-top-button").addEventListener("click", () => reviewGrid.copyAllRows());
  queryRequiredElement("copy-all-bottom-button").addEventListener("click", () => reviewGrid.copyAllRows());
  return batchFlow;
}

/**
 * Starts the engines, reporting state through `dom.engineStatusLine`. Retry re-runs this.
 * Deliberately callback-style, not `await`ed: see the "CALLBACK STYLE IS LOAD-BEARING"
 * note at the top of engine_loader.js.
 */
function startEngineLoading(dom, batchFlow, onPipelineClientReady) {
  const isServedFromCache = Boolean(navigator.serviceWorker && navigator.serviceWorker.controller);
  dom.engineStatusLine.showLoading(isServedFromCache ? RETURN_VISIT_READINESS_MESSAGE : FIRST_VISIT_READINESS_MESSAGE);
  if (batchFlow.enginesWaiters.length > 0) dom.engineStatusLine.deferToPanel(); // the waiting photo's panel says it
  loadProcessingEngines(
    (engines) => {
      console.log("OpenCV build info:\n" + engines.openCvBuildInfo);
      dom.engineStatusLine.showReady();
      batchFlow.setPipelineClient(engines.pipelineClient);
      onPipelineClientReady(engines.pipelineClient);
    },
    (error) => {
      console.error("engine loading failed:", error);
      const retry = () => startEngineLoading(dom, batchFlow, onPipelineClientReady);
      dom.engineStatusLine.showError(ENGINE_FAILURE_MESSAGE, retry);
      batchFlow.onEnginesFailed(retry);
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
  let photoIntake = null;
  const batchFlow = createBatchFlow(dom, { settings, batchHistory, hints, onStartOverRequested: () => photoIntake.resetToEmpty() });
  photoIntake = new PhotoIntake(dom, { batchFlow, hints, toast: dom.toast });
  batchFlow.onEngineFailureShownInPanel = () => dom.engineStatusLine.deferToPanel();
  queryRequiredElement("finish-batch-button").addEventListener("click", () => {
    batchFlow.finish();
    photoIntake.resetToEmpty();
    window.scrollTo({ top: 0, behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
  });
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
    onImageFile: (file) => photoIntake.receiveFile(file),
    onNoUsableImage: (reason) => photoIntake.receiveNoUsableImage(reason),
  });
  hints.present(HINT_KEYS.PASTE_FROM_GMAIL);
  registerServiceWorker();
  startEngineLoading(dom, batchFlow, (pipelineClient) => {
    resolvePipelineClient(pipelineClient);
    settingsPanel.restoreHandwritingReaderAtStartup();
  });
}

document.addEventListener("DOMContentLoaded", main);
