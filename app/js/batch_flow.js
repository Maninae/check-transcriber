/**
 * One batch, start to finish (spec 4): a decoded photo goes to the pipeline worker for
 * detection, the count step confirms the quads, the worker orients, crops and reads each
 * check and the review grid fills row by row, and Finish batch records the batch's payer
 * names and check numbers (text only, storage/batch_history.js) and clears the images.
 *
 * Owns the batch-level state (which step, timings) and the "leave page?" guard. The
 * photo's pixels live in three places only while a batch is open: the decoded canvas
 * (main.js), the count step's display canvas, and the worker's copy. Finish batch and
 * Start over release all three.
 *
 * Feedback (so nothing ever looks stuck): every stage is reported through the progress
 * panel (progress_panel.js) in plain words (progress_copy.js), with a determinate bar once
 * the work is countable; the review heading carries a live summary ("All 6 checks read.
 * 3 fields need a look"); failures offer Start over; Finish ends on a toast.
 */

import { todayIso } from "./fields/date_parsing.js";
import { collectReviewedRowValues } from "./review/review_field_editing.js";
import { STAGE_COPY, describeReviewSummary, describeWorkerStage, parseCheckOfTotal, pluralizeChecks, reviewProgressFraction } from "./progress_copy.js";
import { HINT_KEYS } from "./first_run_hints.js";

const DETECTION_FAILED_MESSAGE = "Something went wrong finding the checks. Start over and paste the photo again.";
const STRAIGHTENING_FAILED_MESSAGE = "Something went wrong straightening the checks. Start over and paste the photo again.";
const ENGINES_FAILED_WHILE_WAITING_MESSAGE = "Your photo is waiting, but the check-reading tools didn't load. Press Retry below.";

function scrollBehavior() {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth";
}

const UNSAVED_CHECKS_MESSAGE = (count) =>
  `You have ${count} ${count === 1 ? "check" : "checks"} that have not been copied.`;

export class BatchFlow {
  /**
   * `parts`: `{ countStep, reviewGrid, batchHistory, stepIndicator, progressPanel, recordedNotice,
   * recordedLine, reviewSummaryElement, hints, toast, onStartOverRequested, reviewSection, pageElement }`.
   */
  constructor(parts) {
    Object.assign(this, parts);
    this.pipelineClient = null;
    this.enginesWaiters = [];
    this.fullResCanvas = null;
    this.step = null;
    this.timings = {};
    this.detectedQuads = null;
    this.orientedQuads = [];
    this.batchToken = 0; // bumped per Continue and per clear; read-chain callbacks from older batches are dropped
    this.readsRunning = false; // Continue's crop-and-read chain is still going
    this.reviewGrid.onFieldStatesChanged = () => this.updateReviewSummary();
    window.addEventListener("beforeunload", (event) => this.guardUnload(event));
  }

  /** Called once the engines are ready; runs any photo that arrived before that. */
  setPipelineClient(pipelineClient) {
    this.pipelineClient = pipelineClient;
    const waiters = this.enginesWaiters;
    this.enginesWaiters = [];
    waiters.forEach((run) => run());
  }

  hasOpenBatch() {
    return this.step !== null;
  }

  setStep(step, checkCount = null) {
    this.step = step;
    this.stepIndicator.show(step, checkCount);
    this.pageElement.classList.toggle("page--wide", step === "count" || step === "review");
  }

  /** The engines failed to load (main.js offers Retry); a photo waiting on them says so. */
  onEnginesFailed() {
    if (this.enginesWaiters.length > 0) this.progressPanel.showError(ENGINES_FAILED_WHILE_WAITING_MESSAGE);
  }

  showFailure(message) {
    this.progressPanel.showError(message, "Start over", () => this.onStartOverRequested());
  }

  /** The line beside the review heading: still reading, or what is left to look at. */
  updateReviewSummary() {
    if (this.step !== "review") {
      this.reviewSummaryElement.textContent = "";
      return;
    }
    const counts = this.reviewGrid.countFieldStates();
    const summary = describeReviewSummary({
      checkCount: this.reviewGrid.getCheckCount(),
      unreadFieldCount: counts.unread,
      flaggedFieldCount: counts.flagged,
      handwritingPending: this.readsRunning && counts.unread === 0,
    });
    this.reviewSummaryElement.textContent = summary.text;
    this.reviewSummaryElement.dataset.tone = summary.tone;
  }

  /** A photo was decoded; `photoArrivedAt` is performance.now() at paste/drop/pick. */
  startWithPhoto(fullResCanvas, photoArrivedAt) {
    this.fullResCanvas = fullResCanvas;
    this.recordedNotice.hidden = true;
    this.timings = { photoArrivedAt };
    this.detectedQuads = null;
    this.setStep("photo");
    const run = () => this.detect(fullResCanvas);
    if (this.pipelineClient) {
      run();
    } else {
      this.progressPanel.showStage(STAGE_COPY.waitingForTools);
      this.enginesWaiters.push(run);
    }
  }

  detect(fullResCanvas) {
    this.progressPanel.showStage(STAGE_COPY.findingChecks);
    this.pipelineClient.loadPhoto(fullResCanvas)
      .then(() => this.pipelineClient.detectChecks((text) => {
        if (fullResCanvas === this.fullResCanvas) this.progressPanel.showStage(describeWorkerStage(text));
      }))
      .then((detections) => {
        if (fullResCanvas !== this.fullResCanvas) return; // replaced while detecting
        this.detectedQuads = detections.map(({ corners }) => corners);
        this.progressPanel.hide();
        this.setStep("count", detections.length);
        this.hints.present(HINT_KEYS.FIX_OUTLINES);
        this.countStep.show(fullResCanvas, detections);
        // Instant, not smooth: the photo must not slide under the pointer while she reaches for a corner.
        this.countStep.dom.sectionElement.scrollIntoView({ behavior: "auto", block: "start" });
        this.timings.countStepShownAt = performance.now();
      })
      .catch((error) => {
        if (fullResCanvas !== this.fullResCanvas) return;
        console.error("check detection failed:", error);
        this.showFailure(DETECTION_FAILED_MESSAGE);
      });
  }

  /** Continue on the count step: orient and crop every confirmed quad, row by row. */
  continueToReview(cornerSets) {
    this.setStep("review", cornerSets.length);
    this.timings.continuedAt = performance.now();
    this.reviewSection.hidden = false;
    this.orientedQuads = [];
    this.batchToken += 1;
    const batchToken = this.batchToken;
    const isCurrent = () => batchToken === this.batchToken; // Start over / Finish / a new Continue retire this chain
    this.hints.withdraw(HINT_KEYS.FIX_OUTLINES);
    this.reviewGrid.start(cornerSets.length);
    this.hints.present(HINT_KEYS.REVIEW_GUIDE);
    this.readsRunning = true;
    const checkCount = cornerSets.length;
    this.progressPanel.placeAtStartOf(this.reviewSection);
    this.progressPanel.showStage(describeWorkerStage(`Straightening check 1 of ${checkCount}`), 0);
    this.updateReviewSummary();
    this.reviewSection.scrollIntoView({ behavior: scrollBehavior(), block: "start" });
    // Timings: every crop shown, every check read by the default readers (the fully populated
    // grid of spec section 5's budget), then gridCompleteAt once any handwriting pass is done too.
    let cropsShown = 0;
    let printedReadsShown = 0;
    this.pipelineClient.orientAndRectifyChecks(
      cornerSets,
      (checkMessage) => {
        if (!isCurrent()) return;
        this.orientedQuads[checkMessage.checkIndex] = checkMessage.orientedCorners;
        this.reviewGrid.setCrop(checkMessage.checkIndex, checkMessage);
        cropsShown += 1;
        if (cropsShown === cornerSets.length) this.timings.cropsCompleteAt = performance.now();
        this.progressPanel.setFraction(reviewProgressFraction({ cropsShown, printedReadsShown, checkCount }));
      },
      (text) => {
        if (!isCurrent()) return;
        // Printed pass: straightened + read out of 2 per check. Handwriting pass: its own check counter.
        const handwritingCheck = /handwriting/i.test(text) ? parseCheckOfTotal(text) : null;
        const fraction = handwritingCheck
          ? (handwritingCheck.current - 1) / handwritingCheck.total
          : reviewProgressFraction({ cropsShown, printedReadsShown, checkCount });
        this.progressPanel.showStage(describeWorkerStage(text), fraction);
      },
      ({ checkIndex, rawReads, pass }) => {
        if (!isCurrent()) return;
        this.reviewGrid.receiveStreamedFieldReads(checkIndex, rawReads);
        if (pass !== "printed") return;
        printedReadsShown += 1;
        if (printedReadsShown === cornerSets.length) this.timings.fieldReadsCompleteAt = performance.now();
        this.progressPanel.setFraction(reviewProgressFraction({ cropsShown, printedReadsShown, checkCount }));
      },
    ).then(() => {
      if (!isCurrent()) return;
      this.readsRunning = false;
      this.countStep.setBusy(false);
      this.progressPanel.showDone(`All ${pluralizeChecks(checkCount)} read`);
      this.updateReviewSummary();
      this.timings.gridCompleteAt = performance.now();
    }).catch((error) => {
      if (!isCurrent()) return;
      console.error("straightening the checks failed:", error);
      this.readsRunning = false;
      this.countStep.setBusy(false);
      this.showFailure(STRAIGHTENING_FAILED_MESSAGE);
    });
  }

  guardUnload(event) {
    if (this.step !== "review") return;
    const unfinishedCount = this.reviewGrid.countUnfinishedRows();
    if (unfinishedCount === 0) return;
    event.preventDefault();
    event.returnValue = UNSAVED_CHECKS_MESSAGE(unfinishedCount); // browsers show their own text
  }

  /** Drops the batch everywhere (Start over, or a new photo replacing this one). */
  clear() {
    this.batchToken += 1;
    this.countStep.clear();
    this.reviewGrid.clear();
    this.reviewSection.hidden = true;
    this.readsRunning = false;
    this.progressPanel.hide();
    this.progressPanel.returnHome();
    this.hints.withdraw(HINT_KEYS.FIX_OUTLINES);
    this.hints.withdraw(HINT_KEYS.REVIEW_GUIDE);
    if (this.pipelineClient) this.pipelineClient.releasePhoto();
    this.fullResCanvas = null;
    this.setStep(null);
    this.updateReviewSummary();
  }

  /**
   * Finish batch (spec 4.6): remember the payer names and each check's (number, payer,
   * date) with today as the batch date, release the images, keep a one-line record until
   * the next photo. Only confident or operator-confirmed values are remembered; an amber read
   * nobody reviewed never enters the autocomplete list or the duplicate history.
   */
  finish() {
    const checkCount = this.reviewGrid.getCheckCount();
    const rowValues = collectReviewedRowValues(this.reviewGrid);
    this.batchHistory.addKnownPayerNames(rowValues.map(({ payer }) => payer));
    this.batchHistory.recordConfirmedChecks(rowValues, todayIso());
    this.clear();
    this.recordedLine.textContent = `${pluralizeChecks(checkCount)} recorded`;
    this.recordedNotice.hidden = false;
  }

  /** For window.__checkTranscriberDebug (tests only; holds no pixels). */
  describeForDebug() {
    return {
      step: this.step,
      detectedQuads: this.detectedQuads,
      currentQuads: this.countStep.quads.length ? this.countStep.getCornerSets() : [],
      orientedQuads: this.orientedQuads,
      timings: { ...this.timings },
      reviewRowCount: this.reviewGrid.getCheckCount(),
      croppedRowCount: this.reviewGrid.rows.filter((row) => row.uprightCropCanvas).length,
      rows: this.reviewGrid.describeRowsForDebug(),
      emailDateIso: this.reviewGrid.emailDateIso,
    };
  }
}
