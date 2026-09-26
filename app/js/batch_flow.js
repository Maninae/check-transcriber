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
 */

import { todayIso } from "./fields/date_parsing.js";
import { collectReviewedRowValues } from "./review/review_field_editing.js";

const UNSAVED_CHECKS_MESSAGE = (count) =>
  `You have ${count} ${count === 1 ? "check" : "checks"} that have not been copied.`;

export class BatchFlow {
  /** `parts`: `{ countStep, reviewGrid, batchHistory, stepIndicator, progressLine, recordedLine, reviewSection, pageElement }`. */
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

  showProgress(text) {
    this.progressLine.hidden = !text;
    this.progressLine.textContent = text ? `${text}…` : "";
  }

  /** A photo was decoded; `photoArrivedAt` is performance.now() at paste/drop/pick. */
  startWithPhoto(fullResCanvas, photoArrivedAt) {
    this.fullResCanvas = fullResCanvas;
    this.recordedLine.hidden = true;
    this.timings = { photoArrivedAt };
    this.detectedQuads = null;
    this.setStep("photo");
    const run = () => this.detect(fullResCanvas);
    if (this.pipelineClient) {
      run();
    } else {
      this.showProgress("Getting ready");
      this.enginesWaiters.push(run);
    }
  }

  detect(fullResCanvas) {
    this.showProgress("Finding checks");
    this.pipelineClient.loadPhoto(fullResCanvas)
      .then(() => this.pipelineClient.detectChecks((text) => this.showProgress(text)))
      .then((detections) => {
        if (fullResCanvas !== this.fullResCanvas) return; // replaced while detecting
        this.detectedQuads = detections.map(({ corners }) => corners);
        this.showProgress(null);
        this.setStep("count", detections.length);
        this.countStep.show(fullResCanvas, detections);
        this.timings.countStepShownAt = performance.now();
      })
      .catch((error) => {
        console.error("check detection failed:", error);
        this.showProgress("Something went wrong finding the checks. Start over to try again");
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
    this.reviewGrid.start(cornerSets.length);
    this.reviewSection.scrollIntoView({ behavior: "smooth", block: "start" });
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
      },
      (text) => { if (isCurrent()) this.showProgress(text); },
      ({ checkIndex, rawReads, pass }) => {
        if (!isCurrent()) return;
        this.reviewGrid.receiveStreamedFieldReads(checkIndex, rawReads);
        if (pass !== "printed") return;
        printedReadsShown += 1;
        if (printedReadsShown === cornerSets.length) this.timings.fieldReadsCompleteAt = performance.now();
      },
    ).then(() => {
      if (!isCurrent()) return;
      this.showProgress(null);
      this.timings.gridCompleteAt = performance.now();
    }).catch((error) => {
      if (!isCurrent()) return;
      console.error("straightening the checks failed:", error);
      this.showProgress("Something went wrong straightening the checks. Start over to try again");
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
    this.showProgress(null);
    if (this.pipelineClient) this.pipelineClient.releasePhoto();
    this.fullResCanvas = null;
    this.setStep(null);
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
    this.recordedLine.textContent = `${checkCount} ${checkCount === 1 ? "check" : "checks"} recorded`;
    this.recordedLine.hidden = false;
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
