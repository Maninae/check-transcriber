/**
 * One batch, start to finish (spec 4): a decoded photo goes to the pipeline worker for
 * detection, the count step confirms the quads, the worker orients and crops each check
 * and the review grid fills row by row, and Finish batch clears it all.
 *
 * Owns the batch-level state (which step, timings) and the "leave page?" guard. The
 * photo's pixels live in three places only while a batch is open: the decoded canvas
 * (main.js), the count step's display canvas, and the worker's copy. Finish batch and
 * Start over release all three.
 */

const UNSAVED_CHECKS_MESSAGE = (count) =>
  `You have ${count} ${count === 1 ? "check" : "checks"} that have not been copied.`;

export class BatchFlow {
  /** `parts`: `{ countStep, reviewGrid, stepIndicator, progressLine, recordedLine, reviewSection, pageElement }`. */
  constructor(parts) {
    Object.assign(this, parts);
    this.pipelineClient = null;
    this.enginesWaiters = [];
    this.fullResCanvas = null;
    this.step = null;
    this.timings = {};
    this.detectedQuads = null;
    this.orientedQuads = [];
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
    this.reviewGrid.start(cornerSets.length);
    this.reviewSection.scrollIntoView({ behavior: "smooth", block: "start" });
    this.pipelineClient.orientAndRectifyChecks(
      cornerSets,
      (checkMessage) => {
        this.orientedQuads[checkMessage.checkIndex] = checkMessage.orientedCorners;
        this.reviewGrid.setCrop(checkMessage.checkIndex, checkMessage);
      },
      (text) => this.showProgress(text),
    ).then(() => {
      this.showProgress(null);
      this.timings.gridCompleteAt = performance.now();
    }).catch((error) => {
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
    this.countStep.clear();
    this.reviewGrid.clear();
    this.reviewSection.hidden = true;
    this.showProgress(null);
    if (this.pipelineClient) this.pipelineClient.releasePhoto();
    this.fullResCanvas = null;
    this.setStep(null);
  }

  /** Finish batch: release the images, keep a one-line record until the next photo. */
  finish() {
    const checkCount = this.reviewGrid.getCheckCount();
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
    };
  }
}
