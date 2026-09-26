/**
 * The step line in the header (spec 4: "the top of the page always shows which step the
 * operator is on and the check count once known"). Hidden on the empty drop zone. Each
 * step wears a small numbered disc (CSS); finished steps show a tick instead.
 */

const STEP_LABELS = { photo: "Photo", count: "Count", review: "Review" };

export class StepIndicator {
  constructor(listElement) {
    this.listElement = listElement;
    this.stepElements = [...listElement.querySelectorAll("[data-step]")];
  }

  /** `step` is "photo" | "count" | "review" | null (null hides the line). */
  show(step, checkCount = null) {
    this.listElement.hidden = step === null;
    const currentPosition = this.stepElements.findIndex((element) => element.dataset.step === step);
    this.stepElements.forEach((element, position) => {
      element.classList.toggle("step-indicator-step--done", currentPosition > position);
    });
    for (const element of this.stepElements) {
      const isCurrent = element.dataset.step === step;
      element.toggleAttribute("aria-current", isCurrent);
      if (isCurrent) element.setAttribute("aria-current", "step");
      const countSuffix = element.dataset.step === "count" && checkCount !== null ? ` (${checkCount})` : "";
      element.textContent = STEP_LABELS[element.dataset.step] + countSuffix;
    }
  }
}
