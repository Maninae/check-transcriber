/**
 * The step line in the header (spec 4: "the top of the page always shows which step the
 * operator is on and the check count once known"). Hidden on the empty drop zone.
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
    for (const element of this.stepElements) {
      const isCurrent = element.dataset.step === step;
      element.toggleAttribute("aria-current", isCurrent);
      if (isCurrent) element.setAttribute("aria-current", "step");
      const countSuffix = element.dataset.step === "count" && checkCount !== null ? ` (${checkCount})` : "";
      element.textContent = STEP_LABELS[element.dataset.step] + countSuffix;
    }
  }
}
