/**
 * Greedy CTC decoding and its confidence (a port of experiments/field_reading/learned/
 * ctc_decoding.py, `mean_char` kind, the one the gate thresholds were tuned on): argmax per
 * timestep, collapse repeats, drop blanks (label 0); confidence = mean over emitted
 * characters of the probability at the step that emitted them. Empty decode -> 0.
 */

const CTC_BLANK_INDEX = 0;

/**
 * Decodes one line. `logProbabilities` is a Float32Array laid out (T, C) (batch of one),
 * `charset` the recognizer's characters (label i + 1 = charset[i]).
 * Returns `{ text, confidence }`.
 */
export function decodeGreedyCtc(logProbabilities, timestepCount, classCount, charset) {
  let previousLabel = CTC_BLANK_INDEX;
  let text = "";
  let probabilitySum = 0;
  let emittedCount = 0;
  for (let step = 0; step < timestepCount; step += 1) {
    const rowOffset = step * classCount;
    let bestLabel = 0;
    let bestLogProbability = logProbabilities[rowOffset];
    for (let label = 1; label < classCount; label += 1) {
      // Strict > keeps the first maximum, like numpy's argmax.
      if (logProbabilities[rowOffset + label] > bestLogProbability) {
        bestLogProbability = logProbabilities[rowOffset + label];
        bestLabel = label;
      }
    }
    if (bestLabel !== CTC_BLANK_INDEX && bestLabel !== previousLabel) {
      text += charset[bestLabel - 1];
      probabilitySum += Math.exp(bestLogProbability);
      emittedCount += 1;
    }
    previousLabel = bestLabel;
  }
  return { text, confidence: emittedCount ? probabilitySum / emittedCount : 0 };
}
