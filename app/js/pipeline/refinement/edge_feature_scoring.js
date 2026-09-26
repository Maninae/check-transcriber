/**
 * Per-pixel edge features on averaged normal profiles, and the boundary score built from them.
 *
 * Port of the feature half of experiments/detection/refinement/edge_profile_sampling.py.
 * Profiles are a Float32Array laid out (sample, offset, channel), numpy float32 semantics:
 * every float32 operation is emulated as Math.fround(a op b) (exact for + - * /, since a
 * double has more than 2 * 24 + 2 bits), and small float32 reductions add left to right,
 * which is what numpy does for these shapes (measured).
 *
 * - `computeTwoClassFeature`: minus the per-sample paper(1)-vs-background(0) projection,
 *   tent past the paper colour, NaN rows where paper and background do not contrast.
 * - `computePaperDistanceFeature`: colour distance to the paper colour.
 * - `computeBoxWindowScores`: mean feature just outside minus just inside each centre.
 * - `computeContactLineScores`: luminance valley depth (white-on-white contact shadow).
 */

const fround = Math.fround;
const MINIMUM_CONTRAST_SQUARED_DENOMINATOR = 1e-9;

/** Float32 mean colour of the outermost `windowPixels` offsets of each sample, (S, C) float32 values. */
export function computeBackgroundColours(profiles, numberOfSamples, numberOfOffsets, numberOfChannels, windowPixels) {
  const backgroundColours = new Float64Array(numberOfSamples * numberOfChannels);
  for (let sample = 0; sample < numberOfSamples; sample += 1) {
    for (let channel = 0; channel < numberOfChannels; channel += 1) {
      let sum = 0;
      for (let offset = numberOfOffsets - windowPixels; offset < numberOfOffsets; offset += 1) {
        sum = fround(sum + profiles[(sample * numberOfOffsets + offset) * numberOfChannels + channel]);
      }
      backgroundColours[sample * numberOfChannels + channel] = fround(sum / windowPixels);
    }
  }
  return backgroundColours;
}

/** -paperness per (sample, offset); see compute_two_class_paperness in the Python for the model. */
export function computeTwoClassFeature(profiles, shape, paperColour, backgroundWindowPixels, minimumContrast, tentBeyondPaper) {
  const { numberOfSamples, numberOfOffsets, numberOfChannels } = shape;
  const backgroundColours = computeBackgroundColours(profiles, numberOfSamples, numberOfOffsets, numberOfChannels, backgroundWindowPixels);
  const feature = new Float64Array(numberOfSamples * numberOfOffsets);
  const paperMinusBackground = new Float64Array(numberOfChannels);
  for (let sample = 0; sample < numberOfSamples; sample += 1) {
    let contrastSquared = 0;
    for (let channel = 0; channel < numberOfChannels; channel += 1) {
      paperMinusBackground[channel] = paperColour[channel] - backgroundColours[sample * numberOfChannels + channel];
      contrastSquared += paperMinusBackground[channel] * paperMinusBackground[channel];
    }
    const rowOffset = sample * numberOfOffsets;
    if (contrastSquared < minimumContrast ** 2) {
      feature.fill(NaN, rowOffset, rowOffset + numberOfOffsets);
      continue;
    }
    const denominator = Math.max(contrastSquared, MINIMUM_CONTRAST_SQUARED_DENOMINATOR);
    for (let offset = 0; offset < numberOfOffsets; offset += 1) {
      const pixelBase = (rowOffset + offset) * numberOfChannels;
      let projection = 0;
      for (let channel = 0; channel < numberOfChannels; channel += 1) {
        const relative = fround(profiles[pixelBase + channel] - backgroundColours[sample * numberOfChannels + channel]);
        projection += relative * paperMinusBackground[channel];
      }
      let paperness = projection / denominator;
      // Past the paper colour (away from the background) is a shadow or ink, not more paper.
      if (tentBeyondPaper && paperness > 1.0) paperness = 2.0 - paperness;
      feature[rowOffset + offset] = -Math.min(Math.max(paperness, 0.0), 1.0);
    }
  }
  return feature;
}

/** Colour distance of each profile pixel to the paper colour (float64). */
export function computePaperDistanceFeature(profiles, shape, paperColour) {
  const { numberOfSamples, numberOfOffsets, numberOfChannels } = shape;
  const feature = new Float64Array(numberOfSamples * numberOfOffsets);
  for (let cell = 0; cell < numberOfSamples * numberOfOffsets; cell += 1) {
    let squaredDistance = 0;
    for (let channel = 0; channel < numberOfChannels; channel += 1) {
      const difference = profiles[cell * numberOfChannels + channel] - paperColour[channel];
      squaredDistance += difference * difference;
    }
    feature[cell] = Math.sqrt(squaredDistance);
  }
  return feature;
}

/**
 * Outer-window mean minus inner-window mean of `feature` at every centre offset.
 *
 * Centres run over [innerWindow, numberOfOffsets - outerWindow); returns (S, numberOfCentres).
 * Windows are [c - inner, c) and (c, c + outer], via a per-row prefix sum (NaN rows stay NaN).
 */
export function computeBoxWindowScores(feature, numberOfSamples, numberOfOffsets, innerWindow, outerWindow) {
  const numberOfCentres = numberOfOffsets - outerWindow - innerWindow;
  const scores = new Float64Array(numberOfSamples * numberOfCentres);
  const cumulative = new Float64Array(numberOfOffsets + 1);
  for (let sample = 0; sample < numberOfSamples; sample += 1) {
    for (let offset = 0; offset < numberOfOffsets; offset += 1) {
      cumulative[offset + 1] = cumulative[offset] + feature[sample * numberOfOffsets + offset];
    }
    for (let centreIndex = 0; centreIndex < numberOfCentres; centreIndex += 1) {
      const centre = centreIndex + innerWindow;
      const innerMean = (cumulative[centre] - cumulative[centre - innerWindow]) / innerWindow;
      const outerMean = (cumulative[centre + outerWindow + 1] - cumulative[centre + 1]) / outerWindow;
      scores[sample * numberOfCentres + centreIndex] = outerMean - innerMean;
    }
  }
  return scores;
}

/**
 * Contact-shadow valley depth for one sample row, written into `scores` row `sample`.
 *
 * depth = mean luminance `halfWidth` either side minus the centre, / `contrastUnit`, clipped
 * to [0, 1]; all float32 like the numpy original (luminance is a float32 channel mean).
 */
export function writeContactLineScoresForSample(profiles, shape, sample, firstCentre, numberOfCentres, halfWidth, contrastUnit, scores) {
  const { numberOfOffsets, numberOfChannels } = shape;
  const luminance = new Float64Array(numberOfOffsets);
  for (let offset = 0; offset < numberOfOffsets; offset += 1) {
    const pixelBase = (sample * numberOfOffsets + offset) * numberOfChannels;
    let sum = 0;
    for (let channel = 0; channel < numberOfChannels; channel += 1) sum = fround(sum + profiles[pixelBase + channel]);
    luminance[offset] = fround(sum / numberOfChannels);
  }
  const lastIndex = numberOfOffsets - 1;
  for (let centreIndex = 0; centreIndex < numberOfCentres; centreIndex += 1) {
    const centre = firstCentre + centreIndex;
    const left = luminance[Math.min(Math.max(centre - halfWidth, 0), lastIndex)];
    const right = luminance[Math.min(Math.max(centre + halfWidth, 0), lastIndex)];
    const depth = fround(fround(fround(left + right) / 2) - luminance[centre]);
    scores[sample * numberOfCentres + centreIndex] = Math.min(Math.max(fround(depth / contrastUnit), 0.0), 1.0);
  }
}
