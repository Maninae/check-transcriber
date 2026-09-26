/**
 * A side's edge curve in the side's own frame: point = start + s * tangent + poly(x) * normal.
 *
 * x = (s - mid) / half is the position normalised to [-1, 1] over the side (conditioning).
 * Degree 1 is a straight side; degree 2 captures a curl's bow. Port of the `SideCurve`
 * dataclass in experiments/detection/refinement/robust_side_curve_fitting.py.
 */

const MINIMUM_HALF_LENGTH = 1e-9;

/** Edge curve of one side plus the evidence summary of the fit that produced it. */
export class SideCurve {
  /** `coefficients` = [constant, linear, (quadratic)] in normalised position. */
  constructor({ sideStart, unitTangent, unitNormal, sideLength, coefficients, inlierCount = 0, medianInlierStrength = 0.0, inlierResidualRms = 0.0 }) {
    this.sideStart = sideStart;
    this.unitTangent = unitTangent;
    this.unitNormal = unitNormal;
    this.sideLength = sideLength;
    this.coefficients = coefficients;
    this.inlierCount = inlierCount;
    this.medianInlierStrength = medianInlierStrength;
    this.inlierResidualRms = inlierResidualRms;
  }

  /** Same frame, new coefficients and fit statistics. */
  withFit(fit) {
    return new SideCurve({ ...this, ...fit });
  }

  /** Map a position (px along the side) to [-1, 1]. */
  normalised(position) {
    const halfLength = Math.max(this.sideLength / 2, MINIMUM_HALF_LENGTH);
    return (position - halfLength) / halfLength;
  }

  /** Offset (px) of the curve at one position. */
  offsetAt(position) {
    return evaluatePolynomial(this.coefficients, this.normalised(position));
  }

  /** Float64Array of offsets at many positions. */
  offsetsAt(positions) {
    return Float64Array.from(positions, (position) => this.offsetAt(position));
  }

  /** [offset px, d offset / d position] at one position. */
  offsetAndSlope(position) {
    const normalisedPosition = this.normalised(position);
    const offset = evaluatePolynomial(this.coefficients, normalisedPosition);
    let derivative = 0;
    let power = 1;
    for (let degree = 1; degree < this.coefficients.length; degree += 1) {
      derivative += this.coefficients[degree] * (degree * power);
      power *= normalisedPosition;
    }
    return [offset, derivative / Math.max(this.sideLength / 2, MINIMUM_HALF_LENGTH)];
  }

  /** { point: image [x, y] on the curve, derivative: d point / d position }. */
  pointAndDerivative(position) {
    const [offset, slope] = this.offsetAndSlope(position);
    return {
      point: [
        this.sideStart[0] + position * this.unitTangent[0] + offset * this.unitNormal[0],
        this.sideStart[1] + position * this.unitTangent[1] + offset * this.unitNormal[1],
      ],
      derivative: [this.unitTangent[0] + slope * this.unitNormal[0], this.unitTangent[1] + slope * this.unitNormal[1]],
    };
  }
}

/** sum_k c_k * x^k, accumulated in increasing degree like numpy's matmul with the power matrix. */
export function evaluatePolynomial(coefficients, normalisedPosition) {
  let value = 0;
  let power = 1;
  for (let degree = 0; degree < coefficients.length; degree += 1) {
    value += power * coefficients[degree];
    power *= normalisedPosition;
  }
  return value;
}
