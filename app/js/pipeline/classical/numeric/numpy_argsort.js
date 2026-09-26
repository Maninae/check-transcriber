/**
 * numpy's default `np.argsort` (kind="quicksort": introsort, insertion sort below 16 elements),
 * plus first-occurrence `argmax` / `argmin`.
 *
 * numpy's quicksort is NOT stable, and which of two equal keys comes first decides segment
 * merge order and pair budgets downstream, so a stable JS sort would diverge from the Python
 * on ties. Verified identical to numpy 2.x on tie-heavy arrays of 3 to 5000 elements.
 */

const SMALL_QUICKSORT_PARTITION = 15; // numpy npysort/quicksort.cpp SMALL_QUICKSORT

/** Index of the most significant set bit (numpy `npy_get_msb`). */
function mostSignificantBit(count) {
  let bitIndex = 0;
  while (count > 1) {
    count >>= 1;
    bitIndex += 1;
  }
  return bitIndex;
}

/** numpy `aheapsort_` on `order[low..high]` (the introsort depth-limit fallback). */
function heapsortIndices(values, order, low, count) {
  // numpy indexes the heap from 1: a[k] is order[low + k - 1]
  const heapGet = (heapIndex) => order[low + heapIndex - 1];
  const heapSet = (heapIndex, value) => {
    order[low + heapIndex - 1] = value;
  };
  let heapSize = count;
  for (let root = heapSize >> 1; root > 0; root -= 1) {
    const saved = heapGet(root);
    let parent = root;
    let child = root << 1;
    while (child <= heapSize) {
      if (child < heapSize && values[heapGet(child)] < values[heapGet(child + 1)]) child += 1;
      if (!(values[saved] < values[heapGet(child)])) break;
      heapSet(parent, heapGet(child));
      parent = child;
      child += child;
    }
    heapSet(parent, saved);
  }
  while (heapSize > 1) {
    const saved = heapGet(heapSize);
    heapSet(heapSize, heapGet(1));
    heapSize -= 1;
    let parent = 1;
    let child = 2;
    while (child <= heapSize) {
      if (child < heapSize && values[heapGet(child)] < values[heapGet(child + 1)]) child += 1;
      if (!(values[saved] < values[heapGet(child)])) break;
      heapSet(parent, heapGet(child));
      parent = child;
      child += child;
    }
    heapSet(parent, saved);
  }
}

/**
 * `np.argsort(values)` with numpy's default introsort, tie order included.
 *
 * Args:
 *   values: array-like of numbers (no NaNs in this pipeline).
 * Returns:
 *   Int32Array of indices that sort `values` ascending, identical to numpy's.
 */
export function numpyArgsort(values) {
  const count = values.length;
  const order = new Int32Array(count);
  for (let index = 0; index < count; index += 1) order[index] = index;
  if (count < 2) return order;
  const swap = (first, second) => {
    const saved = order[first];
    order[first] = order[second];
    order[second] = saved;
  };
  const partitionStack = [];
  const depthStack = [];
  let low = 0;
  let high = count - 1;
  let depthBudget = mostSignificantBit(count) * 2;
  for (;;) {
    if (depthBudget < 0) {
      heapsortIndices(values, order, low, high - low + 1);
    } else {
      while (high - low > SMALL_QUICKSORT_PARTITION) {
        const middle = low + ((high - low) >> 1);
        if (values[order[middle]] < values[order[low]]) swap(middle, low);
        if (values[order[high]] < values[order[middle]]) swap(high, middle);
        if (values[order[middle]] < values[order[low]]) swap(middle, low);
        const pivot = values[order[middle]];
        let left = low;
        let right = high - 1;
        swap(middle, right);
        for (;;) {
          do left += 1; while (values[order[left]] < pivot);
          do right -= 1; while (pivot < values[order[right]]);
          if (left >= right) break;
          swap(left, right);
        }
        swap(left, high - 1);
        if (left - low < high - left) {
          partitionStack.push(left + 1, high);
          high = left - 1;
        } else {
          partitionStack.push(low, left - 1);
          low = left + 1;
        }
        depthBudget -= 1;
        depthStack.push(depthBudget);
      }
      for (let outer = low + 1; outer <= high; outer += 1) {
        const movingIndex = order[outer];
        const movingValue = values[movingIndex];
        let inner = outer;
        while (inner > low && movingValue < values[order[inner - 1]]) {
          order[inner] = order[inner - 1];
          inner -= 1;
        }
        order[inner] = movingIndex;
      }
    }
    if (partitionStack.length === 0) break;
    high = partitionStack.pop();
    low = partitionStack.pop();
    depthBudget = depthStack.pop();
  }
  return order;
}

/** Index of the first maximum (`np.argmax`). */
export function argmaxFirst(values) {
  let bestIndex = 0;
  for (let index = 1; index < values.length; index += 1) {
    if (values[index] > values[bestIndex]) bestIndex = index;
  }
  return bestIndex;
}

/** Index of the first minimum (`np.argmin`). */
export function argminFirst(values) {
  let bestIndex = 0;
  for (let index = 1; index < values.length; index += 1) {
    if (values[index] < values[bestIndex]) bestIndex = index;
  }
  return bestIndex;
}
