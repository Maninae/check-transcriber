/**
 * Draws the count step's SVG overlay from state: one outline per check with its reading-
 * order number, corner handles, the remove (X) button on the selected check, and the
 * rectangle being dragged in "Add a check" mode.
 *
 * The SVG's viewBox is the photo's full-resolution pixel grid, so quads are drawn in
 * their own coordinates; only sizes meant in screen pixels (handle radius, badge, font)
 * are divided by `displayScale` (screen px per photo px). Strokes are non-scaling (CSS).
 * Every interactive element carries data attributes that count_step.js dispatches on.
 */


const SVG_NAMESPACE = "http://www.w3.org/2000/svg";
const HANDLE_RADIUS_SCREEN_PX = 6;
const SELECTED_HANDLE_RADIUS_SCREEN_PX = 8;
const BADGE_RADIUS_SCREEN_PX = 16;
const BADGE_FONT_SCREEN_PX = 17;
const REMOVE_BUTTON_RADIUS_SCREEN_PX = 12;
const BADGE_ALONG_EDGE_RADII = 2.2; // badge centre's distance from the top-left corner, in badge radii

function createSvgElement(tagName, attributes) {
  const element = document.createElementNS(SVG_NAMESPACE, tagName);
  for (const [name, value] of Object.entries(attributes)) element.setAttribute(name, String(value));
  return element;
}

function pointsAttribute(corners) {
  return corners.map(([x, y]) => `${x},${y}`).join(" ");
}

// The corner nearest the photo's top-left, and the one nearest its top-right.
function findTopLeftCorner(corners) {
  return corners.reduce((best, corner) => (corner[0] + corner[1] < best[0] + best[1] ? corner : best));
}

function findTopRightCorner(corners) {
  return corners.reduce((best, corner) => (corner[0] - corner[1] > best[0] - best[1] ? corner : best));
}

function drawNumberBadge(svgElement, quad, number, pixelsPerScreenPx, imageWidth, imageHeight) {
  const [leftX, leftY] = findTopLeftCorner(quad.corners);
  const [rightX, rightY] = findTopRightCorner(quad.corners);
  const radius = BADGE_RADIUS_SCREEN_PX * pixelsPerScreenPx;
  // On the top edge, a little in from the top-left corner: over the check's blank top
  // margin (not the payer name) and clear of the corner handle.
  const edgeLength = Math.hypot(rightX - leftX, rightY - leftY) || 1;
  const along = Math.min(radius * BADGE_ALONG_EDGE_RADII, edgeLength / 2);
  const edgeX = leftX + ((rightX - leftX) / edgeLength) * along;
  const edgeY = leftY + ((rightY - leftY) / edgeLength) * along;
  // Kept fully inside the photo, so a check cut off by the frame edge still shows its number.
  const badgeX = Math.min(Math.max(edgeX, radius), imageWidth - radius);
  const badgeY = Math.min(Math.max(edgeY, radius), imageHeight - radius);
  svgElement.append(
    createSvgElement("circle", { class: "count-number-badge", cx: badgeX, cy: badgeY, r: radius, "pointer-events": "none" }),
  );
  const label = createSvgElement("text", { class: "count-number-text", x: badgeX, y: badgeY, "font-size": BADGE_FONT_SCREEN_PX * pixelsPerScreenPx });
  label.textContent = String(number);
  svgElement.append(label);
}

function drawRemoveButton(svgElement, quad, pixelsPerScreenPx) {
  const [x, y] = findTopRightCorner(quad.corners);
  const radius = REMOVE_BUTTON_RADIUS_SCREEN_PX * pixelsPerScreenPx;
  const arm = radius * 0.4;
  const group = createSvgElement("g", { class: "count-remove-button", "data-remove-quad-id": quad.id, role: "button", "aria-label": "Remove this check" });
  group.append(
    createSvgElement("circle", { cx: x, cy: y, r: radius }),
    createSvgElement("path", { d: `M ${x - arm} ${y - arm} L ${x + arm} ${y + arm} M ${x + arm} ${y - arm} L ${x - arm} ${y + arm}` }),
  );
  svgElement.append(group);
}

/**
 * Re-renders the overlay. `state`: `{ imageWidth, imageHeight, displayScale, quads,
 * selectedQuadId, drawRectangle }`; each quad is `{ id, corners, confident, pending }`
 * and quads are already in reading order (number = position + 1).
 */
export function renderCountOverlay(svgElement, { imageWidth, imageHeight, displayScale, quads, selectedQuadId, drawRectangle }) {
  svgElement.setAttribute("viewBox", `0 0 ${imageWidth} ${imageHeight}`);
  svgElement.replaceChildren();
  const pixelsPerScreenPx = 1 / displayScale;

  // One group per quad (outline + its corner handles) so CSS can reveal the handles on
  // hover; transparent handles still take the pointer, so dragging a corner is one step.
  quads.forEach((quad) => {
    const isSelected = quad.id === selectedQuadId;
    const group = createSvgElement("g", { class: isSelected ? "count-quad count-quad--selected" : "count-quad" });
    const classNames = ["count-outline"];
    if (!quad.confident) classNames.push("count-outline--unsure");
    if (isSelected) classNames.push("count-outline--selected");
    if (quad.pending) classNames.push("count-outline--pending");
    group.append(createSvgElement("polygon", { class: classNames.join(" "), points: pointsAttribute(quad.corners), "data-quad-id": quad.id }));
    if (!quad.pending) {
      const radius = (isSelected ? SELECTED_HANDLE_RADIUS_SCREEN_PX : HANDLE_RADIUS_SCREEN_PX) * pixelsPerScreenPx;
      quad.corners.forEach(([x, y], cornerIndex) => {
        group.append(createSvgElement("circle", { class: "count-corner-handle", cx: x, cy: y, r: radius, "data-handle-quad-id": quad.id, "data-corner-index": cornerIndex }));
      });
    }
    svgElement.append(group);
  });
  // Numbers last, so no handle or neighbouring outline covers them.
  quads.forEach((quad, index) => drawNumberBadge(svgElement, quad, index + 1, pixelsPerScreenPx, imageWidth, imageHeight));
  const selectedQuad = quads.find((quad) => quad.id === selectedQuadId);
  if (selectedQuad) drawRemoveButton(svgElement, selectedQuad, pixelsPerScreenPx);
  if (drawRectangle) {
    const { startX, startY, endX, endY } = drawRectangle;
    svgElement.append(createSvgElement("rect", {
      class: "count-draw-rectangle",
      x: Math.min(startX, endX), y: Math.min(startY, endY), width: Math.abs(endX - startX), height: Math.abs(endY - startY),
    }));
  }
}
