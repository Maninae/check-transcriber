/**
 * Wires the page together: grabs the DOM elements, starts engine loading, hooks up
 * the three input doors, and handles the decode-and-preview flow. This is the only
 * module that touches multiple other modules — everything else is a leaf that only
 * knows its own piece (input doors don't know about HEIC, HEIC detection doesn't
 * know about canvases, and so on).
 */

import { initializeInputDoors } from "./input_doors.js";
import { isLikelyHeicImage } from "./heic_detect.js";
import { decodeAndOrientImage } from "./image_decode.js";
import { loadProcessingEngines, READINESS_MESSAGE, FAILURE_MESSAGE } from "./engine_loader.js";
import { EngineStatusLine } from "./status_line.js";

const ACCEPTED_IMAGE_MIME_TYPES = ["image/jpeg", "image/png", "image/webp"];

const HEIC_HINT_MESSAGE =
  "This is an iPhone HEIC file. Open the attachment in Gmail and use Copy image instead.";
const UNSUPPORTED_TYPE_HINT_MESSAGE =
  "That doesn't look like a photo this tool can read. Use a JPEG, PNG, or WebP image.";
const URL_ONLY_DROP_HINT_MESSAGE =
  "That carried a link to the image, not the image itself. In Gmail, right-click the " +
  "photo and choose Copy image, then paste here (Ctrl+V) instead.";
const START_OVER_CONFIRM_MESSAGE = "Start over with a new photo? The current batch will be cleared.";

// Holds the two canvases produced by image_decode.js for the currently loaded photo.
// Milestone 2 hook: the detection stage (contour finding on `workingCanvas`) attaches
// here, reading `currentPhoto.workingCanvas` once the count step is built.
const currentPhoto = { fullResCanvas: null, workingCanvas: null };

function queryRequiredElement(id) {
  const element = document.getElementById(id);
  if (!element) {
    throw new Error(`expected element #${id} to exist in index.html`);
  }
  return element;
}

function initializeDom() {
  return {
    dropZoneElement: queryRequiredElement("drop-zone"),
    dropZonePromptElement: queryRequiredElement("drop-zone-prompt"),
    fileInputElement: queryRequiredElement("file-input"),
    previewContainerElement: queryRequiredElement("photo-preview-container"),
    previewCanvasElement: queryRequiredElement("photo-preview-canvas"),
    photoReceivedLineElement: queryRequiredElement("photo-received-line"),
    startOverButtonElement: queryRequiredElement("start-over-button"),
    hintLineElement: queryRequiredElement("input-hint-line"),
    engineStatusLine: new EngineStatusLine({
      containerElement: queryRequiredElement("engine-status"),
      textElement: queryRequiredElement("engine-status-text"),
      retryButtonElement: queryRequiredElement("engine-status-retry"),
    }),
  };
}

function showHint(dom, message) {
  dom.hintLineElement.textContent = message;
  dom.hintLineElement.hidden = false;
}

function clearHint(dom) {
  dom.hintLineElement.hidden = true;
  dom.hintLineElement.textContent = "";
}

function hasPhotoLoaded() {
  return currentPhoto.fullResCanvas !== null;
}

/** Draws the decoded photo, shows its dimensions, and reveals the Start Over control. */
function showDecodedPhoto(dom, { fullResCanvas, workingCanvas, width, height }) {
  currentPhoto.fullResCanvas = fullResCanvas;
  currentPhoto.workingCanvas = workingCanvas;

  const displayContext = dom.previewCanvasElement.getContext("2d");
  dom.previewCanvasElement.width = width;
  dom.previewCanvasElement.height = height;
  displayContext.drawImage(fullResCanvas, 0, 0);

  dom.photoReceivedLineElement.textContent = `Photo received: ${width} x ${height}`;
  dom.dropZonePromptElement.hidden = true;
  dom.previewContainerElement.hidden = false;
}

function resetToEmptyDropZone(dom) {
  currentPhoto.fullResCanvas = null;
  currentPhoto.workingCanvas = null;
  dom.previewContainerElement.hidden = true;
  dom.dropZonePromptElement.hidden = false;
  clearHint(dom);
}

/**
 * Handles a File from any of the three input doors: rejects HEIC and unsupported
 * types with an explanatory hint, otherwise decodes, orients, and previews it.
 * `file` is deliberately not retained past this function — see image_decode.js's
 * module docstring for why that matters.
 */
async function handleImageFile(dom, file) {
  if (hasPhotoLoaded() && !window.confirm(START_OVER_CONFIRM_MESSAGE)) {
    return;
  }
  clearHint(dom);

  if (await isLikelyHeicImage(file)) {
    showHint(dom, HEIC_HINT_MESSAGE);
    return;
  }
  if (!ACCEPTED_IMAGE_MIME_TYPES.includes(file.type)) {
    showHint(dom, UNSUPPORTED_TYPE_HINT_MESSAGE);
    return;
  }

  const decoded = await decodeAndOrientImage(file);
  showDecodedPhoto(dom, decoded);
}

function handleNoUsableImage(dom, reason) {
  if (reason === "url-reference-only") {
    showHint(dom, URL_ONLY_DROP_HINT_MESSAGE);
  } else {
    showHint(dom, UNSUPPORTED_TYPE_HINT_MESSAGE);
  }
}

function initializeStartOverButton(dom) {
  dom.startOverButtonElement.addEventListener("click", () => {
    if (window.confirm(START_OVER_CONFIRM_MESSAGE)) {
      resetToEmptyDropZone(dom);
    }
  });
}

/**
 * Loads OpenCV and Tesseract, reporting state through `dom.engineStatusLine`. Retry
 * re-runs this.
 *
 * Deliberately callback-style, not `await`ed: see the "CALLBACK STYLE IS
 * LOAD-BEARING" note at the top of engine_loader.js. `await loadProcessingEngines()`
 * looks like the obvious way to write this and will silently hang the entire page.
 */
function startEngineLoading(dom) {
  dom.engineStatusLine.showLoading(READINESS_MESSAGE);
  loadProcessingEngines(
    (engines) => {
      console.log("OpenCV build info:\n" + engines.openCvBuildInfo);
      dom.engineStatusLine.showReady();
    },
    (error) => {
      console.error("engine loading failed:", error);
      dom.engineStatusLine.showError(FAILURE_MESSAGE, () => startEngineLoading(dom));
    },
  );
}

function registerServiceWorker() {
  if (!("serviceWorker" in navigator)) {
    return;
  }
  // Relative path/scope so this resolves correctly both at http://localhost/ and at
  // the GitHub Pages subpath https://maninae.github.io/check-transcriber/.
  navigator.serviceWorker.register("./sw.js").catch((error) => {
    console.warn("service worker registration failed:", error);
  });
}

function main() {
  const dom = initializeDom();

  initializeInputDoors({
    dropZoneElement: dom.dropZoneElement,
    dropZonePromptElement: dom.dropZonePromptElement,
    fileInputElement: dom.fileInputElement,
    onImageFile: (file) => handleImageFile(dom, file),
    onNoUsableImage: (reason) => handleNoUsableImage(dom, reason),
  });
  initializeStartOverButton(dom);
  registerServiceWorker();
  startEngineLoading(dom);
}

document.addEventListener("DOMContentLoaded", main);
