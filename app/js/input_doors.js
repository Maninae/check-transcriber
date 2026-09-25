/**
 * The three ways a photo gets into the app: paste, drag-and-drop, and click-to-browse.
 * This module's only job is extracting a `File` from whichever door fired and handing
 * it to the caller — it does not validate file type or decode anything, so it stays
 * simple and testable independent of the decode pipeline.
 *
 * One door needs special handling: dragging an image from another browser tab (e.g.
 * Gmail's attachment preview) does not carry a `File`. The browser only gives the
 * source page's DataTransfer a URL/HTML reference to the image, not its pixel bytes,
 * so there is nothing to decode — and even if there were a URL, this app's Content
 * Security Policy only allows network access to its one CDN, so fetching an arbitrary
 * Gmail-hosted image URL would be blocked anyway. The honest behavior is the hint the
 * spec asks for, not a fetch that can never succeed.
 */

const IMAGE_MIME_PREFIX = "image/";

function isImageFile(file) {
  return file.type.startsWith(IMAGE_MIME_PREFIX);
}

/**
 * Wires up paste-anywhere, drag-and-drop onto `dropZoneElement`, and click-to-browse
 * on `dropZonePromptElement`. All three call `onImageFile(file)` on success.
 * `onNoUsableImage(reason)` fires when a drop carried a URL/HTML reference instead of
 * actual image data, so the caller can show the copy-image hint.
 *
 * Click-to-browse is scoped to `dropZonePromptElement` (the empty-state prompt), not
 * the whole drop zone: once a photo is showing, the drop zone also contains the
 * Start Over button, and wiring the click handler to the full zone would pop the file
 * picker open every time that button is clicked. Drag-and-drop stays wired to the
 * full `dropZoneElement` in both states, since replacing an already-loaded photo by
 * dropping a new one on top of it is a real flow (see the paste-again confirm in main.js).
 */
export function initializeInputDoors({
  dropZoneElement,
  dropZonePromptElement,
  fileInputElement,
  onImageFile,
  onNoUsableImage,
}) {
  document.addEventListener("paste", (event) => handlePaste(event, onImageFile));

  dropZoneElement.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropZoneElement.classList.add("drop-zone--drag-active");
  });

  dropZoneElement.addEventListener("dragleave", () => {
    dropZoneElement.classList.remove("drop-zone--drag-active");
  });

  dropZoneElement.addEventListener("drop", (event) => {
    event.preventDefault();
    dropZoneElement.classList.remove("drop-zone--drag-active");
    handleDrop(event, onImageFile, onNoUsableImage);
  });

  dropZonePromptElement.addEventListener("click", () => fileInputElement.click());

  fileInputElement.addEventListener("change", () => {
    const [file] = fileInputElement.files;
    if (file) {
      onImageFile(file);
    }
    // Reset so selecting the same filename again still fires a "change" event.
    fileInputElement.value = "";
  });
}

function handlePaste(event, onImageFile) {
  const items = event.clipboardData?.items;
  if (!items) {
    return;
  }
  for (const item of items) {
    if (item.kind === "file" && item.type.startsWith(IMAGE_MIME_PREFIX)) {
      const file = item.getAsFile();
      if (file) {
        onImageFile(file);
      }
      return;
    }
  }
}

function handleDrop(event, onImageFile, onNoUsableImage) {
  const { files } = event.dataTransfer;
  const imageFile = [...files].find(isImageFile);
  if (imageFile) {
    onImageFile(imageFile);
    return;
  }
  if (files.length > 0) {
    // A file arrived, but not an image (e.g. a dragged .pdf). Same hint path as a
    // pixel-less drop — either way, there is no image for us to decode.
    onNoUsableImage("not-an-image-file");
    return;
  }
  // No File at all: this is the cross-tab drag case (URL/HTML reference only).
  const carriedUrlOrHtml = event.dataTransfer.types.some(
    (type) => type === "text/uri-list" || type === "text/html",
  );
  onNoUsableImage(carriedUrlOrHtml ? "url-reference-only" : "empty-drop");
}
