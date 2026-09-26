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
 *
 * Drag-and-drop covers the WHOLE page, not just the drop zone: while a file (or a Gmail
 * image) is dragged over the window, a full-page overlay says "Drop the photo anywhere",
 * and a drop anywhere is taken in. That also stops a near-miss drop from making the
 * browser open the image in this tab and lose an open batch. Drags that start inside the
 * page (the settings panel's column list) are left alone.
 */

const IMAGE_MIME_PREFIX = "image/";

function isImageFile(file) {
  return file.type.startsWith(IMAGE_MIME_PREFIX);
}

/**
 * Wires up paste-anywhere, drag-and-drop anywhere on the page, and click-to-browse
 * on `dropZonePromptElement`. All three call `onImageFile(file)` on success.
 * `onNoUsableImage(reason)` fires when a drop carried a URL/HTML reference instead of
 * actual image data, so the caller can show the copy-image hint.
 *
 * Click-to-browse is scoped to `dropZonePromptElement` (the empty-state prompt), not
 * the whole drop zone: once a photo is showing, the drop zone also contains the
 * Start Over button, and wiring the click handler to the full zone would pop the file
 * picker open every time that button is clicked. Drag-and-drop works in every state,
 * since replacing an already-loaded photo by dropping a new one is a real flow (see the
 * paste-again confirm in main.js).
 */
export function initializeInputDoors({
  dropZoneElement,
  dropZonePromptElement,
  fileInputElement,
  dropOverlayElement,
  onImageFile,
  onNoUsableImage,
}) {
  document.addEventListener("paste", (event) => handlePaste(event, onImageFile));

  // dragenter/dragleave fire for every child element crossed, so count depth to know when
  // the drag really left the window.
  let dragDepth = 0;
  let dragStartedInPage = false;
  const setDragActive = (isActive) => {
    dropOverlayElement.hidden = !isActive;
    dropZoneElement.classList.toggle("drop-zone--drag-active", isActive);
  };
  document.addEventListener("dragstart", () => { dragStartedInPage = true; });
  document.addEventListener("dragend", () => { dragStartedInPage = false; dragDepth = 0; setDragActive(false); });
  document.addEventListener("dragenter", (event) => {
    if (dragStartedInPage) return;
    event.preventDefault();
    dragDepth += 1;
    setDragActive(true);
  });
  document.addEventListener("dragover", (event) => {
    if (dragStartedInPage) return;
    event.preventDefault(); // without this the browser would navigate to the dropped image
    setDragActive(true);
  });
  document.addEventListener("dragleave", () => {
    if (dragStartedInPage) return;
    dragDepth = Math.max(0, dragDepth - 1);
    if (dragDepth === 0) setDragActive(false);
  });
  document.addEventListener("drop", (event) => {
    if (dragStartedInPage) return;
    event.preventDefault();
    dragDepth = 0;
    setDragActive(false);
    handleDrop(event, onImageFile, onNoUsableImage);
  });

  dropZonePromptElement.addEventListener("click", () => fileInputElement.click());
  dropZonePromptElement.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      fileInputElement.click();
    }
  });

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
