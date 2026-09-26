/**
 * The small inline settings panel (spec 4.3, 4.5, 4.6), opened from the quiet "Settings"
 * link in the page header. Not a modal: it opens above the page content and the page
 * stays usable. The DOM is static in index.html; this module fills the lists and wires
 * each control to AppSettings (settings/app_settings.js) or BatchHistory
 * (storage/batch_history.js).
 *
 * Sections: date display, copy columns (column_order_list.js), the remembered payer names
 * (remove one, clear all), the co-op payee list (add, remove), the "Read handwriting"
 * switch (with a download bar), and "Clear everything this page remembers". Close (or Esc
 * inside the panel) folds it away and returns focus to the Settings link.
 *
 * The handwriting switch is off by default and never turns itself on. Turning it on saves
 * the choice and calls `enableHandwritingReader(enabled, onProgress)`; a failure turns it
 * back off and says so. At page load a switch saved "on" re-enables the reader only if its
 * files are already cached; otherwise it stays on with a "Download now" button, so a page
 * load never starts the 128 MB download by itself. Switching off keeps the cached files;
 * "Remove the download" deletes them.
 */

import { ColumnOrderList } from "./column_order_list.js";

const CLEAR_EVERYTHING_CONFIRM_MESSAGE =
  "Clear everything this page remembers? This removes the remembered payer names, the check history and your settings from this computer.";
const CLEAR_NAMES_CONFIRM_MESSAGE = "Forget every remembered payer name?";
const HANDWRITING_ON_MESSAGE = "On. Handwritten fields are read from now on.";
const HANDWRITING_NOT_DOWNLOADED_MESSAGE = "On, but the handwriting reader is not on this computer yet.";
const HANDWRITING_REMOVED_MESSAGE = "The download was removed from this computer.";
const HANDWRITING_FAILED_MESSAGE = "The handwriting reader could not be downloaded, so this is off again. Check the internet connection and try again.";
const REMOVE_GLYPH = "×";
const DOWNLOAD_PROGRESS_PATTERN = /(\d+(?:\.\d+)?) of (\d+(?:\.\d+)?) MB/;

function createRemovableNameItem(name, removeLabel, onRemove) {
  const item = document.createElement("li");
  item.className = "settings-name";
  const nameText = document.createElement("span");
  nameText.textContent = name;
  const removeButton = document.createElement("button");
  removeButton.type = "button";
  removeButton.className = "settings-remove-button";
  removeButton.textContent = REMOVE_GLYPH;
  removeButton.title = removeLabel;
  removeButton.setAttribute("aria-label", `${removeLabel}: ${name}`);
  removeButton.addEventListener("click", () => onRemove(name));
  item.append(nameText, removeButton);
  return item;
}

export class SettingsPanel {
  /**
   * `elements`: the ids in index.html (see main.js). `services`: `{ settings, batchHistory,
   * localStore, enableHandwritingReader(enabled, onProgress) -> Promise,
   * isHandwritingReaderCached() -> Promise<boolean>, deleteCachedHandwritingReader() -> Promise,
   * onKnownPayerNamesChanged(), onEverythingCleared() }`.
   */
  constructor(elements, services) {
    Object.assign(this, elements);
    Object.assign(this, services);
    this.handwritingRequestId = 0;
    this.columnOrderList = new ColumnOrderList(elements.copyColumnsList, (columns, keyToRefocus) => {
      this.settings.setCopyColumns(columns);
      this.columnOrderList.render(this.settings.getCopyColumns(), keyToRefocus);
    });
    this.toggleButton.addEventListener("click", () => this.setOpen(this.panelElement.hidden));
    this.closeButton.addEventListener("click", () => this.setOpen(false));
    this.panelElement.addEventListener("keydown", (event) => {
      if (event.key === "Escape") this.setOpen(false);
    });
    for (const radio of this.dateFormatRadios) {
      radio.addEventListener("change", () => { if (radio.checked) this.settings.setDateDisplayFormat(radio.value); });
    }
    this.clearKnownPayersButton.addEventListener("click", () => {
      if (!window.confirm(CLEAR_NAMES_CONFIRM_MESSAGE)) return;
      this.batchHistory.clearKnownPayerNames();
      this.renderKnownPayerNames();
      this.onKnownPayerNamesChanged();
    });
    this.addPayeeForm.addEventListener("submit", (event) => {
      event.preventDefault();
      this.settings.addPayeeName(this.addPayeeInput.value);
      this.addPayeeInput.value = "";
      this.renderPayeeNames();
    });
    this.handwritingSwitch.addEventListener("change", () => this.switchHandwritingReader(this.handwritingSwitch.checked));
    this.handwritingDownloadButton.addEventListener("click", () => this.switchHandwritingReader(true));
    this.handwritingRemoveButton.addEventListener("click", () => this.removeHandwritingDownload());
    this.clearEverythingButton.addEventListener("click", () => this.clearEverything());
    this.renderAll();
  }

  setOpen(isOpen) {
    const wasOpen = !this.panelElement.hidden;
    this.panelElement.hidden = !isOpen;
    this.toggleButton.setAttribute("aria-expanded", String(isOpen));
    if (isOpen) this.renderAll(); // names may have been added by Finish batch since last time
    if (wasOpen && !isOpen && this.panelElement.contains(document.activeElement)) this.toggleButton.focus();
  }

  renderAll() {
    const displayFormat = this.settings.getDateDisplayFormat();
    this.dateFormatRadios.forEach((radio) => { radio.checked = radio.value === displayFormat; });
    this.columnOrderList.render(this.settings.getCopyColumns());
    this.renderKnownPayerNames();
    this.renderPayeeNames();
    this.handwritingSwitch.checked = this.settings.isHandwritingReaderEnabled();
  }

  renderKnownPayerNames() {
    const names = this.batchHistory.getKnownPayerNames();
    this.knownPayersList.replaceChildren(...names.map((name) => createRemovableNameItem(name, "Forget this name", (removed) => {
      this.batchHistory.removeKnownPayerName(removed);
      this.renderKnownPayerNames();
      this.onKnownPayerNamesChanged();
    })));
    this.knownPayersEmptyLine.hidden = names.length > 0;
    this.clearKnownPayersButton.hidden = names.length === 0;
  }

  renderPayeeNames() {
    const payeeNames = this.settings.getPayeeNames();
    this.payeeNamesList.replaceChildren(...payeeNames.map((name) => createRemovableNameItem(name, "Remove this payee", (removed) => {
      this.settings.removePayeeName(removed);
      this.renderPayeeNames();
    })));
    this.payeeNamesEmptyLine.hidden = payeeNames.length > 0;
  }

  /** Status text under the switch; "12 of 128 MB" also drives the download bar. */
  showHandwritingStatus(text, tone = "normal") {
    this.handwritingStatusLine.textContent = text || "";
    this.handwritingStatusLine.dataset.tone = tone;
    this.handwritingStatusLine.hidden = !text;
    const downloadMatch = (text || "").match(DOWNLOAD_PROGRESS_PATTERN);
    this.handwritingProgress.hidden = !downloadMatch;
    if (downloadMatch) {
      const percent = Math.round((Number(downloadMatch[1]) / Number(downloadMatch[2])) * 100);
      this.handwritingProgressBar.style.setProperty("width", `${percent}%`);
      this.handwritingProgressTrack.setAttribute("aria-valuenow", String(percent));
    }
  }

  /** The switch was flipped (or the page loaded with it on): tell the reader, report inline. */
  switchHandwritingReader(enabled) {
    this.handwritingRequestId += 1;
    const requestId = this.handwritingRequestId;
    this.settings.setHandwritingReaderEnabled(enabled);
    this.handwritingSwitch.checked = enabled;
    this.handwritingDownloadButton.hidden = true;
    this.showHandwritingStatus(enabled ? "Getting the handwriting reader ready…" : null);
    const isCurrent = () => requestId === this.handwritingRequestId;
    Promise.resolve()
      .then(() => this.enableHandwritingReader(enabled, (progressText) => { if (isCurrent() && enabled) this.showHandwritingStatus(progressText); }))
      .then(() => {
        if (!isCurrent()) return;
        this.showHandwritingStatus(enabled ? HANDWRITING_ON_MESSAGE : null);
        this.refreshRemoveButton();
      })
      .catch((error) => {
        console.warn("switching the handwriting reader failed:", error);
        if (!isCurrent() || !enabled) return;
        this.settings.setHandwritingReaderEnabled(false);
        this.handwritingSwitch.checked = false;
        this.showHandwritingStatus(HANDWRITING_FAILED_MESSAGE, "error");
      });
  }

  /** Called once the engines are ready: re-enables a reader left on, but never downloads by itself. */
  restoreHandwritingReaderAtStartup() {
    this.refreshRemoveButton();
    if (!this.settings.isHandwritingReaderEnabled()) return;
    this.isHandwritingReaderCached().then((isCached) => {
      if (!this.settings.isHandwritingReaderEnabled()) return; // switched off meanwhile
      if (isCached) {
        this.switchHandwritingReader(true);
        return;
      }
      this.handwritingDownloadButton.hidden = false;
      this.showHandwritingStatus(HANDWRITING_NOT_DOWNLOADED_MESSAGE);
    });
  }

  /** "Remove the download" is offered only while the files are cached. */
  refreshRemoveButton() {
    this.isHandwritingReaderCached().then((isCached) => { this.handwritingRemoveButton.hidden = !isCached; });
  }

  /** Deletes the cached files; the reader goes off (it could not start again without them). */
  removeHandwritingDownload() {
    if (this.settings.isHandwritingReaderEnabled()) this.switchHandwritingReader(false);
    this.deleteCachedHandwritingReader().then(() => {
      this.handwritingRemoveButton.hidden = true;
      this.showHandwritingStatus(HANDWRITING_REMOVED_MESSAGE);
    });
  }

  clearEverything() {
    if (!window.confirm(CLEAR_EVERYTHING_CONFIRM_MESSAGE)) return;
    if (this.settings.isHandwritingReaderEnabled()) this.switchHandwritingReader(false);
    this.localStore.removeEveryOwnedKey(); // after the switch, which saves the settings once more
    this.batchHistory.forgetCachedCopies();
    this.settings.resetToDefaultsAfterStorageCleared();
    this.renderAll();
    this.onEverythingCleared();
  }
}
