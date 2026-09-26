/**
 * The small inline settings panel (spec 4.3, 4.5, 4.6), opened from the quiet "Settings"
 * link in the page header. Not a modal: it opens above the page content and the page
 * stays usable. The DOM is static in index.html; this module fills the lists and wires
 * each control to AppSettings (settings/app_settings.js) or BatchHistory
 * (storage/batch_history.js).
 *
 * Sections: date display, copy columns (column_order_list.js), the remembered payer names
 * (remove one, clear all), the co-op payee list (add, remove), the "Read handwriting"
 * switch, and "Clear everything this page remembers".
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
    this.panelElement.hidden = !isOpen;
    this.toggleButton.setAttribute("aria-expanded", String(isOpen));
    if (isOpen) this.renderAll(); // names may have been added by Finish batch since last time
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
    this.payeeNamesList.replaceChildren(...this.settings.getPayeeNames().map((name) => createRemovableNameItem(name, "Remove this payee", (removed) => {
      this.settings.removePayeeName(removed);
      this.renderPayeeNames();
    })));
  }

  showHandwritingStatus(text) {
    this.handwritingStatusLine.textContent = text || "";
    this.handwritingStatusLine.hidden = !text;
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
        this.showHandwritingStatus(HANDWRITING_FAILED_MESSAGE);
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
