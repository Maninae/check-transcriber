/**
 * Builds the settings panel from index.html's static DOM and connects it to the rest of the
 * page, so main.js only hands over the services. Also owns the one rule about the
 * handwriting reader's timing: a switch flipped (or restored at page load) before the
 * engines are ready waits for them, then goes through `pipelineClient.setHandwritingReaderEnabled`.
 */

import { SettingsPanel } from "./settings_panel.js";
import { SETTING_NAMES } from "./app_settings.js";

function queryRequiredElement(id) {
  const element = document.getElementById(id);
  if (!element) throw new Error(`expected element #${id} to exist in index.html`);
  return element;
}

/**
 * `services`: `{ settings, batchHistory, localStore, reviewGrid, whenPipelineClientReady }`,
 * where `whenPipelineClientReady()` resolves to the PipelineClient once the engines are up.
 * Returns the SettingsPanel.
 */
export function createSettingsPanel({ settings, batchHistory, localStore, reviewGrid, whenPipelineClientReady }) {
  settings.onChange((settingName) => {
    if (settingName === SETTING_NAMES.DATE_DISPLAY_FORMAT || settingName === SETTING_NAMES.EVERYTHING) reviewGrid.refreshDateDisplay();
    if (settingName === SETTING_NAMES.PAYEE_NAMES || settingName === SETTING_NAMES.EVERYTHING) reviewGrid.regateAllRows();
  });
  return new SettingsPanel({
    panelElement: queryRequiredElement("settings-panel"),
    toggleButton: queryRequiredElement("settings-button"),
    dateFormatRadios: [...document.querySelectorAll('input[name="date-display-format"]')],
    copyColumnsList: queryRequiredElement("copy-columns-list"),
    knownPayersList: queryRequiredElement("known-payers-list"),
    knownPayersEmptyLine: queryRequiredElement("known-payers-empty"),
    clearKnownPayersButton: queryRequiredElement("clear-known-payers-button"),
    payeeNamesList: queryRequiredElement("payee-names-list"),
    addPayeeForm: queryRequiredElement("add-payee-form"),
    addPayeeInput: queryRequiredElement("add-payee-input"),
    handwritingSwitch: queryRequiredElement("handwriting-switch"),
    handwritingStatusLine: queryRequiredElement("handwriting-status"),
    clearEverythingButton: queryRequiredElement("clear-everything-button"),
  }, {
    settings,
    batchHistory,
    localStore,
    enableHandwritingReader: (enabled, onProgress) =>
      whenPipelineClientReady().then((pipelineClient) => pipelineClient.setHandwritingReaderEnabled(enabled, onProgress)),
    onKnownPayerNamesChanged: () => reviewGrid.regateAllRows(),
    onEverythingCleared: () => reviewGrid.regateAllRows(),
  });
}
