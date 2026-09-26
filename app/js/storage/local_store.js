/**
 * The one door to the browser's localStorage (spec 4.6 and section 7): text only, every key
 * under one namespace so "Clear everything this page remembers" can find and wipe exactly
 * what this app owns and nothing else on the origin.
 *
 * - Values are JSON. Callers pass plain data (names, check numbers, dates, settings);
 *   `writeJson` refuses anything that is not plain JSON data, so a canvas, ImageData or
 *   typed array can never be written by mistake (photo pixels are memory-only).
 * - A missing, corrupt or blocked storage reads as the default: the page keeps working,
 *   it just forgets between visits.
 * - The storage object is injected (`window.localStorage` in the page, a Map-backed stub
 *   in the Node unit tests).
 */

// Every key this app has ever owned starts with this; the version segment is below it.
export const APP_STORAGE_ROOT_PREFIX = "checkTranscriber.";
export const APP_STORAGE_PREFIX = `${APP_STORAGE_ROOT_PREFIX}v1.`;

function isPlainJsonData(value) {
  if (value === null) return true;
  const valueType = typeof value;
  if (valueType === "string" || valueType === "number" || valueType === "boolean") return true;
  if (Array.isArray(value)) return value.every(isPlainJsonData);
  if (valueType !== "object" || Object.getPrototypeOf(value) !== Object.prototype) return false;
  return Object.values(value).every(isPlainJsonData);
}

export class LocalStore {
  /** `storage`: a Web Storage object (`window.localStorage`), or null when blocked. */
  constructor(storage) {
    this.storage = storage;
  }

  readJson(key, defaultValue) {
    if (!this.storage) return defaultValue;
    try {
      const text = this.storage.getItem(APP_STORAGE_PREFIX + key);
      return text === null ? defaultValue : JSON.parse(text);
    } catch (error) {
      console.warn(`could not read ${key} from local storage:`, error);
      return defaultValue;
    }
  }

  writeJson(key, value) {
    if (!isPlainJsonData(value)) throw new Error(`refusing to store non-text data under ${key}`);
    if (!this.storage) return;
    try {
      this.storage.setItem(APP_STORAGE_PREFIX + key, JSON.stringify(value));
    } catch (error) {
      console.warn(`could not save ${key} to local storage:`, error);
    }
  }

  /** Every key this app owns, any version. */
  listOwnedKeys() {
    if (!this.storage) return [];
    const ownedKeys = [];
    for (let index = 0; index < this.storage.length; index += 1) {
      const key = this.storage.key(index);
      if (key && key.startsWith(APP_STORAGE_ROOT_PREFIX)) ownedKeys.push(key);
    }
    return ownedKeys;
  }

  /** "Clear everything this page remembers": removes every owned key; returns how many. */
  removeEveryOwnedKey() {
    const ownedKeys = this.listOwnedKeys();
    ownedKeys.forEach((key) => this.storage.removeItem(key));
    return ownedKeys.length;
  }
}

/** The page's store; tolerates a browser that blocks storage access entirely. */
export function openBrowserLocalStore() {
  try {
    return new LocalStore(window.localStorage);
  } catch (error) {
    console.warn("local storage is unavailable; nothing will be remembered:", error);
    return new LocalStore(null);
  }
}
