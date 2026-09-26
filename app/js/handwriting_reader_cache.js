/**
 * Whether the opt-in handwriting reader's files are already in the service worker's cache
 * (sw.js HANDWRITING_READER_CACHE_NAME). Used at startup: a switch saved "on" only turns the
 * reader on by itself when nothing needs downloading; otherwise the operator gets a Download
 * button, so a page load never starts a 132 MB download on its own.
 */

import { HANDWRITING_READER_URLS } from "./cdn_config.js";

const HANDWRITING_READER_CACHE_NAME = "check-transcriber-handwriting-reader"; // must match sw.js

/** Resolves to true when all three files are cached. */
export function isHandwritingReaderCached() {
  if (!("caches" in self)) return Promise.resolve(false);
  const urls = [HANDWRITING_READER_URLS.encoder, HANDWRITING_READER_URLS.decoder, HANDWRITING_READER_URLS.tokenizer];
  return caches.open(HANDWRITING_READER_CACHE_NAME)
    .then((cache) => Promise.all(urls.map((url) => cache.match(url))))
    .then((responses) => responses.every(Boolean), () => false);
}

/** Deletes the cached download (the settings "remove the download" action). */
export function deleteCachedHandwritingReader() {
  return "caches" in self ? caches.delete(HANDWRITING_READER_CACHE_NAME) : Promise.resolve(false);
}
