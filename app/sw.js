/**
 * Service worker: makes the app instant and fully offline after the first visit.
 *
 * Two caches. `SHELL_CACHE_NAME` holds this app's own files (index.html, css, js) so
 * the page itself loads with no network. `CDN_CACHE_NAME` holds whatever gets fetched
 * from the one CDN origin (OpenCV, Tesseract, and their WASM/language-data files) so
 * the ~17 MB first-visit download in engine_loader.js never happens twice.
 *
 * Deliberately a classic (non-module) service worker, not `type: "module"` — one
 * less browser-support variable to worry about, and this file is simple enough that
 * splitting it into ES modules would add indirection without adding clarity.
 *
 * Cache-first for everything in scope: a cached response wins even if the network
 * has something newer. This is what "instant and fully offline" means, and it is why
 * the caches are versioned — bump SW_VERSION to force every client to fetch fresh
 * copies of everything on their next visit.
 */

const SW_VERSION = "v1";
const SHELL_CACHE_NAME = `check-transcriber-shell-${SW_VERSION}`;
const CDN_CACHE_NAME = `check-transcriber-cdn-${SW_VERSION}`;
const CURRENT_CACHE_NAMES = [SHELL_CACHE_NAME, CDN_CACHE_NAME];

// The app's own files, precached on install so the very next visit is instant even
// before the runtime cache-fill below has had a chance to run. Paths are relative to
// this file's own location, which is also the app's root, so this resolves correctly
// whether served from "/" (local dev) or "/check-transcriber/" (GitHub Pages).
const APP_SHELL_PATHS = [
  "./",
  "./index.html",
  "./styles/base.css",
  "./styles/drop-zone.css",
  "./styles/status-line.css",
  "./js/main.js",
  "./js/cdn_config.js",
  "./js/heic_detect.js",
  "./js/image_decode.js",
  "./js/input_doors.js",
  "./js/engine_loader.js",
  "./js/status_line.js",
];

// The CDN origin pinned in js/cdn_config.js. Kept as a literal here (rather than
// importing the module) because a classic service worker can't use ES module
// `import`; if that origin ever changes, update it in both places.
const CDN_ORIGIN = "https://cdn.jsdelivr.net";

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(SHELL_CACHE_NAME).then((cache) => cache.addAll(APP_SHELL_PATHS)),
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((existingCacheNames) =>
      Promise.all(
        existingCacheNames
          .filter((name) => !CURRENT_CACHE_NAMES.includes(name))
          .map((staleName) => caches.delete(staleName)),
      ),
    ),
  );
  self.clients.claim();
});

/** Serves `request` from `cacheName` if present; otherwise fetches, caches, and returns it. */
async function cacheFirstThenNetwork(request, cacheName) {
  const cache = await caches.open(cacheName);
  const cachedResponse = await cache.match(request);
  if (cachedResponse) {
    return cachedResponse;
  }
  const networkResponse = await fetch(request);
  // `networkResponse.ok` is false for a genuine error AND for every opaque response
  // (status 0, unreadable) — and OpenCV/Tesseract's <script src> and importScripts()
  // loads are "no-cors" by nature, so their responses come back opaque. An opaque
  // response is still cacheable and still replays correctly as a working script (the
  // browser trusts it was a real network response at write time; JS just can't
  // introspect it) — confirmed by testing: gating on `.ok` alone silently skipped
  // caching every one of these files, leaving only the one library file (Tesseract's
  // language data) that happens to be fetched with an explicit `fetch()` call (cors
  // mode, so it's a normal readable response) ever making it into the cache.
  const isCacheable = networkResponse.ok || networkResponse.type === "opaque";
  if (isCacheable) {
    // Awaited deliberately: `respondWith` only keeps this worker alive until the
    // promise IT was given resolves, so an un-awaited `cache.put()` on a large file
    // (OpenCV.js is ~10 MB) can lose the race against worker teardown and never
    // finish writing.
    await cache.put(request, networkResponse.clone());
  }
  return networkResponse;
}

self.addEventListener("fetch", (event) => {
  const { request } = event;
  if (request.method !== "GET") {
    return;
  }

  const requestOrigin = new URL(request.url).origin;
  if (requestOrigin === CDN_ORIGIN) {
    event.respondWith(cacheFirstThenNetwork(request, CDN_CACHE_NAME));
  } else if (requestOrigin === self.location.origin) {
    event.respondWith(cacheFirstThenNetwork(request, SHELL_CACHE_NAME));
  }
  // Any other origin falls through untouched. The page's CSP already forbids
  // requests to anywhere else, so in practice this branch never runs — but if it
  // ever did, the right behavior is a normal passthrough fetch, not a silent block.
});
