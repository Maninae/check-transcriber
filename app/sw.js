/**
 * Service worker: makes the app instant and fully offline after the first visit.
 *
 * Two caches. `SHELL_CACHE_NAME` holds this app's own files (index.html, css, js) so
 * the page itself loads with no network. `CDN_CACHE_NAME` holds whatever gets fetched
 * from the one CDN origin (OpenCV, onnxruntime-web, Tesseract, and their WASM/language-
 * data files) so the ~16 MB first-visit download in engine_loader.js never happens twice.
 * The pipeline worker's own fetches (importScripts, ORT's dynamic import and .wasm)
 * go through here too: a dedicated worker is controlled by its page's service worker.
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

const SW_VERSION = "v2";
const SHELL_CACHE_NAME = `check-transcriber-shell-${SW_VERSION}`;
const CDN_CACHE_NAME = `check-transcriber-cdn-${SW_VERSION}`;
const CURRENT_CACHE_NAMES = [SHELL_CACHE_NAME, CDN_CACHE_NAME];

// The app's own files (including models/upside_down_classifier.onnx), precached on
// install so the very next visit is instant and offline even before the runtime
// cache-fill below has had a chance to run. Generated: run
// `python3 tests/sync_service_worker_shell_list.py` after adding or removing a file
// (the smoke test fails while this list is stale). Paths are relative to
// this file's own location, which is also the app's root, so this resolves correctly
// whether served from "/" (local dev) or "/check-transcriber/" (GitHub Pages).
const APP_SHELL_PATHS = [
  "./",
  "./index.html",
  "./styles/base.css",
  "./styles/count-step.css",
  "./styles/drop-zone.css",
  "./styles/lightbox.css",
  "./styles/review-grid.css",
  "./styles/status-line.css",
  "./styles/step-indicator.css",
  "./js/batch_flow.js",
  "./js/cdn_config.js",
  "./js/count/count_header_text.js",
  "./js/count/count_overlay.js",
  "./js/count/count_step.js",
  "./js/engine_loader.js",
  "./js/heic_detect.js",
  "./js/image_decode.js",
  "./js/input_doors.js",
  "./js/main.js",
  "./js/pipeline/check_pipeline_stage.js",
  "./js/pipeline/check_rectification.js",
  "./js/pipeline/classical/candidates/adjacent_cell_merging.js",
  "./js/pipeline/classical/candidates/candidate_masks.js",
  "./js/pipeline/classical/candidates/candidate_regions.js",
  "./js/pipeline/classical/candidates/collinear_segment_merging.js",
  "./js/pipeline/classical/candidates/line_quadrilateral_hypotheses.js",
  "./js/pipeline/classical/candidates/line_segment_extraction.js",
  "./js/pipeline/classical/candidates/probabilistic_hough_lines.js",
  "./js/pipeline/classical/classical_detector_config.js",
  "./js/pipeline/classical/detect_checks_classical.js",
  "./js/pipeline/classical/geometry/edge_line_snapping.js",
  "./js/pipeline/classical/geometry/full_resolution_edge_refinement.js",
  "./js/pipeline/classical/geometry/quadrilateral_fitting.js",
  "./js/pipeline/classical/geometry/quadrilateral_geometry.js",
  "./js/pipeline/classical/numeric/bilinear_sampling.js",
  "./js/pipeline/classical/numeric/binary_morphology.js",
  "./js/pipeline/classical/numeric/bit_packed_rows.js",
  "./js/pipeline/classical/numeric/fused_multiply_add.js",
  "./js/pipeline/classical/numeric/huber_line_fit.js",
  "./js/pipeline/classical/numeric/mat_helpers.js",
  "./js/pipeline/classical/numeric/numpy_argsort.js",
  "./js/pipeline/classical/numeric/numpy_compatibility.js",
  "./js/pipeline/classical/numeric/numpy_statistics.js",
  "./js/pipeline/classical/numeric/opencv_geometry_formulas.js",
  "./js/pipeline/classical/numeric/opencv_random_generator.js",
  "./js/pipeline/classical/preprocessing/area_resize.js",
  "./js/pipeline/classical/preprocessing/combined_edge_map.js",
  "./js/pipeline/classical/preprocessing/neon_magnitude.js",
  "./js/pipeline/classical/preprocessing/separable_filters_float32.js",
  "./js/pipeline/classical/preprocessing/working_image_channels.js",
  "./js/pipeline/classical/verification/candidate_selection.js",
  "./js/pipeline/classical/verification/interior_appearance.js",
  "./js/pipeline/classical/verification/interior_seam_detection.js",
  "./js/pipeline/classical/verification/quadrilateral_verification.js",
  "./js/pipeline/detection_confidence.js",
  "./js/pipeline/drawn_rectangle_refit.js",
  "./js/pipeline/orientation/check_orientation.js",
  "./js/pipeline/orientation/upside_down_classifier.js",
  "./js/pipeline/photo_detection_stage.js",
  "./js/pipeline/pipeline_client.js",
  "./js/pipeline/pipeline_worker.js",
  "./js/pipeline/quadrilateral_math.js",
  "./js/pipeline/reading_order.js",
  "./js/pipeline/refinement/edge_feature_scoring.js",
  "./js/pipeline/refinement/edge_profile_sampling.js",
  "./js/pipeline/refinement/image_sampling.js",
  "./js/pipeline/refinement/numpy_compatible_math.js",
  "./js/pipeline/refinement/overlap_masking.js",
  "./js/pipeline/refinement/paper_colour_estimation.js",
  "./js/pipeline/refinement/quadrilateral_geometry.js",
  "./js/pipeline/refinement/quadrilateral_refinement.js",
  "./js/pipeline/refinement/refinement_config.js",
  "./js/pipeline/refinement/robust_side_curve_fitting.js",
  "./js/pipeline/refinement/seeded_random_generator.js",
  "./js/pipeline/refinement/side_curve.js",
  "./js/pipeline/refinement/side_edge_tracking.js",
  "./js/pipeline/refinement/side_line_search.js",
  "./js/pipeline/refinement/side_refinement.js",
  "./js/pipeline/worker_stages.js",
  "./js/review/clipboard_rows.js",
  "./js/review/crop_rendering.js",
  "./js/review/field_definitions.js",
  "./js/review/field_undo.js",
  "./js/review/lightbox.js",
  "./js/review/review_grid.js",
  "./js/review/review_row.js",
  "./js/status_line.js",
  "./js/step_indicator.js",
  "./js/toast.js",
  "./models/upside_down_classifier.onnx",
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
