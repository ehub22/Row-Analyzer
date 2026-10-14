/* Cross-origin-isolation shim for static hosting (GitHub Pages).
 *
 * Why: multi-threaded WASM (ONNX Runtime `numThreads > 1`) needs
 * SharedArrayBuffer, which browsers only expose on cross-origin-isolated
 * pages. Isolation requires COOP/COEP *response headers*, and GitHub Pages
 * cannot set custom headers. This service worker re-serves same-origin
 * responses with those headers added, so after one automatic reload the
 * page becomes crossOriginIsolated and threaded WASM unlocks. The app also
 * works without it — the pose-worker pool parallelises across cores either
 * way; this only adds per-session threads on top.
 *
 * This single file serves two roles:
 *  - loaded as a classic <script> in index.html, it registers itself as a
 *    service worker and reloads the page once it controls the page but the
 *    page is not isolated yet (guarded so it can never loop);
 *  - running as the service worker, it injects the headers on same-origin
 *    GET responses. Cross-origin requests (e.g. model downloads, which are
 *    CORS fetches and satisfy COEP themselves) are left untouched.
 */
(function () {
  if (typeof window !== "undefined") {
    // ---- page context: register + reload once ------------------------------
    if (window.crossOriginIsolated) {
      try {
        sessionStorage.removeItem("row-analyzer-coi-reload");
      } catch (e) {
        /* storage unavailable */
      }
      return;
    }
    if (!("serviceWorker" in navigator) || !window.isSecureContext) return;
    // Local dev/preview servers send real COOP/COEP headers (see
    // vite.config.ts), so the shim is only needed on deployed static hosts.
    // Skipping localhost also keeps it out of automated browser tests.
    if (
      window.location.hostname === "localhost" ||
      window.location.hostname === "127.0.0.1" ||
      window.location.hostname === "[::1]"
    )
      return;
    var RELOAD_KEY = "row-analyzer-coi-reload";
    var reloaded = false;
    try {
      reloaded = sessionStorage.getItem(RELOAD_KEY) === "1";
    } catch (e) {
      /* storage unavailable: treat as not reloaded, SW guards prevent loops */
    }
    function reloadOnce() {
      if (window.crossOriginIsolated || reloaded) return;
      reloaded = true;
      try {
        sessionStorage.setItem(RELOAD_KEY, "1");
      } catch (e) {
        /* ignore */
      }
      window.location.reload();
    }
    // When the worker takes control of an already-loaded (non-isolated)
    // page, one reload makes the browser re-request it through the worker,
    // which is what applies the headers to the document itself.
    navigator.serviceWorker.addEventListener("controllerchange", reloadOnce);
    var scriptUrl =
      (document.currentScript && document.currentScript.src) ||
      new URL("./coi-serviceworker.js", window.location.href).href;
    navigator.serviceWorker
      .register(scriptUrl)
      .then(function (registration) {
        // Already-active worker but this page predates it: same reload.
        if (registration.active && !navigator.serviceWorker.controller) {
          reloadOnce();
        }
      })
      .catch(function () {
        /* Service workers unavailable (private mode, old browser): the app
           still works via the pose-worker pool, just without threaded WASM. */
      });
    return;
  }

  // ---- service-worker context: inject isolation headers -------------------
  self.addEventListener("install", function () {
    self.skipWaiting();
  });
  self.addEventListener("activate", function (event) {
    event.waitUntil(self.clients.claim());
  });
  self.addEventListener("fetch", function (event) {
    var request = event.request;
    if (request.method !== "GET") return;
    var url;
    try {
      url = new URL(request.url);
    } catch (e) {
      return;
    }
    // Only same-origin responses need (and may safely receive) the headers.
    if (url.origin !== self.location.origin) return;
    event.respondWith(
      fetch(request)
        .then(function (response) {
          if (!response || response.status === 0) return response;
          var headers = new Headers(response.headers);
          headers.set("Cross-Origin-Opener-Policy", "same-origin");
          headers.set("Cross-Origin-Embedder-Policy", "require-corp");
          headers.set("Cross-Origin-Resource-Policy", "same-origin");
          return new Response(response.body, {
            status: response.status,
            statusText: response.statusText,
            headers: headers,
          });
        })
        .catch(function () {
          return fetch(request);
        }),
    );
  });
})();
