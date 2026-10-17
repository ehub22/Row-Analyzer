import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Cross-origin isolation unlocks SharedArrayBuffer, which ONNX Runtime Web
// needs for multi-threaded WASM (without it numThreads is silently clamped
// to 1). Dev/preview servers can send the headers directly; the deployed
// static site cannot, so public/coi-serviceworker.js adds them there.
const isolationHeaders = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

export default defineConfig({
  plugins: [react()],
  base: "./",
  server: { host: "0.0.0.0", allowedHosts: true, headers: isolationHeaders },
  // Same as `server`: a preview behind a proxy is asked for a host it did not
  // configure, and blocking it shows a blank frame instead of the app.
  preview: {
    host: "0.0.0.0",
    allowedHosts: true,
    headers: isolationHeaders,
  },
  worker: { format: "es" },
  build: { chunkSizeWarningLimit: 1500 },
});
