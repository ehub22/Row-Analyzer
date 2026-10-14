# Row Analyzer

A private, offline rowing-video review workbench for desktop Chrome. **Research preview, not a completed or validated biomechanics product.**

- Pages URL: https://ehub22.github.io/Row-Analyzer/ (deployment status is visible in GitHub Actions).
- [Complete roadmap](ROADMAP.md)
- [Model contracts and setup](docs/MODELS.md)
- [Validation results and limitations](docs/VALIDATION.md)

## What works

- Local MP4/MOV upload and native playback, slow motion, scrubbing, decoded-timestamp frame stepping after analysis.
- WebCodecs + MP4Box decoder, worker/OffscreenCanvas preprocessing, bounded decoded-frame queue, progress and cancellation.
- Browser ONNX Runtime Web pipeline: **WebGPU now actually runs.** The previous build pinned the wrong ONNX Runtime WASM module for the WebGPU entry point, so every GPU session failed on every device and browser with `webgpuInit is not a function` and silently fell back to CPU. The runtime now pins the WASM build each entry point was compiled against, probes the adapter before touching a model, and hands the probed adapter to ONNX Runtime. Every WebGPU session still has to pass a warm-up inference before decoding starts, and when the GPU is unavailable (or dies mid-run: device lost, driver reset) the analysis continues on the **parallel WASM pose-worker pool** with the exact reason recorded — no silent fallback. The workspace and every export show which runtime ran, e.g. `WebGPU (apple · metal-3) detector / WebGPU pose`. Threaded WASM additionally activates when the page is cross-origin isolated (COOP/COEP; the bundled service worker enables this on GitHub Pages). **Requires separately imported authentic detector and whole-body model weights.**
- Top-down YOLOX / YOLOv8/11 detector adapters, aspect-correct padded crops, 133-joint SimCC or heatmap pose adapters, sub-pixel decoding, four-pass flip/multi-scale coordinate fusion.
- Global seat assignment with ambiguity gates; short bidirectional gap filling; observed/inferred/manual/missing provenance; low-confidence jump rejection.
- Per-seat and crew review; catch/finish **body-motion proxies**, stroke rate, drive-to-recovery ratio, crew timing in milliseconds and phase degrees.
- Manual keypoint corrections with local refit, partial COCO-style training labels, JSON sessions, CSV tables, PDF reports.
- Saved-session timing comparison. Explicitly labeled synthetic demonstration.

## Not completed / not validated

Automatic multi-cue camera azimuth, motion-compensated boat geometry, temporal 3D lifting, calibrated viewpoint-normalized biomechanics, blade/handle/oar detection, learned rowing-specific occlusion completion, reliable sequencing diagnoses, normalized 3D/model-stroke comparison, and a GPU companion are **not implemented**. Associated metrics are withheld, not simulated.

Real pretrained weights could not be retrieved in the build environment because model hosts are outside its network allowlist. Browser inference was exercised with **constant test doubles**, not trained models. No rowing PCK/OKS, seat-identity accuracy, or calibrated angle accuracy is claimed. Read the validation report before using this as coaching evidence.

## Run locally

```sh
npm ci
npm run dev
```

Open the displayed address in current desktop Chrome. `npm run dev` binds to `0.0.0.0` for hosted previews. No server receives video or keypoints. No telemetry, third-party fonts, or inference API is used. The static app and ONNX runtime assets are served from the same origin.

```sh
npm test               # geometry, pose decoding, occlusion, seat identity, timing, import tests
npm run build          # strict TypeScript + production bundle
npx playwright install chromium
npm run test:e2e       # production-browser workflow tests
```

## Use

1. Import a short MP4/MOV clip (up to 300 MB, 20,000 decoded-frame guard). Choose the shell, rigging, seat count, best-visible seat, and bow direction in the image.
2. Open **Models & settings**. When the build has model URLs configured, the detector and whole-body models download automatically on the first analysis (with progress) and are cached in this browser. Otherwise import compatible ONNX files from your device. Set input dimensions/format to match the export. Enable flip/multi-scale TTA for four passes per crop. **Parallel pose workers** controls how many CPU workers share those passes (Auto is right for most machines; each extra worker loads another copy of the pose model).
3. If needed, use the detection-region percentage bounds to exclude bystanders/coxswain detection centers. This is a static region; the crew must stay inside it.
4. Analyze. All decoded frames are submitted. Initialization stops if detected people do not match configured seats, rather than silently assigning incorrect identities.
5. Verify identity, observations, and occlusion. Use **Keyframe labels** to correct only visible joints. Select a joint and click to add a missing observation, or drag an existing point.
6. Export your session/report before starting a new session. Nothing is stored persistently automatically. To restore, open the JSON then import the original video with the same filename. Model weights (never video or keypoints) may be cached by the browser; remove them with *Clear cached models*.
7. Open **Compare** and import another exported JSON session. Current comparison is aggregate timing, not normalized 3D technique.

The synthetic demo works without model downloads and is always marked as synthetic in the UI and exports.

## Modules

| Module | Responsibility |
| --- | --- |
| `src/core/video.ts` | MP4 demux, codec validation, timestamped bounded decoding |
| `src/core/detection.ts` | detector output decoding, NMS, padded crop geometry |
| `src/core/pose.ts` | whole-body SimCC/heatmap decoding, mirror mapping, TTA fusion |
| `src/core/tracking.ts` | stable seat slots, global assignment, ambiguity abstention |
| `src/core/geometry.ts` | calibrated 3D transforms, scale utility, camera abstention |
| `src/core/occlusion.ts` | short-gap recovery, jump rejection, manual refit |
| `src/core/strokes.ts` | body-motion cycle proxies, evidence gating, crew synchronization |
| `src/core/export.ts` | sessions, partial training labels, CSV and PDF |
| `src/core/parallel.ts` / `src/core/pool.ts` | worker-pool + WASM-thread planning; least-busy pose job dispatch with failover |
| `src/workers/analysis.worker.ts` | ONNX sessions and offline pipeline orchestration |
| `src/workers/pose.worker.ts` | one pooled pose-inference session (×N, connected by MessageChannel) |

## Privacy and deployment

`google_drive/`, `private/`, `.cache/`, media, and model binaries are ignored. Downloaded sample videos, extracted frames, and athlete imagery are never included in the repository or Pages build. No sample Drive IDs are needed by the published app. Only `dist/` is uploaded by the Pages workflow.

Pages uses GitHub Actions, not a generated `gh-pages` branch. Build/test runs on the working branch `arena/d1e71cca-row-analyzer`. Publication runs after the pull request is merged into `main`, honoring the existing Pages environment policy. Relative asset URLs support `/Row-Analyzer/`. GitHub Pages cannot set the COOP/COEP headers that threaded WASM requires, so `public/coi-serviceworker.js` adds them from a service worker after one reload; the pose-worker pool keeps analysis parallel even where that is unavailable.

## Modeling principles

Bow seat = seat 1; the highest numbered seat is stroke seat. Sweep and sculling are distinct; no symmetric-body assumption is made. Confidence is an **uncalibrated evidence score**, not the probability a point is correct. A 2D shell axis is not camera azimuth. A plausible skeleton is not ground truth. Video panning is not calibrated speed, so split per 500 m is withheld.

## Deployment policy

The existing `github-pages` environment allows only `main`. The workflow therefore builds and tests the session branch, and deploys only the merged main-branch code. No environment protections are disabled or bypassed. The initial API attempt to enable Pages returned 403; subsequent inspection confirmed Pages is configured with GitHub Actions, and normal PR merge is the authorized publication path.
