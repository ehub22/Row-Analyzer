# Row Analyzer

A private, offline rowing-video review workbench for desktop Chrome. **Research preview, not a completed or validated biomechanics product.**

- Pages URL: https://ehub22.github.io/Row-Analyzer/ (deployment status is visible in GitHub Actions).
- [Complete roadmap](ROADMAP.md)
- [Model contracts and setup](docs/MODELS.md)
- [Validation results and limitations](docs/VALIDATION.md)

## What works

- Local MP4/MOV upload and native playback, slow motion, scrubbing, decoded-timestamp frame stepping after analysis.
- WebCodecs + MP4Box decoder, worker/OffscreenCanvas preprocessing, bounded decoded-frame queue, progress and cancellation.
- Browser ONNX Runtime Web pipeline: **WebGPU now actually runs.** The previous build pinned the wrong ONNX Runtime WASM module for the WebGPU entry point, so every GPU session failed on every device and browser with `webgpuInit is not a function` and silently fell back to CPU. The runtime now pins the WASM build each entry point was compiled against, probes the adapter before touching a model, and hands the probed adapter to ONNX Runtime. Every WebGPU session still has to pass a warm-up inference before decoding starts, and when the GPU is unavailable (or dies mid-run: device lost, driver reset) the analysis continues on the **parallel WASM pose-worker pool** with the exact reason recorded — no silent fallback. The workspace and every export show which runtime ran, e.g. `WebGPU (apple · metal-3) detector / WebGPU pose`. Threaded WASM additionally activates when the page is cross-origin isolated (COOP/COEP; the bundled service worker enables this on GitHub Pages). **Requires compatible detector and pose ONNX weights; the build does not bundle them.**
- Top-down YOLOX / YOLOv8/11 detector adapters, aspect-correct padded crops, COCO-17 body-only or COCO-WholeBody-133 SimCC/heatmap pose adapters, sub-pixel decoding, four-pass flip/multi-scale coordinate fusion. Body-only runs retain 133 session slots but mark unmodeled joints missing, never observed.
- Global seat assignment with ambiguity gates; short bidirectional gap filling; observed/inferred/manual/missing provenance; low-confidence jump rejection.
- Per-seat and crew review; catch/finish **body-motion proxies** from thigh (hip-to-knee) elevation, stroke rate, drive-to-recovery ratio, crew timing in milliseconds and phase degrees.
- **Hidden feet are handled as expected occlusion.** On the water the feet are strapped to the stretcher below the gunwale, so a whole-body model can only guess them. With _Feet in view: Hidden by the hull_ (the default) those guesses are discarded instead of tracked: stroke cycles are timed from hips and knees, which stay above the gunwale; the adaptive TTA gate stops paying four passes per crop for joints no pass can reveal; and when the knees also drop behind the hull with the legs flat, the cycle is bridged rather than lost. Choose _Visible_ for ergometer or overhead footage.
- **A knee the gunwale or the oar covers is not tracked either.** With the legs flat the knee drops to the hull edge, and in a side view the loom crosses the legs on the way out; a pose model answers anyway — pinned to the gunwale, or dragged onto the shaft — at a confidence that reads as observed, which lifted the trough every stroke event is timed from and moved the finish proxy over a quarter of the cycle. Each seat's own clear-view knee near the catch calibrates what a visible knee looks like on that footage; where the low part of the cycle measures weaker than that — densely, and for long enough to be an occluder rather than noise — the guesses are discarded with a reason and the finish is timed from the bridged trough instead. A model does not answer the same covered knee the same way twice — it finds the hull edge on one frame and loses the joint on the next — so a crossing it answers inconsistently is still read as one window rather than as runs too short to believe. The catch end of the stroke is left exactly as the model reported it, footage whose knees the model already drops is untouched, and a knee the hull was in front of is not counted as evidence the model failed to find.
- Manual keypoint corrections with local refit, partial COCO-style training labels, JSON sessions, CSV tables, PDF reports.
- Saved-session timing comparison. Explicitly labeled synthetic demonstration.

## Not completed / not validated

Automatic multi-cue camera azimuth, motion-compensated boat geometry, temporal 3D lifting, calibrated viewpoint-normalized biomechanics, blade/handle/oar detection, learned rowing-specific occlusion completion, reliable sequencing diagnoses, normalized 3D/model-stroke comparison, and a GPU companion are **not implemented**. Associated metrics are withheld, not simulated.

The user-supplied `yolox_l.onnx` and `dw-ll_ucoco_384.onnx` were exercised locally on a short clip. Their checkpoint provenance and tracking accuracy are not independently verified; constant test doubles still underpin the pipeline regression tests. A smaller, genuinely trained COCO-17 ONNX is required for any body-only speed comparison—slicing the output of the existing 133-joint model does not speed its backbone. No rowing PCK/OKS, seat-identity accuracy, or calibrated angle accuracy is claimed. Read the validation report before using this as coaching evidence.

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
2. Open **Models & settings**. When the build has model URLs configured, the detector and whole-body models download automatically on the first analysis (with progress) and are cached in this browser. Otherwise import compatible ONNX files from your device. Set **Pose joint layout** to 133 (default) or 17 and match input dimensions/format to the export. Body-only mode requires a separate local COCO-17 model: cached 133-joint weights are never silently used. **Test-time augmentation** chooses how many pose inferences one crop gets: _Adaptive_ (default) runs the single pass first and spends the other three — second scale plus both flips — only where that pass left a body joint below 0.3, _Always_ pays four passes on every crop, _Off_ pays one. **Parallel pose workers** controls how many CPU workers share those passes (Auto is right for most machines; each extra worker loads another copy of the pose model). **Detector cadence** defaults to _Every frame_ for a fresh person detection on each frame. Experimental _Auto_ begins the same way and reuses a guarded crop for at most two frames only when measured detector time justifies it; every decoded frame with a viable seat still gets a fresh pose inference. Validate identities before relying on the experimental mode.
3. If needed, use the detection-region percentage bounds to exclude bystanders/coxswain detection centers. This is a static region; the crew must stay inside it.
4. Analyze. All decoded frames are submitted. Initialization stops if detected people do not match configured seats, rather than silently assigning incorrect identities.
5. Verify identity, observations, and occlusion. Use **Keyframe labels** to correct only visible joints. Select a joint and click to add a missing observation, or drag an existing point.
6. Export your session/report before starting a new session. Nothing is stored persistently automatically. To restore, open the JSON then import the original video with the same filename. Model weights (never video or keypoints) may be cached by the browser; remove them with _Clear cached models_.
7. Open **Compare** and import another exported JSON session. Current comparison is aggregate timing, not normalized 3D technique.

The synthetic demo works without model downloads and is always marked as synthetic in the UI and exports.

## Speed: where the time goes, and what to turn

Every run now measures itself. The workspace's _Analysis notes_ start with a
performance line (also stored in exported sessions as `performance`), for
example:

```
18.4 s for 1200 frames (65.2 fps) · pose 71% · detector 22% · track 1% ·
decode wait 5% · pose.infer 84.1 ms/pass · pose.preprocess 2.3 ms/pass ·
detector.preprocess 6.1 ms/pass · 2400 pose passes, 7200 skipped by adaptive TTA
```

Shares are of wall time, so `pose` and `detector` can add up past 100%: with a
worker pool the two stages run at the same time. `ms/pass` figures are measured
per pass (in the pose workers for `pose.*`), which is what tells inference apart
from preprocessing.

On a 30-frame Quad excerpt of `5.mov` with the provided YOLOX-L/DWPose-L files and four CPU pose workers, **TTA Off took 55.2 s** and Adaptive took 62.0 s with a two-thread detector. Assigning an otherwise idle core to the detector reduced the matched Off run to **40.7 s** (three detector threads, same 4×4 pose pool). YOLOX-L detector inference still averaged **1.27 s per frame**, while pose inference averaged about 0.31 s per pass. These are local provided-weight throughput measurements, not tracking accuracy or a full-clip timing claim. A 17-joint pose model by itself cannot bring this detector under a minute for the 758-frame clip; a compatible smaller detector is also needed. No alternative trained weights are bundled here. After local authorization, the privately downloaded YOLOX-Nano + RTMPose-t COCO-17 pair processed the full clip in **50.5 s** from Analyze to result on bundled Chromium 153, and **35.5 s** on system Chrome 155, both using CPU/WASM, every-frame detection and TTA Off. Browser versions differ, so these are not a controlled A/B. The speed target is met on this machine, but tracking quality remains unvalidated: only one of 33 segmented seat-cycles yielded a reliable nonwithheld rate. The UI withholds a crew average when fewer than half its seats supply reliable cycles.

Levers to test against your run's stage report (the supplied large-model pair is detector-bound):

1. **Test-time augmentation.** Four passes per crop is four whole-body
   inferences. Adaptive (the default) runs one pass first and only pays for the
   rest where a body joint came back below 0.3 — on clean footage that is the
   single biggest saving available, and where a crop _is_ weak the same four
   passes fuse exactly as under _Always_. Joints the session declares hidden
   (the ankles, on the water) do not count: before that, a guessed ankle made
   nearly every on-water crop pay all four passes.
2. **Pose workers and cross-origin isolation.** The pool is how CPU inference
   scales; threaded WASM per session additionally needs COOP/COEP (the bundled
   service worker enables it on Pages after one reload). The settings dialog
   shows the exact plan for the machine it is running on.
3. **Input resolution and model size.** Cost is roughly quadratic in the model
   input, so a whole-body export at 256×192 costs about a third of one at
   384×288, and a medium checkpoint costs a fraction of a large one. The
   settings must match the export exactly, and the contract check fails fast if
   they do not.
4. **The detector, when the report says so.** Experimental Auto cadence begins
   with full detection and only reuses a short-lived crop when detector time is
   high relative to pose time. It refreshes on weak, missing, ambiguous or drifting
   poses, with a new detector run at least every third frame. Joints still come
   from the current frame; predicted crop positions are not fresh detections
   and reduce seat-identity confidence. On a pose-bound run it deliberately
   keeps detecting every frame. _Every frame_ is the default baseline;
   a compatible smaller detector may help if the detector remains the bottleneck.
5. **Fewer pixels of crew.** The detection-region bounds and the boat's seat
   count decide how many crops exist at all; a single scull is one crop per
   frame, an eight is eight.

Fixed costs are worth knowing too: creating ONNX sessions and starting the pool
takes seconds regardless of clip length, so very short clips are dominated by
startup — the performance line reports only the analysis itself.

Two implementation choices are what made the pose stage cheap, and both are
verified rather than assumed:

- **One source bitmap, crop-sized transfers.** Drawing a decoded `VideoFrame`
  repeatedly into a canvas costs a full-frame readback even for a small crop.
  The detector and preview share one bitmap; each pose pass transfers only the
  bounded image region it needs to its worker. The original full-frame crop
  coordinates are retained for keypoint decoding. On `5.mov`,
  `node scripts/crop-equivalence-check.mjs` compares 50 real-frame pose tensors
  (including flips and edge crops) against the full-frame path: all floats
  matched exactly. Transferring _full-frame_ bitmap clones was slower on this
  clip and is not used.
- **Crops are rasterised in the workers.** Each pose worker draws and normalises
  its own crop from that bitmap into a reused buffer. Previously rasterisation
  happened serially on the coordinator while every pooled worker waited for
  work; now it is parallel, off the critical path, and allocates nothing per
  pass. The same pipeline overlaps the next frame's detection with the current
  frame's pose work, because the detector session and the pool are different
  hardware.

`node scripts/pose-perf-check.mjs` (against `npm run dev`) re-runs the
equivalence check and times each stage on your own machine.

## Modules

| Module                                      | Responsibility                                                                                                                                                                             |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `src/core/video.ts`                         | MP4 demux, codec validation, timestamped bounded decoding                                                                                                                                  |
| `src/core/detection.ts`                     | detector output decoding, NMS, padded crop geometry                                                                                                                                        |
| `src/core/pose.ts`                          | whole-body SimCC/heatmap decoding, mirror mapping, TTA fusion                                                                                                                              |
| `src/core/tracking.ts`                      | stable seat slots, global assignment, ambiguity abstention                                                                                                                                 |
| `src/core/geometry.ts`                      | calibrated 3D transforms, scale utility, camera abstention, shared quantile/median                                                                                                         |
| `src/core/occlusion.ts`                     | expected occlusion (feet hidden by the hull, knees behind it), short-gap recovery, jump rejection, manual refit                                                                            |
| `src/core/knees.ts`                         | knee occlusion windows at the finish: per-seat clear-view calibration, dense-window confirmation, a crossing answered differently frame to frame read as one window, discard with a reason |
| `src/core/strokes.ts`                       | thigh-elevation cycle proxies (hips and knees only), hull-trough bridging, evidence gating, crew synchronization                                                                           |
| `src/core/export.ts`                        | sessions, partial training labels, CSV and PDF                                                                                                                                             |
| `src/core/preprocess.ts`                    | frame → model tensor rasterisation: one bitmap per frame, reused canvas/buffer                                                                                                             |
| `src/core/tta.ts`                           | test-time-augmentation policy and the adaptive pass gate (ignores joints hidden for the session)                                                                                           |
| `src/core/queue.ts`                         | bounded, in-order stage pipeline (detector and pose overlap)                                                                                                                               |
| `src/core/perf.ts`                          | per-stage timing and the run's performance summary                                                                                                                                         |
| `src/core/parallel.ts` / `src/core/pool.ts` | worker-pool + WASM-thread planning; least-busy pose job dispatch with failover                                                                                                             |
| `src/workers/analysis.worker.ts`            | ONNX sessions and offline pipeline orchestration                                                                                                                                           |
| `src/workers/pose.worker.ts`                | one pooled pose-inference session (×N, connected by MessageChannel)                                                                                                                        |

## Privacy and deployment

`google_drive/`, `private/`, `.cache/`, media, and model binaries are ignored. Downloaded sample videos, extracted frames, and athlete imagery are never included in the repository or Pages build. No sample Drive IDs are needed by the published app. Only `dist/` is uploaded by the Pages workflow.

Pages uses GitHub Actions, not a generated `gh-pages` branch. Build/test runs on the working branch `arena/d1e71cca-row-analyzer`. Publication runs after the pull request is merged into `main`, honoring the existing Pages environment policy. Relative asset URLs support `/Row-Analyzer/`. GitHub Pages cannot set the COOP/COEP headers that threaded WASM requires, so `public/coi-serviceworker.js` adds them from a service worker after one reload; the pose-worker pool keeps analysis parallel even where that is unavailable.

## Modeling principles

Bow seat = seat 1; the highest numbered seat is stroke seat. Sweep and sculling are distinct; no symmetric-body assumption is made. Confidence is an **uncalibrated evidence score**, not the probability a point is correct. A 2D shell axis is not camera azimuth. A plausible skeleton is not ground truth. Video panning is not calibrated speed, so split per 500 m is withheld.

## Deployment policy

The existing `github-pages` environment allows only `main`. The workflow therefore builds and tests the session branch, and deploys only the merged main-branch code. No environment protections are disabled or bypassed. The initial API attempt to enable Pages returned 403; subsequent inspection confirmed Pages is configured with GitHub Actions, and normal PR merge is the authorized publication path.

### Live inference preview

While analyzing a video, the workspace automatically shows the latest processed
frame with the tracked skeletons, decoded timestamp, and processed-frame count.
Preview updates are capped at five per second and the transferred image is limited
to 960 pixels wide; every frame is still analyzed at the configured model resolution.
The Skeleton toggle and crew/seat selector also work during processing. Playback
and timeline navigation are disabled until processing finishes or is canceled.
These live poses are unreviewed, before final gap recovery and stroke segmentation;
completed results replace the preview automatically. All preview processing stays
on-device, and preview images are released on replacement, cancellation, or error.
