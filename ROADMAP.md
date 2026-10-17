# Row Analyzer — implementation roadmap

## Product contract

A private, offline desktop-Chrome rowing review workbench. Seat 1 is bow seat; the highest numbered seat is stroke seat. Support sweep and sculling without imposing symmetric body motion. Original footage and manual labels remain on-device; never publish private sample footage with the website. A synthetic demonstration is explicitly labeled, never substituted for an uploaded clip. Unavailable measurements read **withheld**, not zero. Confidence scores are evidence quality indicators, not calibrated probabilities.

## Architecture and technology decisions

- React + TypeScript + Vite: static, inspectable client deployable under a GitHub Pages subpath. No video-upload service, analytics, or API keys.
- WebCodecs + MP4 demuxing: decode presentation timestamps, including variable frame rate. Feed frames with bounded backpressure into a dedicated worker. MOV works only when its container/codec is supported; report decoder failures explicitly rather than silently using imprecise seek sampling.
- ONNX Runtime Web: request WebGPU and probe it with a warm-up inference; fall back to WASM when adapter/graph/operator/GPU support is absent (this failure is common across Chrome/Firefox/Safari and must not abort analysis). OffscreenCanvas performs preprocessing in the worker. WASM pose inference runs in a pool of parallel worker sessions (no cross-origin isolation required), plus threaded WASM per session when the page is cross-origin isolated — on Pages, `public/coi-serviceworker.js` supplies the COOP/COEP headers a static host cannot set. Download/import model weights separately from code. Weights may be cached in Cache Storage after download (never Git or Pages); video and keypoints are never persisted automatically.
- Top-down detector → padded, aspect-correct per-person crops → whole-body pose → seat association → offline temporal completion → stroke segmentation → confidence-gated metrics. Each is a separate testable module.
- Separate camera calibration/boat-reference geometry from 2D pose. Pixel angles must never masquerade as camera-normalized 3D biomechanics.
- SVG/canvas overlays and plots; JSON session/correction interchange, CSV stroke tables, PDF coaching report. Comparison uses timestamped stroke events and compatible metric provenance, not arbitrary pixel coordinates.

## Model evaluation and release gate

Candidates are evaluated by suitability, **not an invented rowing benchmark**:

1. **ViTPose large/huge (whole-body fine-tuned head)**: strong heatmap formulation; useful for spatial uncertainty, flip/multi-scale fusion, and sub-pixel refinement. Large weights/attention activations and browser operator support are substantial risks. A standard COCO-17 checkpoint is insufficient for hands/feet. Provide a heatmap adapter for COCO-WholeBody-133 exports.
2. **RTMW / DWPose whole-body (133 joints), high-resolution large/x variants**: practical browser-first candidate, with hands and feet. Their usual ONNX exports output SimCC distributions, **not 2D heatmaps**. Keep/decode these distributions with sub-pixel refinement and TTA; do not relabel them as heatmaps. Prefer RTMW-x where memory permits; evaluate DWPose as a baseline, not a claim of highest accuracy.
3. **Sapiens-class**: high-resolution human foundation models merit offline GPU evaluation. Memory, export compatibility, input resolution, licensing, and checkpoint keypoint taxonomy preclude assuming universal Chrome support. Best suited to an optional local GPU batch companion.
4. **Person detection**: large YOLO/RTMDet-style ONNX detector with NMS and high-resolution input. The app supports a documented detector contract. Models and licenses must be explicitly selected; no hidden fallback to BlazePose.

Use authentic trained model files, record their names/configuration, and reject incompatible tensor shapes. Local model import keeps browser inference independent of model-host CORS and prevents bundling hundreds of MB into Git. Before claiming accuracy, annotate held-out rowing clips and compare PCK/OKS, hand/foot error, seat ID switches, catch/finish timing error, and angle error against calibrated multi-view ground truth. The first release must visibly identify this gate as unmet if weights or labeled ground truth are unavailable.

## Data model

- Session: schema version, source filename (not file contents), dimensions, duration, actual decoded timestamps, boat class, discipline, seat count, per-seat rigging, best-visible seat, processing settings, model provenance, warnings.
- PoseFrame: presentation time in seconds; seat ID; 133 keypoints in source-image pixels; each keypoint has confidence, origin (`observed`, `inferred`, `manual`, `missing`), optional flags. Optional 3D points require an explicit coordinate-system/calibration provenance.
- BoatCalibration: shell axis, azimuth estimate/confidence, known dimensions, camera intrinsics/extrinsics if available, observability flags. A shell axis in the image alone does not determine azimuth.
- Stroke: seat, catch/finish/next-catch timestamps, rate in strokes per minute, drive-to-recovery ratio, evidence fraction, confidence, metrics with units/reasons for withholding.
- Correction: seat, timestamp, joint index, old/new position, source visibility, neighborhood refit extent. COCO-style training export retains frame identity and visibility; inferred labels excluded from ground truth.

## Occlusion and seat strategy

1. Initialize seats by position along shell axis (bow direction must be verified), with best-visible seat selected. Rower detections are not automatically coxswains or seats: initial selection is reviewable.
2. Associate globally using predicted positions and boat-axis ordering; reject implausible motion/ambiguous assignments instead of exchanging identities. Maintain empty tracks through disappearance. Nearest-neighbor association alone is insufficient for crossing eights.
3. Preserve raw observations. Reject sudden low-confidence jumps. Fill only short, bounded gaps bidirectionally, cap inferred confidence, leave long gaps missing. Do not smooth through a flagged discontinuity.
4. Estimate bone lengths from high-confidence observations; validate rather than force hidden limbs into a plausible fiction. Feet/hands can be constrained only if stretcher/handle landmarks or boat-motion transforms are actually observed. Sweep hands share a handle; sculling hands do not.
5. Future learned per-seat phase template and temporal 3D completion (MotionBERT-style) require rowing adaptation. Never treat template outputs as observations. Use corrections as hard local anchors; propagated changes remain inferred.
6. Coaching metrics dominated by inference are withheld. UI differentiates solid observed, dashed inferred, manual corrected, and missing keypoints.

## Camera normalization and scale

Automatic multi-cue camera estimation is a research milestone, not a solved consequence of 2D keypoints. Hull/rigger lines, oar shafts, hip/shoulder seat alignment and compensated optical flow can propose hypotheses. Bow/stern symmetry, unknown focal length, camera motion, and foreshortening create ambiguities. Fuse independent cues and abstain when underconstrained. A user-verified axis is a useful assist, not an automatic azimuth measurement.

Known shell length, rigger spread and oar length constrain scale only with valid projections/calibration. An eight has no single exact standard length; use a manufacturer/user value (roughly 17–18 m as guidance, not ground truth). Do not derive split per 500 m from panning video. A genuine 3D lifting model and camera solution are prerequisites for normalized trunk, shin, knee, hip, elbow, shoulder, head, and oar angles. Withhold these until then. Synthetic projection tests validate the geometry implementation, not monocular depth recovery.

What exists now is a **user-assisted solve**, not the automatic one above, and the distinction is the design. `src/core/camera.ts` reads the crew's own seat line: seats lie on one straight line at a spacing a rigging tape supplies, so a pinhole camera maps them through `u(x) = (a·x + b)/(c·x + 1)`, and three seats fix it. Its vanishing point with a focal length and a principal point gives the azimuth; `c` gives the depth of the seat line's centre; `a − b·c` the image scale along the shell. The cues that do not exist are declared rather than approximated — the principal point is not in the seat line (`b` is only the image column of the line's own origin), the fit is invariant to the spacing's scale, and roll is not observable from a crew at all, so a level clip is something the user confirms. An azimuth is reported only when the cues agree, and `solveAllowsAngles` keeps the same bar `metricVisibility` set. Automatic multi-cue estimation from hull lines, oars and compensated flow stays a research milestone.

## Milestones and acceptance tests

1. **Foundation**: accessible workbench, roadmap, private upload, source playback, frame navigation, boat/rigging setup, clearly marked synthetic demo. Verify build and UI.
2. **Decode/inference**: WebCodecs worker, bounded frame queue, cancellable progress, detector and whole-body ONNX adapters, GPU/WASM fallback, padded crops, flip + multi-scale TTA and sub-pixel decoding. Test tensor decoding/crop mapping and actual clip decode. Accuracy claims require real weights and labels.
3. **Seat/occlusion**: stable seat slots, gated global assignment, missing tracks, short-gap completion, jump flags, manual correction and local propagation. Test masking and crossing/ambiguity fixtures.
4. **Geometry**: camera confidence and abstention; tested boat-frame transforms and scale utilities. A user-assisted seat-line camera solve and a single-frame 3D lift placed against it are implemented and tested on synthetic viewpoints (see _3D analysis_ below). Full automatic camera solve and temporal 3D lift stay gated until validated. Test many synthetic viewpoints.
5. **Analysis**: cycle segmentation, rate, ratio, evidence gating, crew catch/finish synchronization and phase. Single-seat sweep/scull overlays; withhold unsupported biomechanics and oar metrics. Test synthetic rates, lag, incomplete strokes and missing evidence.
6. **Review/export**: corrections/training labels, comparison, per-stroke tables, CSV/JSON/PDF; verify exports preserve confidence and warnings. Camera-position guidance built in.
7. **Validation/deployment**: private Drive samples kept ignored; record decode and tracking validation separately. CI typecheck/test/build; GitHub Pages Actions deployment from the session branch. Verify workflow outcome and publish URL.
8. **Research completion**: annotated rowing benchmark, detector selection excluding coxswains, hull/handle/oar models, automatic motion-compensated calibration, rowing-trained 3D lift, cyclic/bone/handle/stretcher priors and companion GPU inference. Only then promote inferred biomechanics to validated coaching measurements.

## Risks and mitigations

- Heavy occlusion, spray, reflections, far-side hands: retain missing data; improve camera position; keyframe correction; label visibility explicitly.
- Monocular depth ambiguity: never promise viewpoint-invariant angles without calibration; no 2D cosine “correction.”
- Hardware/memory/operator variation: bounded queues, cancel controls, WASM fallback, local files, honest failure messages; optional local GPU companion later.
- Whole-body output taxonomy and preprocessing vary: explicit input dimensions/format; shape checks; record provenance.
- Unknown frame rotation/edit lists and codec support: validate real clips, reject unsupported decoding, compare presentation against native video before timing claims.
- Model licenses and large binaries: don't redistribute unreviewed weights; document upstream sources and licenses.
- Sample videos are not labeled ground truth: visual inspection is not proof of accuracy; publish validation status, never the private clips.

## Labeling/fine-tuning workflow

Choose representative catch/drive/finish/recovery keyframes across sweep/scull, views and occlusion severity; assign seats; correct visible joints and mark hidden joints missing. Export corrections with video/frame IDs and timestamps. Extract matching frames locally, obtain athlete consent, split by athlete/crew/session (not adjacent frames), fine-tune whole-body models with boat/rigging masks and realistic occlusions. Evaluate held-out hand/foot PCK, seat switches and timing/angle errors. Never train on propagated/inferred points as if human-labeled. Re-import compatible ONNX exports and keep a versioned model/benchmark manifest.

## Implementation checkpoint (2026-10-14)

Milestones 1 and most of 2/3/5/6 have working engineering implementations, with the following explicit qualifications:

- Reference weights are external and not validated; runtime was tested using ONNX test doubles.
- Seat association is global image-space assignment, not a completed motion-compensated boat-frame tracker.
- Occlusion recovery is short bounded interpolation and jump gating, not learned cyclic/3D completion or a full biomechanical optimizer.
- TTA currently fuses refined coordinates, not full aligned heatmaps.
- Stroke events are hip-to-ankle body-motion proxies, not blade-entry/release detections. (Since the hidden-feet update below: thigh-elevation proxies from hips and knees.)
- Comparison is aggregate timing, not phase-normalized 3D or model-stroke matching.
- Milestone 4 implements truthful abstention and known-3D transform utilities; automatic calibration/lifting remains unimplemented.
- Milestone 7 has unit/browser/decode checks and deployment automation; actual tracking-accuracy validation is blocked by unavailable pretrained weights and absent labels. One early H.264 sample was not emitted by the test browser in each real clip; see the validation report.
- Milestone 8 and the optional local GPU companion remain research work, not completed features.

## Multithreading update (2026-10-14)

WebGPU failed on every browser/device combination the maintainer tried, and the previous single-threaded, single-session WASM fallback made real clips impractical (1 detector + up to 32 sequential pose inferences per frame). Implemented since the checkpoint above:

- **Pose-worker pool** (`src/workers/pose.worker.ts`, `src/core/pool.ts`): the main thread spawns N pose workers and MessageChannel-connects them to the coordinator, avoiding nested workers (unreliable in WebKit). Per frame, all person × TTA passes dispatch least-busy-first and fuse in original order; the tracker still consumes frames sequentially. Pool startup or mid-run worker failure degrades to fewer workers or a local session with a recorded warning — never a lost analysis.
- **Parallelism planner** (`src/core/parallel.ts`): pure, unit-tested policy mapping cores × cross-origin-isolation × user setting to pool size and per-session WASM threads. Auto leaves a core for decode/UI and caps the pool at 4 because each worker duplicates the pose model in RAM; the settings dialog shows the live plan for the current machine.
- **WebGPU warm-up probe**: WebGPU sessions must complete a zero-tensor inference before decoding starts, so "creates but cannot run" (unsupported ops, buffer limits, blocklists) falls back to the WASM pool with the reason in the session warnings.
- **Cross-origin isolation on static hosting**: Vite sends COOP/COEP for dev/preview; `public/coi-serviceworker.js` adds the same headers on GitHub Pages after one reload, unlocking threaded WASM where the browser supports it. Without isolation the pool alone provides the speedup.

Not changed: decode order, seat tracking, TTA fusion, metrics, and all accuracy gates. Speed on real weights still needs measuring on a maintainer machine (the build sandbox has 2 cores and only test-double models); per-session settings are recorded in each session's `provider` string for that comparison, and each run now carries its own stage timings (see the speed update below).

## GPU fix update (2026-10-14)

"WebGPU fails no matter which device or browser is used" turned out to be an application bug, not a hardware limitation: the analysis worker pinned `ort-wasm-simd-threaded.jsep.*` in `env.wasm.wasmPaths` while importing the `onnxruntime-web/webgpu` entry point, whose compiled-in WASM variant is `ort-wasm-simd-threaded.asyncify.*`. Only the asyncify/jspi builds define the native WebGPU bridge (`webgpuInit`) that the webgpu entry calls, so **every** WebGPU session creation threw `no available backend found. ERR: [webgpu] TypeError: ...webgpuInit is not a function` and the app fell back to CPU — identical on Windows, macOS, Linux, Chrome, Firefox and Safari.

Fixed since the checkpoint above:

- **`src/core/ortRuntime.ts`**: single place that pairs each ONNX Runtime entry point with the WASM build it was compiled against, plus a WebGPU adapter probe that distinguishes no API / no adapter / software-only adapter / specific error and explains each case in the session notes. A probed adapter is handed to ONNX Runtime via `env.webgpu.adapter`.
- **WASM-only runs now load the small plain build** (~14 MB instead of ~27 MB and no asyncify overhead) because the variant is chosen after the adapter probe; pose workers import `onnxruntime-web/wasm` and never ship GPU code.
- **Mid-run GPU loss is survivable**: a detector or pose session that fails during a run (device lost, driver reset, out-of-memory) is rebuilt on WASM — single-flight for concurrent TTA passes — and the run continues with a recorded warning instead of discarding decoded frames.
- **Runtime visibility**: the workspace shows the provider string after a run (`WebGPU (google · swiftshader) detector / WebGPU pose`, or the WASM pool), and it is recorded in the session, CSV/PDF export and comparison validation.
- **Regression coverage**: `tests/ortRuntime.test.ts` re-derives the entry→WASM-module table from the installed package (webgpu→asyncify, wasm→plain, only the pinned WebGPU build defines `webgpuInit`) and unit-tests the probe branches. `scripts/gpu-analysis-check.mjs` runs the real pipeline headlessly and prints the provider, warnings and the exact ONNX Runtime assets fetched.

Verified in the build sandbox with `scripts/gpu-analysis-check.mjs` against the dev server and the production bundle, using constant ONNX test doubles and a generated H.264 clip (Chromium 153, SwiftShader as the only adapter):

| Run                                    | Result                                                                                                                                                          |
| -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Adapter available                      | `provider: WebGPU (google · swiftshader) detector / WebGPU (google · swiftshader) pose`; only the asyncify glue + wasm fetched; no WebGPU failure warnings      |
| No adapter (`requestAdapter()` → null) | `provider: WASM detector / WASM pool (1 worker × 2 threads) pose`; note explains the missing adapter and points at `chrome://gpu`; only the plain build fetched |
| Force WASM                             | WASM pool; note records that WebGPU was skipped on request                                                                                                      |

Still not validated here: real pretrained weights, real hardware GPUs (the sandbox only has a software adapter), Safari/Firefox, and the mid-run device-loss path (implemented and type-checked, not triggered by a real device loss). See the validation report.

## Pose-stage speed update (2026-10-14)

The pool made pose inference parallel, but profiling the CPU side of the pipeline
in the same sandbox found the _coordinator_ was the bottleneck: every crop was
drawn from the decoded `VideoFrame` and normalised on the analysis worker's
thread, once per person × TTA pass, while every pooled worker waited for work.

Measured in the build sandbox (Chromium 153, SwiftShader, 2 vCPU, constant ONNX
test doubles and a generated 640×360 H.264 clip; `scripts/pose-perf-check.mjs`):

| Step                                                         | Per call |
| ------------------------------------------------------------ | -------- |
| `VideoFrame` crop → 288×384 tensor (previous path, per pass) | ~23 ms   |
| frame `ImageBitmap` crop → same tensor (per pass)            | ~7 ms    |
| letterboxed 640×640 detector tensor                          | ~8.6 ms  |
| SimCC decode, 133 joints × both axes                         | ~0.27 ms |
| `fusePoses` (4 passes)                                       | ~0.06 ms |
| `SeatTracker.update` (8 seats)                               | ~0.68 ms |

Implemented since:

- **`src/core/preprocess.ts`**: one `ImageBitmap` per frame; every crop, TTA pass
  and preview draws from it, with a reused canvas and tensor buffer. The
  rasterisation is asserted **bit-identical** to the previous per-pass
  `VideoFrame` path (`scripts/pose-perf-check.mjs`; crop, flipped crop and the
  letterboxed detector draw) — it changes speed only, never the model input.
  Writing that check caught a real bug during development: the letterbox plan
  initially cropped the source instead of scaling it.
- **Rasterisation inside the pose workers** (`src/workers/pose.worker.ts`,
  `src/core/pool.ts`): the coordinator posts the frame's shared bitmap plus a
  crop box; each worker draws, normalises into its own reused buffer, runs, and
  reports its `preprocess`/`infer`/`decode` milliseconds for the session report.
  Jobs are queued per worker so one session and one buffer are never overlapped.
  Where a browser cannot provide a bitmap, the coordinator rasterises and ships
  tensors exactly as before, and says so in the notes.
- **Adaptive test-time augmentation** (`src/core/tta.ts`, settings default): the
  single pass runs first and the remaining three are dispatched only when a body
  joint (COCO-17 subset of the 133-joint layout) came back below 0.3 or the body
  mean is below 0.5. A rejected crop fuses the identical four passes as
  _Always_; on the test-double clip 2880 of 3840 passes were skipped.
- **Detector/pose overlap** (`src/core/queue.ts`): a bounded, in-order stage
  pipeline keeps one frame's pose work in flight while the next frame is
  detected. It is only used when a pose pool exists — a WebGPU or local-WASM run
  shares one session on one thread and processes each frame in place.
- **Self-measuring runs** (`src/core/perf.ts`): per-stage timings and pass counts
  go into the workspace notes and into the exported session as `performance`, so
  the next optimisation is chosen from evidence on the user's own machine.
- **Fixtures for the checks** (`scripts/make_test_models.py --people N`,
  `scripts/make_test_clip.py`): N-person detector doubles and a synthetic H.264
  clip, so the pool path can be exercised end to end without weights. Never
  validated as tracking: these are constant test doubles.

End-to-end on that fixture (40 frames, eight seats, two pose workers, all three
TTA policies against the pre-change build on the same machine): 4-pass runs
20.9 s → 10.5–12.2 s; the new default (adaptive) 7.8–8.6 s. Runs dominated by the
detector barely move, which the report now makes visible instead of guessable.

Still open, in rough order of expected value on CPU-only machines:

- **Detector cadence (implemented as guarded Auto, not unconditional skipping).**
  The run starts with full detection, then reuses a pose-checked crop for at
  most two decoded frames only if measured detector cost exceeds the overlap
  it gives up. Each valid seat still gets pose inference from every decoded
  frame; weak or ambiguous predictions refresh detection and pose on the same
  frame before tracking commits. Predicted crop positions carry provenance and
  lower identity confidence; Every-frame remains the default baseline until
  real-weight identity and timing checks validate the experimental Auto mode.
  Accuracy and real-model speed on labeled rowing clips remain unvalidated.
- **Real-weight measurements.** Every number above is CPU-side pipeline cost with
  constant test doubles; relative weight of inference still needs a maintainer
  machine with real DWPose/RTMW weights, which the build sandbox cannot fetch.
- **Batched pose.** Stacking the 4 TTA passes of a crop into one inference needs
  a dynamic-batch export; the contract check already ignores the batch
  dimension, so the model side is ready but no export has been validated.

## Hidden-feet update (2026-10-15)

On-water footage almost never shows the feet: they are strapped to the
stretcher below the gunwale for the whole clip. A whole-body model still
returns all 133 joints, so for the feet it returns guesses (a shin
extrapolated down from the knee, the handle, the hull edge), and those guesses
routinely clear the 0.3 "observed" threshold. Three parts of the pipeline
trusted them:

- **Stroke segmentation** read the catch and finish from the hip-to-ankle
  distance and counted the ankles as evidence. Guessed ankles produced
  catches that were not there and missed ones that were; because guesses look
  "observed", wrong rates were reported as valid (in a seeded simulation of a
  guessing model, about 31 spm whatever the crew rowed between 18 and 40). A
  model that drops the ankles instead withheld every stroke.
- **Low-rate doubling, feet or not.** The legs stay flat for a long part of
  each recovery at low rates; keypoint noise on that plateau formed extra
  local extrema, so 20 spm read as ~40 even with perfectly visible feet. PR #7's
  1.2 s merge window only hid this above one particular rate.
- **The adaptive TTA gate** required both ankles, so 64–100% of on-water crops
  paid for all four passes: _Adaptive_ silently cost as much as _Always_.

Implemented:

- **Expected occlusion** (`src/core/occlusion.ts`, `Boat.feetHidden`, setup
  field _Feet in view_, default _Hidden by the hull_): ankle and foot outputs
  (joints 15–22) are replaced by explicit missing joints with a reason before
  tracking, preview and export. Manual points are kept. The session records
  the setting; the viewer legend, the joint counter and the PDF report say
  so. _Visible_ restores tracking for ergometer and overhead footage.
- **Thigh-elevation cycles** (`src/core/strokes.ts`): with the ankle fixed on
  the stretcher and the hip on a horizontal slide, the hip→knee segment rises
  to the catch and falls flat at the finish, so the same events come from
  joints above the gunwale. Catches are peaks of **topographic prominence**
  (immune to plateau noise at any rate), timed at the middle of the band
  within 10% of the peak, because the extreme sample of a broad peak wanders
  with noise. Evidence and withholding use hips and knees only.
- **Hull troughs**: with the legs flat the knee sits near the saxboard and can
  drop out of view for 40–50% of every cycle. A blind spot is bridged instead
  of splitting the cycle when the signal is low on both sides and a hip stays
  visible, because the hull can hide a low knee but never a catch; the finish
  is the middle of the blind spot. A bridged cycle longer than 1.5× the seat's
  median is dropped, since a doubled period means a catch was hidden.
- **TTA gate** (`src/core/tta.ts` `gateJoints`): joints declared hidden are
  left out; the gate's rejection rate on simulated on-water crops returns from
  64% (guessed ankles) or 100% (dropped ankles) to the 7% of crops that are
  genuinely weak.
- **Simulation fixture** (`tests/fixtures/rowingSim.ts`): a seeded two-link
  rower with launch drift, boat surge, keypoint noise, knee dropouts, three
  behaviours for hidden feet (guess, drop, visible) and optional hull
  occlusion of the knees. Results and caveats are in
  [VALIDATION.md](docs/VALIDATION.md); `ROW_SIM_REPORT=1 npx vitest run
tests/feet.test.ts` reprints the table.

Not changed: seat tracking, TTA fusion, decoding, and every accuracy gate.
Sessions saved before this update keep their frames; re-importing one re-runs
the new segmentation on them.

## Knees at the finish (2026-10-16)

The hidden-feet update moved stroke timing onto the knees — and the knees are
covered too. With the legs flat the knee drops to the gunwale (46% of every
cycle in the simulation), and in a side view the loom crosses the legs on the
way out (another 14%). A pose model answers for a covered knee in two ways,
and both were tracked as observations:

- **Pinned to the gunwale.** The knee is reported on the hull edge, the lowest
  line it can lie on, at confidence 0.30–0.60: about 21 px from where the knee
  is, and the trough of the thigh-elevation cycle flattened onto that line.
- **Dragged onto the loom.** The knee is reported on the shaft, about 16 px
  off, lifting the trough by up to 19° of elevation — a third of the stroke's
  range, and enough to move the finish proxy by a tenth of a cycle.

Every stroke event is timed from that trough, so the finish proxy wandered
over a quarter of the cycle (0.33–0.63 of it, against a true legs-flat middle
of 0.415) while catches and rate stayed right. Unlike the feet this cannot be
declared hidden for the session: the same knee is perfectly visible a tenth of
a second earlier, at the catch, and that sample is the one the timing needs.

Implemented:

- **Knee occlusion windows** (`src/core/knees.ts`): the top of a seat's
  thigh-elevation range is the catch, which neither the gunwale nor the loom
  reaches, so those frames calibrate what a visible knee looks like on this
  footage — its confidence and its apparent thigh length. Below that band a
  frame is a candidate when its knees, pooled over both legs, look weaker than
  the seat's own clear view; candidates are believed only where they are dense
  (a majority of a ±0.08 s neighbourhood, sustained for at least 0.15 s), each
  dense window is confirmed as a whole and trimmed back to the frames that
  still look occluded. Confirmed knees become missing points with a reason:
  not tracked, not gap-filled, not drawn, not counted as evidence. Manual
  points are kept, and a seat that cannot be calibrated is left alone.
- **A model that changes its mind** (`src/core/knees.ts`): the ordinary case is
  neither a consistent guess nor a consistent dropout but a coin flip between
  them on every frame, and neither answer carries enough evidence on its own —
  a dropout has nothing to judge, and a guess between dropouts is one frame. So
  a frame whose knees are both gone takes its place in the cycle from its
  neighbours and cannot split the crossing it belongs to; a window is confirmed
  only where the model answered at least a quarter of its frames, which keeps
  an honest dropout from being read as a guess; and where the confirmed windows
  cover at least half the time the knee is low — a gunwale, which the knee sits
  behind for half of every stroke, rather than a loom crossing a seventh of it —
  the weak frames between the windows are discarded too, down to the trough
  line and never above it.
- **Expected occlusion vocabulary** (`src/core/occlusion.ts`): the knee flag
  joins `HIDDEN_FEET_FLAG` and `expectedOcclusion()` names the absences the
  setup or the footage explains. The cycle evidence gate (`src/core/strokes.ts`)
  and the joint counter (`src/App.tsx`) read it, so a knee the hull was in
  front of is no longer counted as evidence the model failed to find — without
  that, discarding the guesses would have withheld the very rates the visible
  catch peaks support.
- **Hull troughs, generalized** (`src/core/strokes.ts`): a blind spot counts as
  a hull trough when either end lies inside the trough _band_, not only when it
  is the very lowest sample, because the sample a blind spot ends on is
  smoothed towards the flank beside it; and each end is judged on its own
  elevation as well as on the smoothed one, because the descent after a catch
  runs at about 100°/s and smoothing ±0.1 s inflates a blind-spot edge by
  10–15°. Without this the discarded windows split cycles at high rates.
- **Simulation fixture** (`tests/fixtures/rowingSim.ts`): the loom now hides
  the knee for a set fraction of the cycle centred on the finish, reported
  either dropped or dragged along the shaft within bounded offsets — or a
  per-frame coin flip between the two — and the ground truth says which frames
  each occluder covered. `tests/knees.test.ts`
  pins the behaviour; `ROW_SIM_REPORT=1 npx vitest run tests/knees.test.ts`
  reprints the before/after table (summarized in
  [VALIDATION.md](docs/VALIDATION.md)).
- **Demonstration crew** (`src/core/demo.ts`): its procedural knee used to
  slide along a straight line, so the thigh grew by a quarter of its length
  between the catch and the finish — precisely what the occlusion pass looks
  for. The demo leg is now a rigid two-link chain solved from hip and ankle;
  the pass judges all eight seats and discards nothing.

Measured on the seeded simulation (30 s clips, three seeds, 20 and 28 spm, eight
occlusion scenarios): no covered knee is tracked as an observation any more
(was 100% of them at the gunwale and on the loom, 52% and 49% where the model
alternates between the two answers, 16–21 px from the truth; now 0% for the
gunwale however it is answered, and 1–4% at the edges of a loom window), the
finish proxy returns to 0.34–0.59 of the cycle from 0.29–0.64 — and to
0.43–0.49 in every scenario but the alternating loom at 28 spm — clear knees
are discarded in 0–3.2% of their frames, and every cycle is still found at
every rate. Footage where the model _drops_ the covered knee instead
of guessing it is byte-identical to before: there is nothing to discard, and
the bridged trough already carried the finish.

Not changed: the catch end of the stroke, seat tracking, decoding, TTA fusion
and every accuracy gate. The adaptive TTA gate still requires both knees: a
covered knee is either guessed at an ordinary confidence, which the gate
already accepts, or dropped, and a dropped knee needs no second opinion —
relaxing the gate for knees the hull covers was measured to save no passes at
all, so it was not added.

Still open, in rough order of expected value:

- **Validate on real footage.** Everything above is measured on a simulator
  whose assumptions about model failure may not match a real export. Label
  catches and finishes on a few on-water clips (the label export already
  supports it) and compare.
- **Stretcher-anchored leg angles.** The ankle could be completed from the
  knee arc (the knee moves on a circle around the fixed ankle), but that is
  ill-conditioned with the shin near vertical, and boat surge moves the
  stretcher in the image. It needs real footage to validate, so knee and
  ankle angles stay withheld.
- **Fine-tune with occluded feet labelled missing**, so the model stops
  guessing them (see _Labeling/fine-tuning workflow_ above) — and with knees
  behind the gunwale or the loom labelled missing too, which would make the
  occlusion windows unnecessary rather than merely handled.
- **A knee latched onto the handle near the catch** is not detected: the
  occlusion pass judges only the low part of the cycle, because the catch end
  must stay untouched. Catching it needs a length reference that does not
  depend on a per-frame detection box, and boxes merge when rowers overlap.
- **Short loom windows** (under 0.5 s) are discarded but not bridged, so their
  finish comes from the low samples that survive on either side of the shaft:
  0.38–0.48 of the cycle measured against a true 0.415.
- **An alternating loom at 28 spm** is the weakest row measured, 0.34–0.59
  against the same true 0.415: the dropouts inside a crossing leave holes short
  enough for the existing short-gap recovery to interpolate across, and those
  `inferred` knees then enter the cycle signal a few pixels off. Excluding
  interpolated joints from that signal was measured and cost a cycle in thirty,
  so they stay in and the residual is documented instead.
- **Tracker re-acquisition** using the seat-order constraint: a seat is
  dropped after 31 frames (~1 s) without a match and never comes back.
- **Far-side eye and ear** are permanently weak in a side view and still count
  in the TTA gate; a **fused joint** is labelled observed whenever one pass
  cleared 0.3, even when its fused confidence is far below that (stroke timing
  checks confidence, so it is unaffected; the evidence fraction is not).

## 3D analysis (2026-10-17)

Normalized angles were the gated half of milestone 4 and the lift adapter of
milestone 8. Both are implemented now, in the only form the evidence supports:
a camera solve the user assists and can audit, a single-frame lift that is
checked against the keypoints it was fitted to, and angles that are either in
the boat's frame or withheld with a reason. Nothing here claims that monocular
depth has been recovered.

Implemented:

- **A true-3D fixture** (`tests/fixtures/rowingSim3d.ts`): the existing 2D
  kinematics, unchanged and imported rather than copied, in boat-frame metres —
  x along the shell towards the rowers' feet, y up, z to a rower's right — with
  a crew seated along it, hip and shoulder widths, boat surge, keypoint noise
  scaled to the crop, hidden feet and a pinhole camera with a focal length, an
  azimuth, a distance, a height and a principal point. Per-frame truth carries
  boat-frame joints, projected pixels and the hip-midpoint depth. An
  `azimuthDrift` option turns the camera during the clip, because a launch
  follows the shell.
- **The seat-line solve** (`src/core/camera.ts`, `tests/camera3d.test.ts`): see
  _Camera normalization and scale_ above for the geometry and what it cannot
  give. It abstains on a square-on view (no perspective in the spacing at all),
  on one so nearly square that the turn is inside the keypoint noise, without a
  spacing or a lens, without the level confirmation, on a mirrored clip, on a
  bow direction that contradicts the hips, on a seat line that does not fit one
  equally spaced straight line, and on a clip whose two halves solve to
  different azimuths. Recovered azimuths are within 0.15° of the truth at 10,
  18, 30 and 45° off square.
- **Normalized angles** (`src/core/angles.ts`, `tests/angles3d.test.ts`):
  interior knee, hip and elbow angles, thigh and shin angles from the boat's
  vertical, and layback averaged over both shoulders so a sweep rower's twist is
  not read as a lean. An angle whose joints were not all lifted is withheld
  naming the joint. The thigh is reported alongside the knee because on the
  water it is the one leg angle that is fully observed: feet are below the
  gunwale, so the knee and shin angles do not exist there.
- **The lift adapter** (`src/core/lift.ts`, `tests/lift3d.test.ts`): places a
  learned lift against the solve by fitting the scale and the in-plane position
  that make the lifted pose land on the observed keypoints through the
  calibrated camera, with the depth taken from the calibration rather than
  invented. It refuses a pose that does not land (naming the reprojection in
  pixels), a scale that implies an impossible segment length (naming it in
  centimetres), an output whose shape it does not recognize (naming the shape),
  and too few observed joints to place anything. A discarded keypoint stays
  discarded: the lift never resurrects a knee the gunwale hid.
- **Plumbing**: `Session.camera` is the solve, `Session.calibration` what the
  user measured, `Session.angles` one row per frame aligned to `frames`, and
  `Session.lift` what produced them or why nothing did. The lift runs as a
  second decode pass, only when a solve and imported weights both exist. The
  setup panel asks for the spacing, the lens and the two confirmations; the
  camera panel shows the cues or the abstention; exports carry the angles, with
  `withheld` written out rather than left blank.

Measured, over every lifted frame of one seat (8 s clips, 24 spm, eight seats,
1800 px lens, 16 m out):

| Camera off square | Knee error, measured in the image | Knee error, lifted into the boat frame | Layback error, image | Layback, lifted |
| ----------------- | --------------------------------- | -------------------------------------- | -------------------- | --------------- |
| 10°               | 3.8° mean, 11.8° worst            | 0.0°                                   | 5.7° worst           | 0.02° worst     |
| 24°               | 4.4° mean, 16.7° worst            | 0.0°                                   | 8.3° worst           | 0.0° worst      |
| 45°               | 8.6° mean, 29.4° worst            | 0.0°                                   | 14.5° worst          | 0.02° worst     |

The lifted column is exact because the test's oracle _is_ the fixture's own 3D,
so this validates the geometry implementation — projection, solve, placement,
rotation into the boat frame, angle definitions — and says nothing about a
lift model's accuracy on real footage. The image column is the reason to do it
at all: the same knee, read off the pixels, is wrong by degrees at every
azimuth a launch actually films from, and worse the further the camera is from
the beam. No 2D cosine correction recovers that, which is what the risk
register says.

Not done, and gated:

- **No lift weights exist.** Nothing is bundled and nothing downloads: the
  adapter's contract is written (17-joint COCO body, camera-frame metres, root
  relative, x right, y up, z away from the camera) and a model can be imported
  locally, but no rowing-validated 3D lift has been evaluated, so normalized
  angles stay withheld until one is. Milestone 8's benchmark is still the gate.
- **The depth axis's sign is a contract, not a measurement.** A rower is nearly
  planar and the plane is nearly the image plane, so flipping a lift's z
  reprojects almost as well (6 px against a 2 px honest fit) and the interior
  angles are unchanged because a reflection preserves them. What it would
  corrupt is anything lateral — boat set, shoulder level, trunk twist — which is
  exactly why those stay withheld. The convention must be checked once against a
  known-good release, not per frame.
- **One azimuth per clip.** The solve reports a single camera and prices the
  assumption with a stability cue over the two halves; a launch that pans or
  surges is not described by one azimuth, and a clip whose halves disagree more
  than 6° is refused. Motion-compensated calibration is milestone 8.
- **Pitch, roll and camera height are not solved.** Roll is asked for. Pitch and
  height would need a vertical reference the crew does not provide, so anything
  absolute in height — handle height above the water — stays withheld.
- **Metric lengths stay withheld.** A wrong tape scales the distance and the
  pixels-per-metre by the same factor and leaves the azimuth untouched, which is
  why angles can be reported and lengths cannot. `pxPerMetre` is directional: it
  is the scale along the shell axis, and a metre across the boat spans
  `f/depth` pixels instead.
- **The lift is single-frame.** No temporal smoothing, no bone-length priors
  across a clip, no cyclic prior. A per-frame lift placed per-frame is what has
  been tested; the temporal model is milestone 8.
- **Nothing was measured on real footage.** Every number above comes from the
  simulator, including the occlusion and noise it was told to produce.
