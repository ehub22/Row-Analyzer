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

## Milestones and acceptance tests
1. **Foundation**: accessible workbench, roadmap, private upload, source playback, frame navigation, boat/rigging setup, clearly marked synthetic demo. Verify build and UI.
2. **Decode/inference**: WebCodecs worker, bounded frame queue, cancellable progress, detector and whole-body ONNX adapters, GPU/WASM fallback, padded crops, flip + multi-scale TTA and sub-pixel decoding. Test tensor decoding/crop mapping and actual clip decode. Accuracy claims require real weights and labels.
3. **Seat/occlusion**: stable seat slots, gated global assignment, missing tracks, short-gap completion, jump flags, manual correction and local propagation. Test masking and crossing/ambiguity fixtures.
4. **Geometry**: camera confidence and abstention; tested boat-frame transforms and scale utilities. Full automatic camera solve and temporal 3D lift stay gated until validated. Test many synthetic viewpoints.
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
- Stroke events are hip-to-ankle body-motion proxies, not blade-entry/release detections.
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

Not changed: decode order, seat tracking, TTA fusion, metrics, and all accuracy gates. Speed on real weights still needs measuring on a maintainer machine (the build sandbox has 2 cores and only test-double models); per-session settings are recorded in each session's `provider` string for that comparison.

## GPU fix update (2026-10-14)

"WebGPU fails no matter which device or browser is used" turned out to be an application bug, not a hardware limitation: the analysis worker pinned `ort-wasm-simd-threaded.jsep.*` in `env.wasm.wasmPaths` while importing the `onnxruntime-web/webgpu` entry point, whose compiled-in WASM variant is `ort-wasm-simd-threaded.asyncify.*`. Only the asyncify/jspi builds define the native WebGPU bridge (`webgpuInit`) that the webgpu entry calls, so **every** WebGPU session creation threw `no available backend found. ERR: [webgpu] TypeError: ...webgpuInit is not a function` and the app fell back to CPU — identical on Windows, macOS, Linux, Chrome, Firefox and Safari.

Fixed since the checkpoint above:

- **`src/core/ortRuntime.ts`**: single place that pairs each ONNX Runtime entry point with the WASM build it was compiled against, plus a WebGPU adapter probe that distinguishes no API / no adapter / software-only adapter / specific error and explains each case in the session notes. A probed adapter is handed to ONNX Runtime via `env.webgpu.adapter`.
- **WASM-only runs now load the small plain build** (~14 MB instead of ~27 MB and no asyncify overhead) because the variant is chosen after the adapter probe; pose workers import `onnxruntime-web/wasm` and never ship GPU code.
- **Mid-run GPU loss is survivable**: a detector or pose session that fails during a run (device lost, driver reset, out-of-memory) is rebuilt on WASM — single-flight for concurrent TTA passes — and the run continues with a recorded warning instead of discarding decoded frames.
- **Runtime visibility**: the workspace shows the provider string after a run (`WebGPU (google · swiftshader) detector / WebGPU pose`, or the WASM pool), and it is recorded in the session, CSV/PDF export and comparison validation.
- **Regression coverage**: `tests/ortRuntime.test.ts` re-derives the entry→WASM-module table from the installed package (webgpu→asyncify, wasm→plain, only the pinned WebGPU build defines `webgpuInit`) and unit-tests the probe branches. `scripts/gpu-analysis-check.mjs` runs the real pipeline headlessly and prints the provider, warnings and the exact ONNX Runtime assets fetched.

Verified in the build sandbox with `scripts/gpu-analysis-check.mjs` against the dev server and the production bundle, using constant ONNX test doubles and a generated H.264 clip (Chromium 153, SwiftShader as the only adapter):

| Run | Result |
| --- | --- |
| Adapter available | `provider: WebGPU (google · swiftshader) detector / WebGPU (google · swiftshader) pose`; only the asyncify glue + wasm fetched; no WebGPU failure warnings |
| No adapter (`requestAdapter()` → null) | `provider: WASM detector / WASM pool (1 worker × 2 threads) pose`; note explains the missing adapter and points at `chrome://gpu`; only the plain build fetched |
| Force WASM | WASM pool; note records that WebGPU was skipped on request |

Still not validated here: real pretrained weights, real hardware GPUs (the sandbox only has a software adapter), Safari/Firefox, and the mid-run device-loss path (implemented and type-checked, not triggered by a real device loss). See the validation report.
