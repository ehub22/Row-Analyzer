# Validation report — 2026-10-14

## Status: engineering preview, accuracy gate NOT passed

This report distinguishes working software tests from actual athlete-tracking accuracy. No pretrained model inference on these rowing clips, PCK/OKS benchmark, or calibrated 3D evaluation has been completed. Do not describe this release as accurate rowing biomechanics.

## Automated core tests

`npm test`: **28 passing tests** at initial implementation.

- Known 3D joint geometry rotated across eight camera viewpoints, then transformed back with known calibration: normalized angles agree to numerical precision. **This validates transforms, not monocular recovery.**
- Degenerate-angle and end-on scale rejection; azimuth abstention from 2D-only evidence.
- Aspect-correct crop padding, NMS, output-shape rejection.
- Whole-body SimCC sub-pixel peaks, heatmap decoding and 133-joint shape contract.
- Body/feet/face/hands flip map is an involution.
- Synthetic masking test: three-frame wrist gaps are filled from both ends with <1 pixel error on the generated trajectory. Inference confidence <=0.45. This is a small synthetic test, not performance on real occluded limbs.
- Long gaps, ambiguous identity and low-confidence jumps are not bridged.
- Manual anchors and propagated-label provenance are distinct.
- Global assignment tolerates permuted detections and disappearance; ambiguous overlap is withheld rather than relabeled.
- Generated 25 spm cycle timing and crew lags; stationary data produces no strokes; inferred-only cycles are withheld.
- Session shape validation.

## Browser workflow tests

`npm run test:e2e`: **6 passing production-build browser tests**.

- Empty workspace does not invent results.
- Clearly labeled demo, per-seat review, frame stepping, crew timeline.
- Manual correction / export / reset workflow.
- PDF and JSON downloads; imported-session comparison.
- Malformed session error without application crash.
- Narrow desktop/tablet layout has no horizontal page overflow at 600 px.

Additional smoke check: no page errors while loading demo, reviewing crew timing and exporting PDF. Initial build and `npm audit` pass (0 reported vulnerabilities at build time).

## Private Drive sample checks

Private videos were downloaded through the user's authorized Drive connection. They remain ignored, are not in this report as images, and are not in the website.

| File | Content inspected | Metadata | Chrome decoder result |
| --- | --- | --- | --- |
| `1.mov` | On-water eight, visible coxswain, riggers/oars and partially hidden lower limbs | H.264 `avc1.64001f`, 1280×720, 646 samples, 21.535 s | 645 emitted frames; monotonically increasing presentation times, 0–21.501667 s |
| `3.mov` | Erg session with other people in view | H.264 `avc1.640028`, 1920×1080, 264 samples, 8.8 s | 263 emitted frames; monotonically increasing presentation times, 0–8.766667 s |

The sandbox's Chromium decoder omitted the sample at 0.033333 s in each clip. This is a **known decode-validation issue**, not a passed full-frame-coverage test. The pipeline warns on emitted/encoded count mismatch and retains actual timestamps. It does not invent missing frames. Verify native desktop Chrome and investigate H.264 B-frame handling before requiring exact all-frame coverage. Rotated track matrices are currently rejected; arbitrary edit-list synchronization has not been validated.

`1.mov` includes a coxswain, so initializing every detected person as a seat would be wrong. The app stops if person count does not match configured seats. A static detection region can exclude coxswain/bystander centers; it is not a semantic coxswain classifier or motion-compensated shell detector. Visual inspection does not establish seat identity accuracy.

## ONNX runtime integration check

- Generated constant ONNX test doubles with the documented detector and 133-joint SimCC shapes using `scripts/make_test_models.py`.
- Ran a 0.5-second privately derived video through the worker: WebCodecs → OffscreenCanvas preprocessing → WASM ONNX detector and four-pass pose → seat tracker → temporal recovery → results.
- Completed successfully against both the development server and production bundle; stationary test outputs yielded zero complete cycles, not fabricated stroke metrics.
- **Test doubles are not trained models and are not available in the app UI or deployed bundle.** This validates plumbing only.
- WebGPU performance/accuracy with real models remains untested on this machine.

The official model-download hosts (and GitHub LFS media redirects) were outside this environment's permitted outbound hosts. Attempts to retrieve the DWPose pretrained checkpoint failed; a pointer file was not misrepresented as weights. Users can import authentic compatible ONNX weights on their own machine.

## Model download, cache and compatibility update

Implemented: automatic streamed download with progress for the detector and whole-body ONNX models, Cache Storage reuse with exact-size and ONNX-header checks and corrupt-entry eviction, a *Clear cached models* action, local-file fallback at every stage, and shape-level I/O contract checks on the created ONNX sessions.

Drive metadata (checked through the user's connection, not byte-verified here):

| Drive file | Size in bytes | Hash |
| --- | --- | --- |
| `yolox_l.onnx` | 216,746,733 | not recorded |
| `dw-ll_ucoco_384.onnx` | 134,399,116 | not recorded |

**Not validated in this update:**

- The Drive ONNX files were **not** retrieved. The sandbox cannot reach Google Drive directly, and Arena's connector refuses transfers over 100 MiB, so neither file reached the workspace. Their I/O graphs, opset, and dynamic-shape behavior are therefore **unverified**. The 216 MB / 134 MB sizes are consistent with full-precision YOLOX-L and DWPose-l exports but do not prove it.
- Pose tracking was **not** validated with real weights. The only inference run used the generated constant test doubles on a synthetic H.264 clip. This exercised download, cache, fallback, the contract checks and the worker pipeline end to end. It says nothing about detection or joint accuracy.
- No public, CORS-readable download URL has been configured. Until `VITE_YOLOX_URL` and `VITE_DWPOSE_URL` are set, the auto-download path reports *No download URL* and users must import the files.

Automated checks run for this update: 46 unit tests (including the streamed download, truncated/HTML payload rejection, SHA-256 mismatch, abort, cache eviction and the I/O contract cases); the 6 production browser tests; and a browser run that auto-downloaded both test doubles, reloaded with the cache, cleared the cache, recovered from a forced download failure via local files, and completed analysis with no page errors.

## Multithreading update — engineering checks (2026-10-14)

Implemented a parallel WASM pose-worker pool, threaded-WASM planning, a WebGPU warm-up probe with automatic fallback, and a COOP/COEP service-worker shim for Pages. Checks run in the build sandbox (2 CPU cores, headless Chromium, **test doubles only — no real-model speed or accuracy claim**):

- 60 unit tests pass, including 14 new ones: the parallelism planner (pool sizing, core-budget split, no threads without isolation + SharedArrayBuffer, explicit overrides) and the pool dispatcher (least-busy routing, out-of-order completion, tensor-buffer transfer, per-job error isolation, dead-endpoint failover, dispose semantics).
- 8 production-build browser tests pass, including a new one for the parallel-worker settings control and live plan readout.
- End-to-end runs through the real UI with a compute-heavy pose test double (real MatMul work, SimCC contract) on a generated 30-frame H.264 clip completed via: 1 pose worker, 2 pose workers (provider string reports `WASM pool (2 workers × 1 thread)`), TTA on (4 concurrent passes/person), and the automatic fallback path (pool deliberately broken during development → local session completed the run with a warning).
- The sandbox browser reproduces the maintainer's WebGPU failure (`Failed to get GPU adapter`): both models now record the precise reason in session warnings and analysis completes on the WASM pool instead of relying on session-creation fallback alone.
- **Not measured:** wall-clock speedup with real DWPose/YOLOX weights (sandbox has 2 cores and no trained models; pool-vs-single timing there is oversubscription-bound and not representative), threaded-WASM engagement under a genuinely cross-origin-isolated desktop page, Safari/Firefox worker-pool behavior, and long-clip memory with 4 × ~134 MB pose sessions. Verify on a maintainer machine via the `provider` string and per-run timing before quoting any speedup factor.

## Remaining release gates

1. Retrieve licensed reference detector + whole-body weights, verify hashes/contracts and test GPU/WASM agreement on actual pixels.
2. Annotate visible joints, hands, feet and seat IDs across complete strokes of both sweep/scull samples. Record occluded joints as such.
3. Measure per-joint normalized pixel/PCK/OKS errors, identity switches, time-to-reacquire, long-gap behavior, TTA benefit and catch/finish error against observed blade entry/release.
4. Extend the synthetic masking suite to real held-out annotations at increasing gap lengths and crossing/overlap conditions.
5. Implement and validate boat motion compensation, hull/rigger/oar cues, camera intrinsics/azimuth and temporal 3D lifting against calibrated multi-view ground truth.
6. Add observed handle/stretcher/oar constraints, cyclic templates, bone-length checks, bidirectional filtering, angle reliability tests and sequencing diagnoses.
7. Test multiple GPUs, browser memory failure/recovery, VFR clips, rotated/edit-list media and long clips. Current 300 MB / 20,000-frame guards prevent unlimited memory growth but encoded clips are still loaded in memory.

Until these gates pass, normalized angles, layback, twist, sequencing faults, sculling crossover, handle height, blade release, boat set and split per 500 m stay withheld.

## Publication status

Code and GitHub Actions configuration are on the working branch. GitHub CI build, all unit tests and all six browser tests passed. The initial Pages enablement request returned 403; subsequent inspection confirmed Pages is configured for Actions but its environment allows only `main`. The first session-branch deployment was rejected by that protection. The workflow now publishes only main-branch merges, without modifying environment policy. See GitHub Actions for the final deployment result.

## GPU regression fix — engineering checks (2026-10-14)

**The failure the user reported was an application bug, not a device limitation.** The analysis worker pinned `onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.{mjs,wasm}` through `ort.env.wasm.wasmPaths` while importing the `onnxruntime-web/webgpu` entry point. That entry is compiled to load the **asyncify** WASM build, which is the only one that defines the native WebGPU bridge `webgpuInit`; the jsep build does not define it at all. Every WebGPU session therefore failed identically on every device and browser:

```text
WebGPU failed for <model> (no available backend found.
ERR: [webgpu] TypeError: j(...).webgpuInit is not a function); using WASM instead.
```

Reproduced headlessly before the fix (Chromium 153, `--enable-unsafe-webgpu` with SwiftShader, secure-context page, dev server and production bundle, constant ONNX test doubles from `scripts/make_test_models.py` plus a generated H.264 clip): the page reported a working adapter while the exported session recorded `provider: WASM detector / WASM pool (1 worker × 2 threads) pose` with exactly the message above.

After the fix (`src/core/ortRuntime.ts` pins each entry point to its matching WASM build, probes the adapter first, and passes it via `env.webgpu.adapter`):

| Run (`scripts/gpu-analysis-check.mjs`) | Provider | ONNX Runtime assets actually fetched |
| --- | --- | --- |
| Adapter available | `WebGPU (google · swiftshader) detector / WebGPU (google · swiftshader) pose` | asyncify glue + asyncify wasm only |
| `requestAdapter()` → null | `WASM detector / WASM pool (1 worker × 2 threads) pose` | plain glue + plain wasm only |
| Force WASM | `WASM detector / WASM pool (1 worker × 2 threads) pose` | plain glue + plain wasm only |

Both the dev server and the production bundle (`vite preview`) were exercised; the run completed with no page errors in every case, and the notes carry the precise reason when the GPU is not used ("handed out no adapter at all … check chrome://gpu", "a software adapter; GPU inference is optional", "Force WASM compatibility mode is on"). Automated checks for this update: 70 unit tests (10 new in `tests/ortRuntime.test.ts`, including the entry→WASM-module pairing re-derived from the installed package and the adapter-probe branches) and the 8 production browser tests.

Added developer tooling (not shipped in the app bundle): `scripts/lib/chromium.mjs` launches a Chromium with the sandbox's bundled NSS/SwiftShader libraries (used by `smoke.mjs`, `worker-check.mjs`, `decode-check.mjs` and `gpu-analysis-check.mjs`), and `scripts/gpu-analysis-check.mjs` prints the provider, the session warnings, and every ONNX Runtime asset requested — the check that made this bug visible.

**Still not validated:** real pretrained YOLOX/DWPose weights, real hardware GPUs and their drivers (this sandbox only offers a SwiftShader software adapter — accuracy and speed claims on real GPUs remain unmet), Safari and Firefox, GPU memory limits with 216 MB fp32 detector weights, and the mid-run device-loss recovery path (implemented, type-checked, and covered by design review only — no device-loss event was triggered in this environment).
