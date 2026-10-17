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

| File    | Content inspected                                                               | Metadata                                             | Chrome decoder result                                                          |
| ------- | ------------------------------------------------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------------ |
| `1.mov` | On-water eight, visible coxswain, riggers/oars and partially hidden lower limbs | H.264 `avc1.64001f`, 1280×720, 646 samples, 21.535 s | 645 emitted frames; monotonically increasing presentation times, 0–21.501667 s |
| `3.mov` | Erg session with other people in view                                           | H.264 `avc1.640028`, 1920×1080, 264 samples, 8.8 s   | 263 emitted frames; monotonically increasing presentation times, 0–8.766667 s  |

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

Implemented: automatic streamed download with progress for the detector and whole-body ONNX models, Cache Storage reuse with exact-size and ONNX-header checks and corrupt-entry eviction, a _Clear cached models_ action, local-file fallback at every stage, and shape-level I/O contract checks on the created ONNX sessions.

Drive metadata (checked through the user's connection, not byte-verified here):

| Drive file             | Size in bytes | Hash         |
| ---------------------- | ------------- | ------------ |
| `yolox_l.onnx`         | 216,746,733   | not recorded |
| `dw-ll_ucoco_384.onnx` | 134,399,116   | not recorded |

**Not validated in this update:**

- The Drive ONNX files were **not** retrieved. The sandbox cannot reach Google Drive directly, and Arena's connector refuses transfers over 100 MiB, so neither file reached the workspace. Their I/O graphs, opset, and dynamic-shape behavior are therefore **unverified**. The 216 MB / 134 MB sizes are consistent with full-precision YOLOX-L and DWPose-l exports but do not prove it.
- Pose tracking was **not** validated with real weights. The only inference run used the generated constant test doubles on a synthetic H.264 clip. This exercised download, cache, fallback, the contract checks and the worker pipeline end to end. It says nothing about detection or joint accuracy.
- No public, CORS-readable download URL has been configured. Until `VITE_YOLOX_URL` and `VITE_DWPOSE_URL` are set, the auto-download path reports _No download URL_ and users must import the files.

Automated checks run for this update: 46 unit tests (including the streamed download, truncated/HTML payload rejection, SHA-256 mismatch, abort, cache eviction and the I/O contract cases); the 6 production browser tests; and a browser run that auto-downloaded both test doubles, reloaded with the cache, cleared the cache, recovered from a forced download failure via local files, and completed analysis with no page errors.

## Multithreading update — engineering checks (2026-10-14)

Implemented a parallel WASM pose-worker pool, threaded-WASM planning, a WebGPU warm-up probe with automatic fallback, and a COOP/COEP service-worker shim for Pages. Checks run in the build sandbox (2 CPU cores, headless Chromium, **test doubles only — no real-model speed or accuracy claim**):

- 60 unit tests pass, including 14 new ones: the parallelism planner (pool sizing, core-budget split, no threads without isolation + SharedArrayBuffer, explicit overrides) and the pool dispatcher (least-busy routing, out-of-order completion, tensor-buffer transfer, per-job error isolation, dead-endpoint failover, dispose semantics).
- 8 production-build browser tests pass, including a new one for the parallel-worker settings control and live plan readout.
- End-to-end runs through the real UI with a compute-heavy pose test double (real MatMul work, SimCC contract) on a generated 30-frame H.264 clip completed via: 1 pose worker, 2 pose workers (provider string reports `WASM pool (2 workers × 1 thread)`), TTA on (4 concurrent passes/person), and the automatic fallback path (pool deliberately broken during development → local session completed the run with a warning).
- The sandbox browser reproduces the maintainer's WebGPU failure (`Failed to get GPU adapter`): both models now record the precise reason in session warnings and analysis completes on the WASM pool instead of relying on session-creation fallback alone.
- **Not measured:** wall-clock speedup with real DWPose/YOLOX weights (sandbox has 2 cores and no trained models; pool-vs-single timing there is oversubscription-bound and not representative), threaded-WASM engagement under a genuinely cross-origin-isolated desktop page, Safari/Firefox worker-pool behavior, and long-clip memory with 4 × ~134 MB pose sessions. Verify on a maintainer machine via the `provider` string and per-run timing before quoting any speedup factor.

## Hidden-feet update — simulation checks (2026-10-15)

On-water side views hide the feet (strapped to the stretcher below the gunwale) for the whole clip, and whole-body models answer for them anyway. This update discards those answers, times stroke cycles from hips and knees, and leaves hidden joints out of the adaptive TTA gate (see the roadmap's _Hidden-feet update_).

**Everything below comes from a synthetic simulator, not from footage or trained weights.** `tests/fixtures/rowingSim.ts` generates what a pose model plausibly returns for a rower filmed from a launch:

- a two-link leg with the ankle fixed on the stretcher and the hip on a horizontal slide; knee angle 50° at the catch, 175° with the legs flat (28–55% of the cycle); trunk swing and arm draw in sequence;
- 300 px/m, keypoint noise σ = 1.2% of the box height, 0.3 m launch drift, 0.08 m stroke-locked boat surge, 3% per-frame knee dropouts;
- three behaviours for the hidden feet: **guessed** (a drifting shin extrapolated from the knee that sometimes latches onto the handle; confidence 0.1–0.62, so ~60% of guesses read as "observed"), **dropped** (confidence 0.02–0.32) and **visible** (erg footage);
- optionally, **knees behind the hull**: a saxboard at hip height hides the knee whenever the legs are nearly flat (46% of every cycle), and the model drops it every time.

The simulator encodes assumptions about how a model fails, so it pins the pipeline's behaviour on those failures; it does not measure accuracy on real rowing. Each row: 5 seeds × 40 s at 30 fps. "Catch" is maximum leg compression in the simulation, not blade entry. Reproduce the new-pipeline table with `ROW_SIM_REPORT=1 npx vitest run tests/feet.test.ts`; the old rows ran `git show 1ec983a:src/core/strokes.ts` on the same simulator.

Before (hip-to-ankle segmentation, ankles as evidence):

| Feet in the model output | True rate (spm)        | Cycles in clip           | Strokes reported              | Mean reported rate               | Worst stroke error |
| ------------------------ | ---------------------- | ------------------------ | ----------------------------- | -------------------------------- | ------------------ |
| visible                  | 18 / 20                | 55 / 65                  | 114 / 127                     | 36.6 / 40.1                      | 32 / 29 spm        |
| visible                  | 26 / 32 / 40           | 85 / 105 / 130           | 85 / 105 / 130                | 26.0 / 32.0 / 40.0               | ≤ 0.9 spm          |
| guessed                  | 18 / 20 / 26 / 32 / 40 | 55 / 65 / 85 / 105 / 130 | 82 / 84 / 84 / 87 / 88        | 32.0 / 31.1 / 30.7 / 31.1 / 32.4 | 18–31 spm          |
| dropped                  | 18–40                  | 55–130                   | 0 (17–24 found, all withheld) | —                                | —                  |

Guessed ankles produced a rate of about 31 spm whatever the crew rowed, reported as valid. Visible feet doubled the rate at ≤ 20 spm (plateau noise on the legs-flat phase; the 1.2 s merge window of PR #7 only hid it at higher rates).

After (thigh elevation, hips and knees only). Guessed, dropped and visible feet give **identical** results, because the segmentation never reads them:

| True rate (spm) | Cycles in clip | Cycles found | Mean rate | Worst stroke error | Catch timing (mean, worst) | Finish phase |
| --------------- | -------------- | ------------ | --------- | ------------------ | -------------------------- | ------------ |
| 18              | 55             | 50           | 18.00     | 0.37 spm           | −99 ms, 133 ms             | 0.44–0.48    |
| 20              | 65             | 60           | 19.99     | 0.45 spm           | −93 ms, 133 ms             | 0.44–0.48    |
| 26              | 85             | 85           | 25.99     | 0.65 spm           | −71 ms, 108 ms             | 0.44–0.48    |
| 32              | 105            | 100          | 32.01     | 1.33 spm           | −57 ms, 100 ms             | 0.44–0.48    |
| 40              | 130            | 130          | 40.02     | 0.91 spm           | −48 ms, 67 ms              | 0.42–0.49    |

With the knees also behind the hull while the legs are flat: 50 / 61 / 85 / 102 / 130 cycles found at the same rates, mean rate within 0.02 spm, worst stroke 0.37–1.33 spm, catch timing −43 to −87 ms (worst 133 ms), finish phase 0.44–0.49. Without the hull-trough bridge this scenario found **no cycles at all**.

- Every cycle found is within 0.25 s of a true catch; none is spurious. The cycles not found are those whose catch band is cut off by the start or end of the clip.
- The catch proxy is early by a consistent 40–100 ms (the middle of a broad peak, as designed). Stroke rate is unaffected, and crew timing compares seats with the same bias.
- TTA gate, share of crops sent to four passes: 7% with the gate that skips hidden joints, for every feet behaviour; the previous gate sent 7% (visible), 64% (guessed) and 100% (dropped). With the knees behind the hull it is about 50% (old gate 81%), because a knee hidden in a frame is a genuinely weak body joint.
- Regression tests (`tests/feet.test.ts`) pin these behaviours: rate and timing at 18–40 spm, no doubling at low rates, invariance to whatever the feet joints contain, no cycle across a blind spot, no bridge across a blind spot that could hide a catch, withholding when hips and knees are only inferred, and the session field, demo and import validation.

Automated checks run for this update: 125 unit tests (plus the opt-in simulation report) and the 8 production browser tests.

**Not validated:** behaviour on real footage with real weights; whether a real model guesses or drops hidden feet (both are handled); the thigh-elevation catch and finish against observed blade entry and release; knee visibility in a given boat and camera position.

## Knees at the finish — simulation checks (2026-10-16)

With the legs flat the knee drops to the gunwale, and in a side view the loom crosses the legs on the way out. A pose model still answers: pinned to the hull edge, or dragged onto the shaft, at a confidence that clears the observed threshold. Those answers were tracked as observations and lifted the trough every stroke event is timed from (see the roadmap's _Knees at the finish_).

**Everything below comes from the same synthetic simulator, not from footage or trained weights.** For this update `tests/fixtures/rowingSim.ts` gained:

- **the loom**: the knee is covered for a set fraction of the cycle (14%) centred on the finish, and the model either **drops** it or **drags** it along the shaft — bounded per-joint offsets of 2–9% of the box height across the shaft and ±5% along it, at confidence 0.35–0.78, so a dragged knee reads as an ordinary observation;
- **the gunwale**, already there, now also answered as a **snap**: the knee is placed on the hull edge (hip height) at confidence 0.30–0.60, instead of only being dropped;
- **per-frame ground truth** for which occluder covered the knee, so a discarded sample can be checked against what the camera could actually see;
- **an alternating answer**: either occluder can also be answered by a fresh per-frame coin flip between the guess and the dropout, which is what a model actually does — it does not treat the same covered knee the same way twice.

Each row: 3 seeds × 30 s at 30 fps, one seat, 20 and 28 spm. "Cycles" counts the cycles whose catches are far enough from the clip edges to be timed; "found" counts every reported stroke, including the ones cut off by an edge (each of those is still within 0.15 s of a true catch). "Finish phase" is where in its cycle the finish proxy landed — the simulated legs are flat from 28% to 55%, so the middle is 0.415. "Covered knee tracked" is the share of frames the fixture says were covered in which a knee was still read as an observation, with its mean distance from the true knee. "Clear knees discarded" is the pass's false-positive rate on frames the fixture says were visible. Reproduce with `ROW_SIM_REPORT=1 npx vitest run tests/knees.test.ts`.

The _before_ rows run the same clips through the pipeline with the knee pass disabled. The segmentation changes that came with this update did not move them: they match the numbers measured before those changes.

| Pass   | Scenario            | spm     | Cycles / found / matched    | Max rate error  | Finish phase              | Covered knee tracked (px off) | Clear knees discarded | Windows |
| ------ | ------------------- | ------- | --------------------------- | --------------- | ------------------------- | ----------------------------- | --------------------- | ------- |
| before | clear               | 20 / 28 | 24 / 24 / 24 · 36 / 39 / 36 | 0.22 / 0.57 spm | 0.44–0.47 · 0.44–0.48     | —                             | —                     | 0       |
| before | gunwale snap        | 20 / 28 | 24 / 25 / 24 · 36 / 39 / 36 | 0.43 / 0.57 spm | **0.33–0.56 · 0.44–0.63** | **100% (21.1) · 100% (21.4)** | —                     | 0       |
| before | gunwale alternating | 20 / 28 | 24 / 24 / 24 · 36 / 39 / 36 | 0.43 / 0.57 spm | **0.29–0.64 · 0.29–0.54** | **52% (20.9) · 52% (20.9)**   | —                     | 0       |
| before | gunwale dropout     | 20 / 28 | 24 / 25 / 24 · 36 / 39 / 36 | 0.45 / 0.73 spm | 0.46–0.48 · 0.45–0.48     | —                             | —                     | 0       |
| before | loom drag           | 20 / 28 | 24 / 24 / 24 · 36 / 39 / 36 | 0.22 / 0.57 spm | **0.34–0.59 · 0.34–0.59** | **100% (15.8) · 100% (15.7)** | —                     | 0       |
| before | loom alternating    | 20 / 28 | 24 / 24 / 24 · 36 / 39 / 36 | 0.22 / 0.57 spm | **0.34–0.60 · 0.34–0.61** | **49% (15.2) · 49% (15.2)**   | —                     | 0       |
| before | loom dropout        | 20 / 28 | 24 / 24 / 24 · 36 / 39 / 36 | 0.22 / 0.57 spm | 0.40–0.41 · 0.38–0.42     | —                             | —                     | 0       |
| before | gunwale + loom      | 20 / 28 | 24 / 24 / 24 · 36 / 39 / 36 | 0.22 / 0.57 spm | **0.46–0.49 · 0.43–0.53** | **100% (18.8) · 100% (18.9)** | —                     | 0       |
| fixed  | clear               | 20 / 28 | 24 / 24 / 24 · 36 / 39 / 36 | 0.22 / 0.57 spm | 0.44–0.47 · 0.44–0.48     | —                             | 0.00% / 0.19%         | 0 / 1   |
| fixed  | gunwale snap        | 20 / 28 | 24 / 25 / 24 · 36 / 39 / 36 | 0.45 / 0.73 spm | **0.44–0.48 · 0.43–0.48** | **0% · 0%**                   | 2.26% / 3.18%         | 30 / 42 |
| fixed  | gunwale alternating | 20 / 28 | 24 / 25 / 24 · 36 / 39 / 36 | 0.45 / 0.73 spm | **0.44–0.48 · 0.43–0.48** | **0% · 0%**                   | 2.26% / 2.97%         | 30 / 42 |
| fixed  | gunwale dropout     | 20 / 28 | 24 / 25 / 24 · 36 / 39 / 36 | 0.45 / 0.73 spm | 0.46–0.48 · 0.45–0.48     | —                             | 0.00% / 0.00%         | 0 / 0   |
| fixed  | loom drag           | 20 / 28 | 24 / 24 / 24 · 36 / 39 / 36 | 0.22 / 0.57 spm | **0.40–0.48 · 0.38–0.48** | **2% (17.8) · 3% (17.7)**     | 0.46% / 1.24%         | 30 / 43 |
| fixed  | loom alternating    | 20 / 28 | 24 / 24 / 24 · 36 / 39 / 36 | 0.22 / 0.57 spm | **0.39–0.43 · 0.34–0.59** | **1% (19.8) · 4% (17.1)**     | 0.42% / 1.31%         | 29 / 39 |
| fixed  | loom dropout        | 20 / 28 | 24 / 24 / 24 · 36 / 39 / 36 | 0.22 / 0.57 spm | 0.40–0.41 · 0.38–0.42     | —                             | 0.00% / 0.22%         | 0 / 1   |
| fixed  | gunwale + loom      | 20 / 28 | 24 / 25 / 24 · 36 / 39 / 36 | 0.45 / 0.73 spm | **0.44–0.48 · 0.43–0.49** | **0% · 0%**                   | 1.76% / 2.65%         | 30 / 42 |

- Every timeable cycle is found in every scenario and rate, before and after; no reported stroke is spurious. Rate and catch timing are unchanged by the pass (max error ≤ 0.73 spm throughout).
- The finish proxy is what the pass fixes: a guessed knee moved it over a quarter of the cycle (0.29–0.64), and after the pass it stays inside 0.34–0.59 against a true middle of 0.415, and inside 0.38–0.49 in every scenario but the alternating loom at 28 spm.
- **A model that alternates is the ordinary case**, and it needs the crossing read as one window: a frame whose knees are both gone has no thigh elevation of its own, so it is placed where its neighbours sit and cannot split the crossing it belongs to. Alternating clips then confirm 29–42 windows over three clips — one per stroke, as the consistent clips do — and no covered knee is tracked at the gunwale at all, against 52% before the pass. A pass that instead let each dropout split its crossing found the same cycles but left the finish phase at 0.30–0.61, because the guesses between the dropouts survived into the trough.
- Two rules keep that cleanup from overreaching, both measured. A window is confirmed only where the model **answered** at least a quarter of its frames: a crossing it dropped throughout holds nothing to discard, and the handful of clear frames at its edges that measure weak by chance are not guesses at a covered joint — without the rule the pass discarded about five visible knees per clip on honest dropout footage. And the frames **between** confirmed windows are discarded only where those windows cover at least half the time the knee is low, and then only down to the trough line: a gunwale the knee sits behind for half of every stroke answers the frames between its windows the same way, while a loom crossing a seventh of the cycle leaves a genuinely visible trough that scattered discards would only poke holes in. Discarding the whole band lost cycles at 28 spm (a blind spot that ends above the trough line is not bridged), and capping the cleanup at the 90th percentile of window elevations instead of the trough line measured **worse than no cleanup at all** — the guesses left standing above the cap break the trough into islands.
- The surviving 2–3% of tracked knees in the loom rows are the frames at the edges of a window, where the shaft has left the joint but the reported knee is still weak; they sit ~18 px off and are diluted by the ±0.1 s smoothing the trough is read from.
- False discards cost 0–3.2% of clearly visible knee samples, all of them in the low part of the cycle next to a confirmed window; the top of that range is the gunwale at 28 spm, where the between-window cleanup does most of its work. No near-catch knee (leg extension below 0.2) was discarded in any measured clip, and any knee the pass leaves standing is byte-identical to the model's output.
- Where the model **drops** a covered knee instead of guessing it, the pass finds nothing to do and the frames are byte-identical to before: the hull-trough bridge already carried the finish. Both model behaviours now give the same cycles.
- Evidence accounting: a cycle whose knees were discarded reports `observedFraction` 0.975–1.000, not the 0.75 a naive count of missing knees would give, because an expected occlusion is not evidence the model failed to find. Without that, discarding the guesses would have pushed harsher footage (a hull covering 70–80% of the cycle) below the reliability gate and withheld rates the visible catch peaks support. An alternating clip reports 0.87–0.90 rather than that: the dropouts in it are honest absences, real shortfalls in what the camera saw, and the pass does not explain them away.
- The adaptive TTA gate is **unchanged**, and that is a measured result, not an omission: the gate rejects 3–6% of crops when the knee is guessed (it already accepts a knee at ordinary confidence) and 48–50% when it is dropped — but a dropped knee is missing, so a second opinion cannot reveal it. Relaxing the gate for knees the hull covers saved no passes in any scenario, so the code was not added.
- Regression tests (`tests/knees.test.ts`) pin: a covered knee is never tracked as an observation and always carries its reason; the catch end of the stroke is untouched; false discards stay under 3%; manual points survive; a seat that cannot be calibrated is left alone; the pass is a no-op when the feet are in view and when the model already dropped the knee; it is idempotent; rate, finish phase and evidence hold at 20–28 spm across eight occlusion scenarios, two of them alternating — where one window per crossing is required, and hardly any of the crossing may be left tracked; and the demonstration crew — whose procedural leg was made rigid for this — discards nothing.

Automated checks run for this update: 176 unit tests (plus the opt-in simulation report), `tsc -b`, `vite build` and all 8 production browser tests (Chromium 153 extracted from `@sparticuz/chromium`, since the sandbox cannot download Playwright's own). The demonstration crew's leg geometry changed in this update, so the browser checks matter here: its first metric card still reads 25.0 spm and its crew-timing table still has 21 rows.

**Not validated:** whether a real model snaps a knee to the gunwale or drags it onto the loom as the simulator assumes, or does something else (three behaviours are simulated — the guess, the dropout, and a per-frame coin flip between them — and a real clip could show a fourth); the gunwale height and loom width of any real boat and camera position; the finish proxy against observed blade release; the short holes an alternating model leaves inside a crossing, which the existing short-gap recovery interpolates across and which then enter the cycle signal as `inferred` knees — excluding them was measured and made timing worse (one cycle lost in thirty), so they stay in and are part of why the alternating loom at 28 spm is the weakest row above; and a knee that latches onto the **handle near the catch**, which the pass deliberately does not look for (see the roadmap's open items).

## Remaining release gates

1. Retrieve licensed reference detector + whole-body weights, verify hashes/contracts and test GPU/WASM agreement on actual pixels.
2. Annotate visible joints, hands, feet and seat IDs across complete strokes of both sweep/scull samples. Record occluded joints as such.
3. Measure per-joint normalized pixel/PCK/OKS errors, identity switches, time-to-reacquire, long-gap behavior, TTA benefit and catch/finish error against observed blade entry/release. Check the thigh-elevation cycle timing (and hull-trough bridging) on labelled on-water clips; it is so far validated only in simulation.
4. Extend the synthetic masking suite to real held-out annotations at increasing gap lengths and crossing/overlap conditions.
5. Implement and validate boat motion compensation, hull/rigger/oar cues, camera intrinsics/azimuth and temporal 3D lifting against calibrated multi-view ground truth. A user-assisted seat-line solve and a single-frame lift adapter exist and are tested in simulation (below); motion compensation, automatic cues and temporal lifting remain, and none of it has met real footage.
6. Add observed handle/stretcher/oar constraints, cyclic templates, bone-length checks, bidirectional filtering, angle reliability tests and sequencing diagnoses.
7. Test multiple GPUs, browser memory failure/recovery, VFR clips, rotated/edit-list media and long clips. Current 300 MB / 20,000-frame guards prevent unlimited memory growth but encoded clips are still loaded in memory.

Until these gates pass, twist, sequencing faults, sculling crossover, handle height, blade release, boat set and split per 500 m stay withheld. Normalized joint angles are implemented and are reported when — and only when — a session supplies a spacing, a lens and a level confirmation, the cues agree, and a 3D lift model has been imported; no such model has been evaluated on rowing, so in practice no released session reports them yet.

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

| Run (`scripts/gpu-analysis-check.mjs`) | Provider                                                                      | ONNX Runtime assets actually fetched |
| -------------------------------------- | ----------------------------------------------------------------------------- | ------------------------------------ |
| Adapter available                      | `WebGPU (google · swiftshader) detector / WebGPU (google · swiftshader) pose` | asyncify glue + asyncify wasm only   |
| `requestAdapter()` → null              | `WASM detector / WASM pool (1 worker × 2 threads) pose`                       | plain glue + plain wasm only         |
| Force WASM                             | `WASM detector / WASM pool (1 worker × 2 threads) pose`                       | plain glue + plain wasm only         |

Both the dev server and the production bundle (`vite preview`) were exercised; the run completed with no page errors in every case, and the notes carry the precise reason when the GPU is not used ("handed out no adapter at all … check chrome://gpu", "a software adapter; GPU inference is optional", "Force WASM compatibility mode is on"). Automated checks for this update: 70 unit tests (10 new in `tests/ortRuntime.test.ts`, including the entry→WASM-module pairing re-derived from the installed package and the adapter-probe branches) and the 8 production browser tests.

Added developer tooling (not shipped in the app bundle): `scripts/lib/chromium.mjs` launches a Chromium with the sandbox's bundled NSS/SwiftShader libraries (used by `smoke.mjs`, `worker-check.mjs`, `decode-check.mjs` and `gpu-analysis-check.mjs`), and `scripts/gpu-analysis-check.mjs` prints the provider, the session warnings, and every ONNX Runtime asset requested — the check that made this bug visible.

**Still not validated:** real pretrained YOLOX/DWPose weights, real hardware GPUs and their drivers (this sandbox only offers a SwiftShader software adapter — accuracy and speed claims on real GPUs remain unmet), Safari and Firefox, GPU memory limits with 216 MB fp32 detector weights, and the mid-run device-loss recovery path (implemented, type-checked, and covered by design review only — no device-loss event was triggered in this environment).

## Frame-preserving speed update — engineering checks (2026-10-15)

The supplied private `5.mov` is a 25.298 s, 1920×1080 four-rower sculling quad with 759 encoded samples. Headless Chrome emitted **758** (the missing sample is reported, never invented). With four-person **constant ONNX test doubles**, adaptive TTA, force-WASM, the full detection region, and a four-worker pose pool:

| Browser run                              | Analysis wall | Pose passes | Detector runs | Caveat                                         |
| ---------------------------------------- | ------------: | ----------: | ------------: | ---------------------------------------------- |
| Full-frame bitmap dispatch, post-planner |        47.8 s |       3,032 |           758 | same four workers × four WASM threads          |
| Crop-sized bitmap transfers              |   26.7–26.9 s |       3,032 |           758 | same models, clip, pool, and detector workload |

That is a roughly **44% wall-time reduction for the pipeline fixture**, not a measured gain with pretrained weights. Experimental Auto detector cadence stayed in full-detection mode on this pose-bound fixture (zero predicted crops), so the saving came from limiting worker transfers to the pixels each pose crop needs, not from dropping detector or pose calls. Every-frame detection remains the default until identity and timing are checked with real weights and labels. The test-double graph returns constants independent of video pixels and cannot establish tracking accuracy or real-weight throughput. `PLAYWRIGHT_EXECUTABLE_PATH=/usr/bin/google-chrome VIDEO=5.mov node scripts/crop-equivalence-check.mjs` compared 50 actual decoded-frame pose tensors (five frames × five in-frame/edge crops × original/flipped); all values matched exactly between the original full-frame draw and the cropped-transfer draw. `npm test`: 135 passed, one skipped; `npm run build` and all eight production browser tests passed. The separate older `scripts/pose-perf-check.mjs` diagnostic returned nonzero on this Chrome: its full-frame **detector** VideoFrame-vs-bitmap comparison differs (maximum raw channel difference 127), while its pose-crop comparisons were zero. The new pose-only crop check above passed; the detector discrepancy needs separate investigation and is not evidence about the transferred pose crop. The Vite build still warns that the existing nonmodule `coi-serviceworker.js` tag cannot be bundled; the page serves it separately.

The developer benchmark (`scripts/gpu-analysis-check.mjs`) exports the actual provider, decode/pose/detector counts, setup versus total wait, and stage timings. A physical NVIDIA device by itself does not prove Chrome selected hardware WebGPU; this headless test used WASM.

## Provided-weight CPU profile and body-only mode (2026-10-15)

The user added files named `yolox_l.onnx` (216,746,733 B) and `dw-ll_ucoco_384.onnx` (134,399,116 B). Their model provenance and rowing accuracy have **not** been independently authenticated. On a private 30-frame/1.0 s Quad excerpt from `5.mov`, with every-frame detection and four WASM pose workers × four threads, all runs decoded 30/30 frames and produced 120 seat-frames with no post-fix page errors. The initial pair granted the detector two WASM threads; the final Off run used one otherwise-idle CPU core to grant it three:

| Supplied pair                 | Analysis wall | Detector infer/call | Pose infer/pass | Pose passes |
| ----------------------------- | ------------: | ------------------: | --------------: | ----------: |
| TTA Off · detector 2 threads  |        55.2 s |            1,761 ms |          298 ms |         120 |
| Adaptive · detector 2 threads |        62.0 s |            1,772 ms |          271 ms |         207 |
| TTA Off · detector 3 threads  |    **40.7 s** |        **1,274 ms** |          310 ms |         120 |

Detector and pose stages **overlap**; their totals must not be added. Adaptive spent 87 more pose passes and 6.8 s more wall time than Off at the same two-thread detector setting. On this 20-logical-core CPU, giving the detector a spare third thread improved the matched Off excerpt from 55.2 to 40.7 s (about 26% faster); pose workers remained 4×4. Even then YOLOX-L detection costs about 1.27 s per frame, so replacing 133-joint pose with a smaller 17-joint model **cannot by itself** make the 758-frame clip finish under 60 s. A compatible much smaller detector is also needed; neither the full-clip provided-weight runtime nor a smaller-detector accuracy comparison has been measured. These are throughput findings, not validation of keypoints or seat identities.

Earlier provided-weight attempts surfaced a real UI bug: the App closed the prior live-preview bitmap before React's Viewer draw effect could consume it, causing a detached-image exception and a blank workspace. Preview handles now close after replacement commits. The app accepts **separately imported COCO-17 ONNX** as an explicit mode; it decodes 17 joints and pads unavailable 17–132 as missing in the existing 133-slot sessions without fabricating hand/face/foot observations. Constant 17-joint SimCC and heatmap test doubles exercise the browser plumbing, not accuracy. At the user's request, official OpenMMLab RTMPose-t and Megvii YOLOX-Nano ONNX files were downloaded, checked and held in ignored private directories, never bundled or published. Neither archive/release declares separate weight-redistribution terms; the source-code licenses alone do not establish those rights.

## Small-model full-clip throughput and evidence (2026-10-15)

After URL-specific authorization for local graph execution, the four-rower `5.mov` was analyzed with YOLOX-Nano `[1,3,416,416] → [1,3549,85]` and RTMPose-t COCO-17 `[batch,3,256,192] → SimCC x/y`. Settings: every-frame detector, TTA **Off**, four WASM pose workers × four threads, detector three threads, feet declared hidden. Both runs emitted **758 of 759** encoded frames; no missing frame was invented. The runtime and model identities were recorded in the session, with 133 slots per frame and 116 unmodeled joints flagged missing.

| Local browser                  | Analysis wall | Analyze-to-result | Pose passes | Detector calls | Outcome                                                                             |
| ------------------------------ | ------------: | ----------------: | ----------: | -------------: | ----------------------------------------------------------------------------------- |
| Bundled Chromium 153, headless |        49.4 s |        **50.5 s** |       2,994 |            758 | complete, no page errors; two aborted blob requests and one nonfatal MP4Box warning |
| System Chrome 155, headless    |        34.7 s |        **35.5 s** |       3,003 |            758 | complete, no page errors                                                            |

The browser/launch environments differ, so these are not a controlled A/B speed comparison. Both meet the <60 s **throughput** target on this 20-logical-core CPU; neither establishes seat-ID, joint or stroke-event accuracy. A local review of the rendered skeletons showed approximate alignment to four rowers, but hip/knee confidence was insufficient for most timing metrics. Of 33 segmented seat-cycles in the Chrome run, only **one** yielded a nonwithheld rate (seat 2); the other seats yielded none. The crew-average rate is now withheld when fewer than half the seats contribute reliable cycles, rather than presenting a one-cycle estimate as a crew measurement. No labeled ground truth exists for this clip.

On a separate 30-frame excerpt, Adaptive TTA dispatched 330 passes versus 120 with Off and improved visible body-joint coverage on several seats; a full-clip Adaptive run was **not** benchmarked and should not be assumed to fit under one minute. The models remain private local files; any redistribution would require verified weight terms.

## 3D analysis — simulation checks (2026-10-17)

**What was validated is the geometry implementation. What was not is monocular
depth recovery, and no lift model was involved at any point.**

`npm test`: **238 passing, 2 skipped** across 17 files, including 24 camera-solve
tests, 15 angle tests and 23 lift tests. `npx playwright test`: **9 passing**
browser tests. `tsc --noEmit` and `vite build` clean.

Ground truth is `tests/fixtures/rowingSim3d.ts`: the existing 2D kinematics
(imported, not duplicated) in boat-frame metres, a crew seated along a shell at
0.9 m, hip and shoulder widths, 0.08 m of boat surge, keypoint noise at 1.2% of
crop height, feet hidden below the gunwale, and a pinhole camera at 1800 px,
16 m out, 1.6 m high, on a 1280×720 frame. Each frame records boat-frame joints,
projected pixels and the hip-midpoint depth.

### Camera solve

Eight seats, 8 s clips, 24 spm, noise as above. Distance is compared against the
mean hip-midpoint depth over the whole crew and clip, which is the depth the
solve's `c` reports; scale against `f·cos(az)·distance ÷ depth²`, the image scale
of a metre _along the shell axis_.

| True azimuth | Recovered | Error | Depth (solved vs truth) | Scale error | Fit residual | Confidence | Source     |
| ------------ | --------- | ----- | ----------------------- | ----------- | ------------ | ---------- | ---------- |
| 10°          | 9.96°     | 0.04° | 16.12 vs 16.12 m        | 0.00%       | 0.02 px      | 81%        | calibrated |
| 18°          | 17.96°    | 0.04° | 16.22 vs 16.22 m        | 0.01%       | 0.02 px      | 97%        | calibrated |
| 24°          | 23.96°    | 0.04° | 16.29 vs 16.29 m        | 0.01%       | 0.02 px      | 94%        | calibrated |
| 30°          | 29.96°    | 0.04° | 16.36 vs 16.35 m        | 0.01%       | 0.02 px      | 92%        | calibrated |
| 45°          | 44.96°    | 0.04° | 16.51 vs 16.50 m        | 0.02%       | 0.02 px      | 83%        | calibrated |

Confidence is lowest at 10° because the perspective is poorly conditioned there
(the vanishing point runs away), and at 45° because assuming the principal point
is the frame centre is worth about 1.0° of azimuth at that turn. Both are
reported as cues, not hidden.

Every abstention below fires with a reason naming what to do. Nine were checked:

| Refused                            | Reason class                                                               |
| ---------------------------------- | -------------------------------------------------------------------------- |
| Square-on (0°)                     | No perspective in the spacing at all; azimuth and distance cannot separate |
| Nearly square-on (0.4°) with noise | The turn is inside the keypoint noise                                      |
| No seat spacing                    | A known length on the shell is missing                                     |
| No focal length                    | Azimuth and distance stay locked together                                  |
| Clip not confirmed level           | The boat's vertical would be an assumption                                 |
| Mirrored clip                      | The camera would have to be behind the shell                               |
| Camera turned 14° during the clip  | The two halves solve to different azimuths                                 |
| Only two seats                     | Three fix a projective map                                                 |
| 70° off square                     | Beyond the 60° the solve will report                                       |

### Normalized angles against the same angles measured in the image

One seat, every lifted frame of an 8 s clip, feet declared visible so the knee
and shin angles exist to compare:

| Camera off square | Knee, measured in the image | Knee, lifted into the boat frame | Layback, image (worst) | Layback, lifted (worst) |
| ----------------- | --------------------------- | -------------------------------- | ---------------------- | ----------------------- |
| 10°               | 3.8° mean, 11.8° worst      | 0.00°                            | 5.7°                   | 0.02°                   |
| 24°               | 4.4° mean, 16.7° worst      | 0.00°                            | 8.3°                   | 0.00°                   |
| 45°               | 8.6° mean, 29.4° worst      | 0.00°                            | 14.5°                  | 0.02°                   |

The lifted column is exact because the test's oracle **is** the fixture's own 3D
joints: this measures the pipeline (projection → solve → placement → rotation →
angle definition), not a model. The image column is the reason the pipeline
exists — the same knee read off pixels is wrong by degrees at every azimuth a
launch films from, and a 2D cosine correction does not recover it.

### Lift adapter checks

Placing the oracle through the adapter reproduces the boat-frame angles with a
reprojection of about 2 px against 1.5 px of keypoint noise, at 10–45° off
square. It also:

- accepts a lift in millimetres (fitted scale 0.00096, same angles), because the
  first guess at the scale comes from the crop, not from the model's units;
- leaves every angle unchanged when the calibration's distance is 30% wrong —
  the pose is placed further out and scaled up to match, which is why normalized
  angles can be reported while metric lengths stay withheld;
- refuses a lift with its y axis flipped (31 px reprojection against a 7 px
  limit), one mirrored in x, one whose depth is off by 20 m (84 px), an output of
  24 joints, a coordinate tensor that is not three numbers a joint, an ambiguous
  `[4,4,3,17]` layout, and a frame with only five observed joints;
- does not lift a joint the 2D pipeline discarded: with the left knee rewritten
  to `missing`, the knee, hip and shin angles through it are withheld naming the
  knee, while the other leg, the layback and the elbows are still reported;
- withholds on water, where the feet are below the gunwale: knee and shin angles
  do not exist there, thigh, hip, elbow and layback do.

### What none of this establishes

- **No real footage.** Every number comes from a simulator, including its
  occlusion and noise. A launch clip has spray, a hull, wake, a pan and rowers
  who overlap.
- **No lift model was run.** No rowing-validated 3D lift exists to evaluate, none
  is bundled and none downloads. The adapter is tested against an oracle.
  Milestone 8's annotated benchmark is still the gate, and until it passes no
  released session reports an angle.
- **No monocular depth recovery.** Depth is an input from the calibration. A
  wrong spacing or a wrong lens scales the distance and the pixels-per-metre and
  leaves the azimuth alone.
- **The depth axis's sign is not checkable per frame.** A rower is nearly planar
  and nearly parallel to the image plane, so a z-flipped lift reprojects at 6 px
  against a 2 px honest fit and its interior angles are unchanged, because a
  reflection preserves them. The convention is a contract with the model, to be
  checked once against a known-good release; it is the reason boat set, shoulder
  level and trunk twist stay withheld.
- **One azimuth per clip.** A camera that pans or surges is not described by a
  single solve; the stability cue prices it and a 14° turn across the clip is
  refused. Motion-compensated calibration remains a research milestone.
- **No temporal model.** The lift is per frame, with no smoothing, no bone-length
  priors across a clip and no cyclic prior.
