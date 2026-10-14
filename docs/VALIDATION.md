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
