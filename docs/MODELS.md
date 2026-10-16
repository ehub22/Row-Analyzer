# Model setup and accuracy contract

## Bring authentic weights

The app imports local `.onnx` files; it does not execute Python or send video to model servers. Get models from official upstream projects and review the applicable code, data, and weight licenses before use or redistribution:

- DWPose: https://github.com/IDEA-Research/DWPose
- MMPose RTMPose / RTMW: https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose
- ViTPose: https://github.com/ViTAE-Transformer/ViTPose
- Sapiens: https://github.com/facebookresearch/sapiens
- YOLOX: https://github.com/Megvii-BaseDetection/YOLOX

**Whole-body baseline pair:** YOLOX-L `yolox_l.onnx` plus DWPose `dw-ll_ucoco_384.onnx`. These supplied files have been run locally, but their provenance and rowing accuracy are not independently verified. A COCO-17 body checkpoint is now supported as a separate mode, not a substitute for 133-joint manual labeling needs. The [OpenMMLab RTMPose-t archive](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.zip) was safely extracted to an ignored local directory at the user's request and its ONNX I/O contract inspected. It is **not bundled, cached by default, or redistributed**: the archive carries no weight-license terms, and the [MMPose source license](https://github.com/open-mmlab/mmpose/blob/main/README.md) alone does not establish checkpoint redistribution rights. After URL-specific local approval, the small pose model and YOLOX-Nano processed the entire clip in under 60 s on the measured CPU/browser setups (see validation report). This is throughput, **not** joint, seat-identity or stroke-timing accuracy validation.

## Supported contracts

### YOLOX
- Input float32 NCHW `[1,3,S,S]`, square fixed input (usually 640).
- Raw **BGR** 0–255; top-left letterbox with padding value 114.
- Output `[1,N,85]`: raw grid-relative x/y and log-width/height, objectness and 80 class scores; person class index 0.
- Strides 8/16/32; decode, confidence threshold 0.35, NMS IoU 0.45.
- A graph with already-decoded coordinates is **not** compatible with this adapter.
- A [YOLOX-Nano COCO ONNX](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx) was downloaded to an ignored private directory for local evaluation. Static metadata shows `[1,3,416,416]` → `[1,3549,85]` with raw-grid output; this appears compatible with the adapter's size-derived grid decode and raw BGR/114 letterbox. After URL-specific local approval it ran successfully on the full clip with the COCO-17 model, though detection and pose accuracy are **not** benchmark-validated. No separate weight-license notice was found; it is not bundled or redistributed.

### YOLOv8 / YOLO11 (optional detector adapter)
- RGB 0–1, square NCHW input, top-left letterbox.
- Output `[1,84,N]` with decoded center-x/y/width/height and class scores; no objectness channel, no embedded NMS.
- End-to-end `[1,N,6]` or YOLO26 exports are rejected.
- Review Ultralytics licensing separately; weights are not bundled here.

### SimCC pose (COCO-17 body or COCO-WholeBody-133)
- Select **Pose joint layout** explicitly: 133 remains the default; 17 requires a separately imported local trained COCO-17 ONNX file. The cached/downloadable DWPose whole-body weights are never used in body-only mode.
- Input `[1,3,H,W]`; default whole-body **W=288, H=384**, while a compatible small body-only export may use **W=192, H=256**. Settings must match the actual graph.
- RGB normalization: mean `[123.675,116.28,103.53]`, standard deviation `[58.395,57.12,57.375]`.
- Two outputs in x-then-y order: `[1,J,Nx]`, `[1,J,Ny]`, with the same coordinate scale on both axes and **J=17 or 133 matching the selection**. Symbolic metadata is verified by an actual zero-input run before video decoding.
- COCO-17 uses the same first 17 body indices as COCO-WholeBody-133, which adds 6 feet, 68 face, 21 left-hand and 21 right-hand joints. No inserted OpenPose neck or arbitrary 17-joint taxonomy.
- Argmax plus local parabolic sub-pixel refinement. Confidence uses minimum axis peak clipped to [0,1], not a calibrated probability. Crop geometry, normalization and confidence require empirical validation for every new export. Taking the first 17 outputs of a 133-joint graph **does not speed up its backbone**.

### Heatmap pose exports
- Same image preprocessing; one `[1,J,Hh,Wh]` heatmap tensor with J matching the selected 17/133 layout.
- Peak + local 2D parabolic refinement and inverse crop mapping.
- The export must match the exact non-UDP coordinate convention. Generic ViTPose body checkpoints/UDP exports are not automatically compatible.
- Current TTA fuses **decoded coordinates**, weighted by confidence, and penalizes inter-pass disagreement. It does **not** yet perform full heatmap-space multi-scale fusion or DARK refinement.

### TTA
Two crop scales (1.25, 1.5) × original/flipped input. Preserve aspect ratio and swap all body, foot, facial, and hand mirror indices. Keep per-joint provenance and confidence. Padded crops extend beyond the image with black padding. Per-frame detection plus per-person four-pass pose inference is deliberately slow.

The *Adaptive* policy (default) runs the first pass alone and accepts it when every gated body joint reaches 0.3 and the body mean reaches 0.5; otherwise the same four passes run and fuse. Joints the session declares hidden are not gated: with *Feet in view: Hidden by the hull* (the on-water default) that is the ankles, which no extra pass can reveal.

### Detector cadence

*Every frame* is the default for fresh person detection. Experimental *Auto* starts with full detection and only attempts to reuse short-lived crops when measured detector cost dominates pose, with an anchor at least every third decoded frame. The next frame's crop comes from a completed, confidence-gated seat pose; weak/ambiguous crops trigger fresh detection and a same-frame pose retry. Every decoded frame with a valid seat still runs inference with the selected pose model. A predicted **box** is not a fresh person observation: each frame records `boxSource` and `detectorAge`, and identity confidence is capped by that age. Existing sessions without those optional fields were made before cadence. *Every frame* restores full detection for comparisons. This policy is not validated for real-weight seat identity, hands or timing accuracy on rowing footage.

### Hidden feet
On the water the feet (ankles 15–16, feet 17–22) are below the gunwale for the whole clip, so whatever the model returns for them is a guess. When the session declares them hidden, the worker replaces them with missing joints flagged *Feet hidden by the hull* before tracking, and stroke cycles use hips and knees only. A manually placed foot point is kept. Choose *Visible* for ergometer or overhead footage. A COCO-17 model may estimate visible ankles 15–16 but **never supplies toe/heel joints 17–22**, regardless of camera setup; those slots stay unavailable unless manually labeled.

## Automatic download and cache

Weights are **never** committed to Git or bundled in the Pages build (`*.onnx` is ignored; `dist/` contains only the app). Instead, the app loads each model in this order:

1. **Local file** chosen under *Models & processing* — always wins.
2. **Cache Storage** (`row-analyzer-models-v1`) — a previous successful download, re-validated for exact size and ONNX header; a corrupt entry is evicted.
3. **Streamed download** from the configured URL, with a live progress bar. The file must match the expected byte length (and SHA-256 when configured) before it is cached.
4. Otherwise an actionable error and the local-file import path.

The automatic pose cache/download specification is **133-joint DWPose only**. In 17-joint mode a local, compatible COCO-17 file is mandatory; the cached whole-body model is never substituted. The session keeps 133 slots for backward-compatible exports, with 17–132 marked *not provided by model* unless manually labeled.

Build-time configuration (repository **variables**, passed by the Pages workflow; not secrets):

| Variable | Meaning |
| --- | --- |
| `VITE_YOLOX_URL` | HTTPS URL of `yolox_l.onnx` |
| `VITE_DWPOSE_URL` | HTTPS URL of `dw-ll_ucoco_384.onnx` |
| `VITE_YOLOX_BYTES`, `VITE_DWPOSE_BYTES` | Expected exact sizes. Defaults are the Drive metadata sizes (216,746,733 and 134,399,116 bytes). `0` disables the size check (test builds only). |

Requirements for a download URL: HTTPS, CORS readable from the Pages origin (`Access-Control-Allow-Origin`), and the exact file bytes. Google Drive sharing links are generally **not** CORS-readable from another origin, so host the files somewhere that is, after confirming the licenses allow redistribution to your own users. Verify CORS with a browser before relying on it. With no URL set, the app shows *No download URL in this build* and the local-file import still works.

Downloaded data is model weights only. Video frames and keypoints are never cached. *Clear cached models* in the dialog removes them.

Each ONNX session also runs a shape-level contract check when it is created (`src/core/compat.ts`): YOLOX `[1,N,85]` or YOLOv8/11 `[1,84,N]`, square detector input, DWPose input matching the configured width/height, and the **selected 17- or 133-joint** SimCC/heatmap outputs. Symbolic metadata is confirmed with a zero-input inference before video decoding. A failed check stops analysis before any frame is decoded. Passing this check confirms the I/O contract only, not accuracy.

## Browser runtime

WebGPU is attempted first (unless *Force WASM* is set) and must pass a warm-up inference on a zero tensor before any frame is decoded: sessions that create successfully can still fail on the first real run (unsupported operators, GPU buffer limits on large fp32 models, driver blocklists), and that failure falls back to WASM automatically with the reason recorded in the session warnings instead of aborting the analysis. Adapter creation fails outright on many Chrome-on-Linux, Firefox, and Safari setups, so the reason is always recorded and shown: no `navigator.gpu`, no adapter at all, only a software adapter, or a specific ONNX Runtime error.

### WASM build selection (this was the "GPU never works" bug)

ONNX Runtime Web ships one JavaScript entry point per execution-provider set, and **each entry expects exactly one of the prebuilt WASM modules**. The pairing is compiled into the entry, not chosen at runtime:

| Entry point | WASM module it loads | Exposes the WebGPU bridge |
| --- | --- | --- |
| `onnxruntime-web/webgpu` | `ort-wasm-simd-threaded.asyncify` | yes (`webgpuInit`) |
| `onnxruntime-web` / `/all` | `ort-wasm-simd-threaded.jsep` | no |
| `onnxruntime-web/wasm` | `ort-wasm-simd-threaded` | no |
| `onnxruntime-web/jspi` | `ort-wasm-simd-threaded.jspi` | yes (`webgpuInit`) |

The app previously pinned the **jsep** module via `env.wasm.wasmPaths` while importing the **webgpu** entry, which asks the module for `webgpuInit` — a function that only exists in the asyncify/jspi builds. Result: `WebGPU failed for <model> (no available backend found. ERR: [webgpu] TypeError: ...webgpuInit is not a function)` on every browser and device, because it was a static file mismatch, not a hardware limit. `src/core/ortRuntime.ts` now pins the matching pair, `tests/ortRuntime.test.ts` re-derives the table above from the installed package and fails if an upgrade changes it, and WebGPU runs use the asyncify build while WASM-only runs use the smaller plain build (about 14 MB instead of 27 MB, and no asyncify overhead in the WASM pool).

A probed adapter is passed to ONNX Runtime through `env.webgpu.adapter`, so the library does not request a second, possibly different adapter. A *software* adapter (SwiftShader) is detected and reported; it is not used by default because the parallel WASM pool is usually faster. If a GPU session dies mid-run, that session is rebuilt on WASM and the run continues with a recorded warning instead of discarding the decoded frames.

WASM inference is parallelised two ways (see `src/core/parallel.ts` for the exact policy):

1. **Pose-worker pool (works everywhere, no page setup).** Pose inference dominates runtime (up to 4 TTA passes × every detected person, per frame), and every pass is independent. The main thread spawns N pose workers (Auto: one per core minus one, capped at 4; configurable under *Models & settings → Parallel pose workers*) and connects each directly to the analysis worker over a `MessageChannel`, so tensors never relay through the UI thread. Each worker owns a WASM pose session; jobs are routed least-busy-first and results are reassembled in pass order before TTA fusion. If a worker fails to start or dies mid-run, the analysis continues on the remaining workers or on a local session in the analysis worker — reduced speed is reported, never hidden. Cost: each pool worker loads its own copy of the pose model (~134 MB for DWPose-l), so raising the worker count trades RAM for speed.
2. **Threaded WASM (needs cross-origin isolation).** ONNX Runtime's `numThreads > 1` requires `SharedArrayBuffer`, which browsers only grant on cross-origin-isolated pages (COOP + COEP headers); without isolation ORT silently clamps to 1 thread. Local dev/preview sends the headers via Vite; the GitHub Pages deployment gets them from `public/coi-serviceworker.js`, a service worker that re-serves same-origin responses with COOP/COEP after one automatic reload. When isolation is active, the core budget is split across the pool (e.g. 8 cores → 4 workers × 2 threads); when it is not, each session runs single-threaded and the pool provides the parallelism. The active configuration is shown in *Models & settings* and recorded in each session's `provider` string.

The detector runs once per frame in the analysis worker (with the full thread budget when isolated, since the pool is idle at that point). Video decode, detection, and per-frame ordering remain sequential; tracking still consumes frames in timestamp order exactly as before. Model weights are loaded into memory for each analysis run and may be kept in Cache Storage as described above. Large/huge transformers may exceed GPU/storage-buffer/browser memory limits; compatible ONNX operators alone do not guarantee feasibility.

## Optional local GPU companion — proposed, not implemented

A useful higher-accuracy extension is an offline Python CLI (not a public video-upload service):

1. Decode presentation timestamps with PyAV.
2. Use PyTorch/MMPose or ONNX Runtime CUDA for RTMDet/YOLOX + RTMW-x or compatible whole-body ViTPose/Sapiens models.
3. Fuse spatial distributions over flips and scales; retain raw outputs for QA.
4. Apply calibrated camera estimation plus a rowing-adapted temporal 3D model.
5. Export the versioned session JSON described in `src/core/types.ts`; optional 3D data must add explicit coordinate-frame/camera/model provenance in a future schema version.
6. Import the JSON and original clip into this browser review UI.

This avoids mixed-content/localhost problems on an HTTPS Pages site. If a future live companion API is added, require explicit opt-in, a loopback-only service, authenticated requests, restricted origins and a same-origin local proxy. Do not expose a GPU inference service to the network by default.

## Labeling and adaptation

Export keyframe labels after correcting visible joints. The export is **partial COCO-style**, not a complete ready-to-train COCO dataset: it contains video filename, frame timestamp, seat, box, joint names and only manually placed points with visibility 2. Unlabeled and inferred points have visibility 0. Extract the matching frames locally and add image IDs/license/consent metadata before training. Split evaluation by athlete, crew and recording session. Never count adjacent frames or propagated coordinates as independent human ground truth.
