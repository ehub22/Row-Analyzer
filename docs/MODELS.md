# Model setup and accuracy contract

## Bring authentic weights

The app imports local `.onnx` files; it does not execute Python or send video to model servers. Get models from official upstream projects and review the applicable code, data, and weight licenses before use or redistribution:

- DWPose: https://github.com/IDEA-Research/DWPose
- MMPose RTMPose / RTMW: https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose
- ViTPose: https://github.com/ViTAE-Transformer/ViTPose
- Sapiens: https://github.com/facebookresearch/sapiens
- YOLOX: https://github.com/Megvii-BaseDetection/YOLOX

**Recommended initial browser evaluation pair:** YOLOX-L `yolox_l.onnx` plus DWPose `dw-ll_ucoco_384.onnx`. This is a practical whole-body baseline, not proof it is the most accurate rowing model. RTMW-x high-resolution and whole-body ViTPose large/huge should be benchmarked against it before selecting a production model. Do not substitute a COCO-17 body checkpoint.

## Supported contracts

### YOLOX
- Input float32 NCHW `[1,3,S,S]`, square fixed input (usually 640).
- Raw **BGR** 0–255; top-left letterbox with padding value 114.
- Output `[1,N,85]`: raw grid-relative x/y and log-width/height, objectness and 80 class scores; person class index 0.
- Strides 8/16/32; decode, confidence threshold 0.35, NMS IoU 0.45.
- A graph with already-decoded coordinates is **not** compatible with this adapter.

### YOLOv8 / YOLO11 (optional detector adapter)
- RGB 0–1, square NCHW input, top-left letterbox.
- Output `[1,84,N]` with decoded center-x/y/width/height and class scores; no objectness channel, no embedded NMS.
- End-to-end `[1,N,6]` or YOLO26 exports are rejected.
- Review Ultralytics licensing separately; weights are not bundled here.

### DWPose / compatible RTMW SimCC
- Input `[1,3,H,W]`; default **W=288, H=384**, RGB.
- Mean `[123.675,116.28,103.53]`, standard deviation `[58.395,57.12,57.375]`.
- Two outputs in x-then-y order: `[1,133,2W]`, `[1,133,2H]` (dimension-derived scale also supported).
- COCO-WholeBody order: 17 body, 6 feet, 68 face, 21 left hand, 21 right hand. No inserted OpenPose neck.
- Argmax plus local parabolic sub-pixel refinement. Confidence uses minimum axis peak clipped to [0,1], not a calibrated probability. Different SimCC training normalizations require empirical threshold calibration.

### Whole-body heatmap exports
- Same image preprocessing; one `[1,133,Hh,Wh]` heatmap tensor.
- Peak + local 2D parabolic refinement and inverse crop mapping.
- The export must match the exact non-UDP coordinate convention. Generic ViTPose body checkpoints/UDP exports are not automatically compatible.
- Current TTA fuses **decoded coordinates**, weighted by confidence, and penalizes inter-pass disagreement. It does **not** yet perform full heatmap-space multi-scale fusion or DARK refinement.

### TTA
Two crop scales (1.25, 1.5) × original/flipped input. Preserve aspect ratio and swap all body, foot, facial, and hand mirror indices. Keep per-joint provenance and confidence. Padded crops extend beyond the image with black padding. Per-frame detection plus per-person four-pass pose inference is deliberately slow.

## Automatic download and cache

Weights are **never** committed to Git or bundled in the Pages build (`*.onnx` is ignored; `dist/` contains only the app). Instead, the app loads each model in this order:

1. **Local file** chosen under *Models & processing* — always wins.
2. **Cache Storage** (`row-analyzer-models-v1`) — a previous successful download, re-validated for exact size and ONNX header; a corrupt entry is evicted.
3. **Streamed download** from the configured URL, with a live progress bar. The file must match the expected byte length (and SHA-256 when configured) before it is cached.
4. Otherwise an actionable error and the local-file import path.

Build-time configuration (repository **variables**, passed by the Pages workflow; not secrets):

| Variable | Meaning |
| --- | --- |
| `VITE_YOLOX_URL` | HTTPS URL of `yolox_l.onnx` |
| `VITE_DWPOSE_URL` | HTTPS URL of `dw-ll_ucoco_384.onnx` |
| `VITE_YOLOX_BYTES`, `VITE_DWPOSE_BYTES` | Expected exact sizes. Defaults are the Drive metadata sizes (216,746,733 and 134,399,116 bytes). `0` disables the size check (test builds only). |

Requirements for a download URL: HTTPS, CORS readable from the Pages origin (`Access-Control-Allow-Origin`), and the exact file bytes. Google Drive sharing links are generally **not** CORS-readable from another origin, so host the files somewhere that is, after confirming the licenses allow redistribution to your own users. Verify CORS with a browser before relying on it. With no URL set, the app shows *No download URL in this build* and the local-file import still works.

Downloaded data is model weights only. Video frames and keypoints are never cached. *Clear cached models* in the dialog removes them.

Each ONNX session also runs a shape-level contract check when it is created (`src/core/compat.ts`): YOLOX `[1,N,85]` or YOLOv8/11 `[1,84,N]`, square detector input, DWPose input matching the configured width/height, and 133-joint SimCC or heatmap outputs. A failed check stops analysis before any frame is decoded. Passing this check confirms the I/O contract only, not accuracy.

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
