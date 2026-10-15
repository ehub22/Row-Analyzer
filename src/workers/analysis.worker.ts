import * as ort from "onnxruntime-web/webgpu";
import {
  configureOrtRuntime,
  ortWasmOnlyVariant,
  ortWebGpuVariant,
  probeWebGpuAdapter,
  webGpuProviderLabel,
  type GpuProbe,
} from "../core/ortRuntime";
import { decodeVideo } from "../core/video";
import { decodeYolo, decodeYolox } from "../core/detection";
import { decodeSimCC, decodeHeatmap, unflip, fusePoses } from "../core/pose";
import { SeatTracker } from "../core/tracking";
import { checkDetectorContract, checkPoseContract } from "../core/compat";
import { recoverOcclusion } from "../core/occlusion";
import { segmentStrokes } from "../core/strokes";
import { estimateCamera } from "../core/geometry";
import { FrameSource, Rasterizer } from "../core/preprocess";
import { SequentialPipeline } from "../core/queue";
import { StageTimer, buildReport } from "../core/perf";
import { assessPose, describeTta, passPlan, type PosePass } from "../core/tta";
import {
  planParallelism,
  readParallelEnv,
  summarizePlan,
  type ParallelPlan,
} from "../core/parallel";
import { PoseWorkerPool, type PoseJob } from "../core/pool";
import type {
  Boat,
  ModelSettings,
  Point,
  PoseFrame,
  Session,
  Box,
} from "../core/types";

interface Job {
  file: File;
  detector: File;
  pose: File;
  settings: ModelSettings;
  boat: Boat;
  /** Plan computed by the main thread (it also sizes the worker pool). */
  parallel?: ParallelPlan;
}

const send = (type: string, payload: unknown) => postMessage({ type, payload });
const errText = (e: unknown) =>
  (e instanceof Error ? e.message : String(e)).slice(0, 300);

// MessagePorts to the pose-worker pool, connected by the main thread before
// the job message arrives (see App.tsx). Empty when the pool is unavailable,
// in which case pose inference runs on a local session exactly as before.
const poolPorts: MessagePort[] = [];
// The pool of the currently running job, so transport-level worker errors
// relayed by the main thread ("pool-error") can mark endpoints dead.
let activePool: PoseWorkerPool | null = null;
// Stage timer of the current run: pose workers are created before the run
// starts, so their reported pass timings land here once it does.
let timerRef: StageTimer | null = null;

interface LoadedModel {
  session: ort.InferenceSession;
  provider: "WebGPU" | "WASM";
  /** Adapter description when running on the GPU, for the session record. */
  gpuLabel?: string;
}

/**
 * Creates a session, preferring WebGPU when an adapter was probed successfully.
 * A WebGPU session that *creates* can still fail on its first real inference
 * (unsupported operator, buffer-size limits on large fp32 models, driver
 * blocklists), so every WebGPU candidate must pass a warm-up inference with a
 * zero tensor before any video is decoded. Any failure falls back to WASM and
 * is reported, never silently swallowed.
 */
async function loadModel(
  bytes: ArrayBuffer,
  name: string,
  opts: {
    /** False for Force WASM, or when the WebGPU probe found no adapter. */
    allowWebGpu: boolean;
    gpuLabel: string;
    probeDims: (session: ort.InferenceSession) => number[];
    notes: string[];
  },
): Promise<LoadedModel> {
  if (opts.allowWebGpu) {
    try {
      const session = await ort.InferenceSession.create(bytes, {
        executionProviders: ["webgpu"],
      });
      try {
        const dims = opts.probeDims(session);
        const probe = new ort.Tensor(
          "float32",
          new Float32Array(dims.reduce((a, b) => a * b, 1)),
          dims,
        );
        try {
          const out = await session.run({ [session.inputNames[0]]: probe });
          Object.values(out).forEach((t) => t.dispose());
        } finally {
          probe.dispose();
        }
        return { session, provider: "WebGPU", gpuLabel: opts.gpuLabel };
      } catch (e) {
        await session.release().catch(() => {});
        throw e;
      }
    } catch (e) {
      const note = `WebGPU failed for ${name} (${errText(e)}); using WASM instead.`;
      opts.notes.push(note);
      send("status", note);
    }
  }
  return {
    session: await ort.InferenceSession.create(bytes, {
      executionProviders: ["wasm"],
    }),
    provider: "WASM",
  };
}

/** Fixed input dims from session metadata, with a per-index fallback. */
function probeDimsFrom(
  session: ort.InferenceSession,
  fallback: number[],
): number[] {
  const meta = session.inputMetadata[0] as
    { shape?: readonly (number | string)[] } | undefined;
  const shape = meta?.shape ?? [];
  return fallback.map((f, i) =>
    typeof shape[i] === "number" && (shape[i] as number) > 0
      ? (shape[i] as number)
      : f,
  );
}

// Rasterisation lives in core/preprocess.ts: frames are converted to one
// ImageBitmap (core/preprocess.ts explains why that is many times cheaper than
// drawing the VideoFrame per pass, with identical pixels) and each model input
// shape keeps one reusable canvas + tensor buffer here.

async function infer(session: ort.InferenceSession, input: ort.Tensor) {
  try {
    return await session.run({ [session.inputNames[0]]: input });
  } finally {
    input.dispose();
  }
}

const tensorShape = (m: {
  isTensor: boolean;
  shape?: readonly (number | string)[];
}) => (m.isTensor && m.shape ? m.shape : []);

/** Decodes one local pose inference into source-image points (unflipped). */
function decodePoseOutputs(
  session: ort.InferenceSession,
  outputs: Record<string, ort.Tensor>,
  format: "simcc" | "heatmap",
  crop: Box,
  flip: boolean,
): Point[] {
  const vals = session.outputNames.map((n) => outputs[n]);
  let points: Point[];
  if (format === "simcc") {
    if (
      vals.length !== 2 ||
      vals[0].dims.length !== 3 ||
      vals[0].dims[1] !== 133 ||
      vals[1].dims[1] !== 133
    )
      throw Error(
        "Expected a 133-joint whole-body SimCC export, not a 17-joint body model.",
      );
    points = decodeSimCC(
      vals[0].data as Float32Array,
      vals[1].data as Float32Array,
      vals[0].dims[2],
      vals[1].dims[2],
      crop,
    );
  } else {
    if (vals[0].dims.length !== 4 || vals[0].dims[1] !== 133)
      throw Error("Expected [1,133,H,W] heatmaps.");
    points = decodeHeatmap(
      vals[0].data as Float32Array,
      vals[0].dims[3],
      vals[0].dims[2],
      crop,
    );
  }
  return flip ? unflip(points, crop) : points;
}

async function run(job: Job) {
  const plan: ParallelPlan =
    job.parallel ?? planParallelism(readParallelEnv(), job.settings.workers);
  const notes: string[] = [];
  send("status", "Loading detector and whole-body pose model…");
  send("status", `Parallel plan: ${summarizePlan(plan)}.`);

  // WebGPU is probed before any model is touched: "navigator.gpu exists" is
  // not the same as "an adapter is available", and the reason has to reach the
  // user. The probed adapter is handed to ONNX Runtime so that it does not ask
  // for a second, possibly different, one.
  let gpu: GpuProbe | null = null;
  if (job.settings.wasm) {
    notes.push(
      "Force WASM compatibility mode is on: WebGPU is not attempted for this run.",
    );
  } else {
    gpu = await probeWebGpuAdapter();
    if (gpu.kind === "adapter" && gpu.adapter) {
      (ort.env.webgpu as { adapter?: unknown }).adapter = gpu.adapter;
      notes.push(gpu.reason);
      send("status", gpu.reason);
    } else {
      notes.push(`WebGPU is unavailable: ${gpu.reason}`);
    }
  }
  const allowWebGpu = Boolean(gpu && gpu.kind === "adapter");
  const gpuLabel = gpu ? webGpuProviderLabel(gpu) : "WebGPU";

  // ORT reads wasmPaths/numThreads when the WASM runtime initialises (first
  // session in this worker), so the choice is made after the probe:
  //  * WebGPU runs on the asyncify build — the only one that carries the
  //    native WebGPU bridge (`webgpuInit`) that `onnxruntime-web/webgpu`
  //    calls. Pinning a build without it is what made every GPU run fail with
  //    "webgpuInit is not a function" on every device and browser.
  //  * Runs that cannot use WebGPU load the plain build instead: smaller
  //    (~14 MB vs ~27 MB) and free of asyncify overhead in the WASM pool.
  configureOrtRuntime(
    ort.env,
    allowWebGpu ? ortWebGpuVariant : ortWasmOnlyVariant,
    plan.detectorThreads,
  );

  const detectorBytes = await job.detector.arrayBuffer();
  const poseBytes = await job.pose.arrayBuffer();
  const d = await loadModel(detectorBytes, job.detector.name, {
    allowWebGpu,
    gpuLabel,
    probeDims: (s) => probeDimsFrom(s, [1, 3, 640, 640]),
    notes,
  });

  // Pose execution strategy, in order:
  //  1. a probed WebGPU session in this worker (pool never initialised);
  //  2. the parallel WASM pose-worker pool (main thread supplied the ports);
  //  3. a local WASM session — also the fallback if the pool dies mid-run.
  let localPose: LoadedModel | null = null;
  let pool: PoseWorkerPool | null = null;
  let poolAlive = false;
  let poseProvider = "";

  if (allowWebGpu) {
    const attempt = await loadModel(poseBytes, job.pose.name, {
      allowWebGpu: true,
      gpuLabel,
      probeDims: (s) =>
        probeDimsFrom(s, [1, 3, job.settings.height, job.settings.width]),
      notes,
    });
    if (attempt.provider === "WebGPU") {
      localPose = attempt;
      poseProvider = gpuLabel;
    } else {
      // loadModel already fell back to a WASM session; keep it only as the
      // emergency fallback and prefer the pool for actual inference.
      localPose = attempt;
    }
  }

  if (!localPose || localPose.provider !== "WebGPU") {
    if (poolPorts.length) {
      pool = new PoseWorkerPool(
        poolPorts.map((port) => ({
          send: (message: unknown, transfer?: Transferable[]) => {
            if (transfer) port.postMessage(message, transfer);
            else port.postMessage(message);
          },
        })),
        // Worker-side pass timings feed the same report as the coordinator's
        // own stages, so "preprocess or inference?" is answerable from the
        // session notes instead of guessed.
        {
          onPassTimings: (timings) => {
            if (timings.preprocess)
              timerRef?.add("pose.preprocess", timings.preprocess);
            if (timings.infer) timerRef?.add("pose.infer", timings.infer);
            if (timings.decode) timerRef?.add("pose.decode", timings.decode);
          },
        },
      );
      // Wire inbound port events into the pool's dispatcher.
      poolPorts.forEach((port, i) => {
        port.onmessage = (e: MessageEvent) => pool?.handleMessage(i, e.data);
        port.onmessageerror = () =>
          pool?.handleError(i, new Error("Pose worker port error."));
      });
      pool.init({
        // The File is a blob reference: each worker reads its own copy,
        // instead of structured-cloning a ~134 MB ArrayBuffer per worker.
        poseFile: job.pose,
        format: job.settings.format,
        width: job.settings.width,
        height: job.settings.height,
        wasmThreads: plan.wasmThreads,
      });
      activePool = pool;
      try {
        // A pose worker whose script fails to load never answers init;
        // without this timeout the run would hang instead of falling back.
        await Promise.race([
          pool.ready(),
          new Promise<never>((_, reject) =>
            setTimeout(
              () =>
                reject(new Error("pose workers did not start within 120 s")),
              120_000,
            ),
          ),
        ]);
        // ready() resolves on the first worker; give the rest a brief grace
        // period to finish session creation so early frames (and the
        // reported worker count) use the whole pool, not just the fastest.
        const graceDeadline = Date.now() + 5_000;
        while (pool.settledCount < pool.size && Date.now() < graceDeadline)
          await new Promise((r) => setTimeout(r, 100));
        poolAlive = true;
        // The pool is healthy: release the emergency local session (created
        // when the WebGPU probe fell back) instead of holding a fifth copy
        // of the pose model in RAM. ensureLocalPose() recreates it on the
        // unlikely chance every pool worker dies mid-run.
        if (localPose) {
          await localPose.session.release().catch(() => {});
          localPose = null;
        }
        if (pool.aliveCount < pool.size)
          notes.push(
            `Only ${pool.aliveCount} of ${pool.size} pose workers started; continuing with reduced parallelism.`,
          );
        poseProvider = `WASM pool (${pool.aliveCount} worker${pool.aliveCount === 1 ? "" : "s"} × ${plan.wasmThreads} thread${plan.wasmThreads === 1 ? "" : "s"})`;
      } catch (e) {
        notes.push(
          `Pose worker pool unavailable (${errText(e)}); using one local WASM pose session.`,
        );
        pool.dispose();
        pool = null;
      }
    }
    if (!poolAlive && !localPose) {
      localPose = await loadModel(poseBytes, job.pose.name, {
        allowWebGpu: false,
        gpuLabel,
        probeDims: (s) =>
          probeDimsFrom(s, [1, 3, job.settings.height, job.settings.width]),
        notes,
      });
    }
    if (!poolAlive && localPose)
      poseProvider = `WASM (${plan.detectorThreads} thread${plan.detectorThreads === 1 ? "" : "s"})`;
  }

  try {
    const detectorIn = tensorShape(d.session.inputMetadata[0]);
    const detectorOut = tensorShape(d.session.outputMetadata[0]);
    // Fail fast on contract mismatches before decoding any frame. In pool
    // mode each pose worker already ran checkPoseContract during init (a
    // bad model kills every worker, pool.ready() rejects, and the local
    // fallback session created above then throws the precise error here).
    checkDetectorContract(detectorIn, detectorOut, job.settings.detector);
    if (localPose) {
      checkPoseContract(
        tensorShape(localPose.session.inputMetadata[0]),
        localPose.session.outputMetadata.map(tensorShape),
        job.settings.format,
        job.settings.width,
        job.settings.height,
      );
    }
    const shape = detectorIn;
    const size = typeof shape[2] === "number" ? shape[2] : 640;
    if (size !== shape[3] && typeof shape[3] === "number")
      throw Error("Detector must have a square NCHW input.");

    // Every stage is timed: which one dominates depends on the machine (see
    // core/perf.ts), and the breakdown is written into the session.
    const timer = new StageTimer();
    timerRef = timer;
    const detectorRaster = new Rasterizer(size, size, job.settings.detector),
      poseRaster = new Rasterizer(
        job.settings.width,
        job.settings.height,
        "pose",
      );
    const poseDims = [1, 3, job.settings.height, job.settings.width] as const;

    // A GPU session can also die *during* a run (device lost, driver reset,
    // out of memory on a large frame). Throwing away thousands of decoded
    // frames is the wrong answer: rebuild that session on WASM, say so, and
    // keep going. Rebuilding needs a fresh input tensor because the failed
    // attempt disposed it, hence the factory argument.
    let detectorSession = d.session;
    let detectorProvider = d.provider;
    let detectorLabel =
      d.provider === "WebGPU" ? (d.gpuLabel ?? "WebGPU") : d.provider;
    const runDetector = async (makeInput: () => ort.Tensor) => {
      try {
        return await timer.timeAsync("detector.infer", () =>
          infer(detectorSession, makeInput()),
        );
      } catch (e) {
        if (detectorProvider !== "WebGPU") throw e;
        notes.push(
          `The GPU detector session failed mid-analysis (${errText(e)}); continuing on WASM for the rest of this run.`,
        );
        send("status", "GPU detector session failed; continuing on WASM.");
        await detectorSession.release().catch(() => {});
        const fallback = await loadModel(detectorBytes, job.detector.name, {
          allowWebGpu: false,
          gpuLabel,
          probeDims: (session) => probeDimsFrom(session, [1, 3, 640, 640]),
          notes,
        });
        detectorSession = fallback.session;
        detectorProvider = "WASM";
        detectorLabel = "WASM";
        return await timer.timeAsync("detector.infer", () =>
          infer(detectorSession, makeInput()),
        );
      }
    };

    // Memoised so concurrent pool failures create exactly one fallback.
    let localPosePromise: Promise<LoadedModel> | null = null;
    const ensureLocalPose = async (): Promise<LoadedModel> => {
      if (localPose) return localPose;
      localPosePromise ??= (async () => {
        const m = await loadModel(poseBytes, job.pose.name, {
          allowWebGpu: false,
          gpuLabel,
          probeDims: (s) =>
            probeDimsFrom(s, [1, 3, job.settings.height, job.settings.width]),
          notes,
        });
        checkPoseContract(
          tensorShape(m.session.inputMetadata[0]),
          m.session.outputMetadata.map(tensorShape),
          job.settings.format,
          job.settings.width,
          job.settings.height,
        );
        return m;
      })();
      localPose = await localPosePromise;
      return localPose;
    };

    // Single-flight downgrade used when the local pose session dies mid-run
    // (device lost, driver reset). TTA passes run concurrently, so several
    // passes can observe the same failure: they all wait on one rebuild
    // instead of each releasing the dead session and loading another model.
    let poseFallback: Promise<LoadedModel> | null = null;
    const fallbackToWasmPose = (dead: LoadedModel): Promise<LoadedModel> => {
      poseFallback ??= (async () => {
        notes.push(
          `The GPU pose session failed mid-analysis; continuing on WASM for the rest of this run. Speed is reduced.`,
        );
        send("status", "GPU pose session failed; continuing on WASM.");
        await dead.session.release().catch(() => {});
        if (localPose === dead) localPose = null;
        localPosePromise = null;
        poseProvider = `WASM (${plan.detectorThreads} thread${plan.detectorThreads === 1 ? "" : "s"} after GPU loss)`;
        return ensureLocalPose();
      })();
      return poseFallback;
    };

    // Frame rasters are shared by every pass of a frame (core/preprocess.ts).
    // Where the browser cannot hand out a bitmap, the coordinator rasterises
    // itself and ships tensors exactly as it always did; that is slower, so it
    // is reported rather than hidden.
    let bitmapFallbackReported = false;

    /**
     * Rasterises one crop locally. The local path reuses one buffer because a
     * local session runs strictly sequentially; a pool tensor job must get a
     * fresh buffer instead, because dispatch *transfers* it.
     */
    const rasterPose = (
      source: FrameSource,
      crop: Box,
      flip: boolean,
      out?: Float32Array,
    ): Float32Array => {
      const t0 = performance.now();
      const data = poseRaster.tensor(
        source.image,
        source.width,
        source.height,
        crop,
        flip,
        out,
      );
      timer.add("pose.preprocess", performance.now() - t0);
      return data;
    };

    /** Runs one crop/TTA pass, on the pool when active, else locally. */
    const posePass = async (
      source: FrameSource,
      crop: Box,
      flip: boolean,
    ): Promise<Point[]> => {
      if (pool && poolAlive) {
        const bitmap = source.bitmapImage;
        if (!bitmap && !bitmapFallbackReported) {
          bitmapFallbackReported = true;
          notes.push(
            "This browser did not provide an ImageBitmap for the decoded frames, so crop rasterisation stays on the coordinator thread and the pose workers receive finished tensors. Parallelism is unaffected; per-pass preprocessing cost is not shared.",
          );
        }
        const jobArg: PoseJob = bitmap
          ? // The worker rasterises this pass: no per-pass cost on this
            // thread, and it happens in parallel across the pool.
            {
              bitmap,
              transfer: false,
              crop,
              flip,
              format: job.settings.format,
            }
          : {
              data: rasterPose(
                source,
                crop,
                flip,
                new Float32Array(poseRaster.pixels),
              ),
              dims: [...poseDims] as [number, number, number, number],
              crop,
              flip,
              format: job.settings.format,
            };
        try {
          return await timer.timeAsync("pose.pass", () =>
            pool!.dispatch(jobArg),
          );
        } catch (e) {
          if (poolAlive) {
            poolAlive = false;
            notes.push(
              `A pose worker failed mid-analysis (${errText(e)}); finishing on the local WASM session. Speed is reduced for the rest of this run.`,
            );
            send(
              "status",
              "Pose worker failed; continuing on the local session.",
            );
          }
          await ensureLocalPose();
        }
      }
      let model = await ensureLocalPose();
      let outputs: Record<string, ort.Tensor>;
      const input = () =>
        new ort.Tensor("float32", rasterPose(source, crop, flip), poseDims);
      try {
        // Timed like the pooled path's worker-reported `pose.infer`, so GPU and
        // CPU runs are comparable in the session report.
        outputs = await timer.timeAsync("pose.infer", () =>
          infer(model.session, input()),
        );
      } catch (e) {
        // A GPU session lost mid-run must not abort the analysis: rebuild it
        // on WASM and redo this pass. A WASM session that fails is a real
        // error and stays fatal.
        if (model.provider !== "WebGPU") throw e;
        model = await fallbackToWasmPose(model);
        outputs = await timer.timeAsync("pose.infer", () =>
          infer(model.session, input()),
        );
      }
      try {
        const decoded = await timer.timeAsync("pose.decode", async () =>
          decodePoseOutputs(
            model.session,
            outputs,
            job.settings.format,
            crop,
            flip,
          ),
        );
        return decoded;
      } finally {
        Object.values(outputs).forEach((t) => t.dispose());
      }
    };

    // Pose passes a run dispatched, and how many the adaptive gate skipped.
    let posePasses = 0,
      posePassesSaved = 0;

    /**
     * Runs a set of passes. Independent passes fan out across the pool; a local
     * session stays strictly sequential because every local pass rasterises
     * into the same reusable buffer and one session cannot overlap runs.
     */
    const runPasses = (
      source: FrameSource,
      passes: PosePass[],
    ): Promise<Point[][]> =>
      pool && poolAlive
        ? Promise.all(passes.map((p) => posePass(source, p.crop, p.flip)))
        : passes.reduce<Promise<Point[][]>>(
            async (acc, p) => [
              ...(await acc),
              await posePass(source, p.crop, p.flip),
            ],
            Promise.resolve([]),
          );

    /**
     * Runs the passes for one detection and fuses them.
     *
     * `always`/`off` dispatch every pass of the plan concurrently (results are
     * reassembled in plan order for fusion). `adaptive` runs the first pass
     * alone and only pays for the rest when that pass is weak; when it does,
     * the same four passes fuse exactly as under `always`.
     */
    const analyseBox = async (
      source: FrameSource,
      box: Box,
    ): Promise<{ box: Box; points: Point[] }> => {
      const passes = passPlan(
        job.settings.tta,
        box,
        job.settings.width,
        job.settings.height,
      );
      let poses: Point[][];
      if (job.settings.tta === "adaptive") {
        const first = await posePass(source, passes[0].crop, passes[0].flip);
        posePasses++;
        if (assessPose(first).accepted) {
          // Skipping the other passes is the whole point of the policy: the
          // first pass already found every body joint at a usable confidence.
          posePassesSaved += passes.length - 1;
          return { box, points: first };
        }
        const rest = await runPasses(source, passes.slice(1));
        posePasses += rest.length;
        poses = [first, ...rest];
      } else {
        poses = await runPasses(source, passes);
        posePasses += poses.length;
      }
      return {
        box,
        points: await timer.timeAsync("fuse", async () => fusePoses(poses)),
      };
    };

    const tracker = new SeatTracker(job.boat),
      frames: PoseFrame[] = [],
      timestamps: number[] = [];
    send(
      "status",
      `Decoding every frame · detecting people · estimating 133 joints · ${describeTta(job.settings.tta)} · pose via ${poseProvider}`,
    );
    let lastPreview = -Infinity,
      processed = 0,
      totalSamples = 0;
    const startedAt = performance.now();

    /**
     * Pose work for one frame, in decode order.
     *
     * Detection and pose are only staged when they use different hardware: with
     * a worker pool the detector session and the pool are busy at the same time
     * instead of taking turns, and the tracker's order requirement is met by a
     * FIFO consumer. Without a pool (WebGPU, or a failed pool) both stages share
     * one session on one thread, so overlapping them would buy nothing and hold
     * extra decoded frames in memory — those runs process each frame in place,
     * exactly as before.
     */
    const processFrame = async ({
      source,
      time,
      boxes,
    }: {
      source: FrameSource;
      time: number;
      boxes: Box[];
    }): Promise<void> => {
      try {
        const detections: { box: Box; points: Point[] }[] = [];
        if (boxes.length) {
          await timer.timeAsync("pose", async () => {
            if (pool && poolAlive) {
              detections.push(
                ...(await Promise.all(
                  boxes.map((box) => analyseBox(source, box)),
                )),
              );
            } else {
              // A local session is single-threaded: passes stay sequential.
              for (const box of boxes)
                detections.push(await analyseBox(source, box));
            }
          });
        }
        const tracked = await timer.timeAsync("track", async () =>
          tracker.update(detections, time),
        );
        frames.push(...tracked);
        processed++;
        if (totalSamples)
          send("progress", 0.02 + 0.92 * Math.min(1, processed / totalSamples));
        // Bound UI work and transfer only the latest frame, never the history.
        const now = performance.now();
        if (now - lastPreview >= 200) {
          lastPreview = now;
          await timer.timeAsync("preview", async () => {
            const width = source.width,
              height = source.height;
            const scale = Math.min(1, 960 / width);
            const preview = new OffscreenCanvas(
              Math.max(1, Math.round(width * scale)),
              Math.max(1, Math.round(height * scale)),
            );
            preview
              .getContext("2d")!
              .drawImage(source.image, 0, 0, preview.width, preview.height);
            const bitmap = preview.transferToImageBitmap();
            postMessage(
              {
                type: "preview",
                payload: {
                  bitmap,
                  width,
                  height,
                  time,
                  processed,
                  frames: tracked,
                },
              },
              [bitmap],
            );
          });
        }
      } finally {
        source.dispose();
      }
    };

    const pipeline = new SequentialPipeline<{
      source: FrameSource;
      time: number;
      boxes: Box[];
    }>({
      capacity: 3,
      onDrop: (item) => item.source.dispose(),
      handler: processFrame,
    });
    /** True while the pose stage can work on the next frame while the detector
     * works on the previous one. */
    const canOverlap = () => Boolean(pool && poolAlive);

    const meta = await decodeVideo(
      job.file,
      async (frame, time, info) => {
        totalSamples = info.samples.length;
        if (timestamps.length > 20000)
          throw Error(
            "Clip exceeds the 20,000-frame memory guard. Analyze shorter clips.",
          );
        // The clone outlives this callback: the decoder closes its own frame
        // as soon as the visitor returns, while pose work for this frame is
        // still queued or in flight.
        const source = await FrameSource.own(frame.clone());
        timestamps.push(time);
        try {
          const out = await timer.timeAsync("detector", async () =>
            runDetector(() => {
              const t0 = performance.now();
              const data = detectorRaster.tensor(
                source.image,
                source.width,
                source.height,
                null,
                false,
              );
              timer.add("detector.preprocess", performance.now() - t0);
              return new ort.Tensor("float32", data, [1, 3, size, size]);
            }),
          );
          let boxes: Box[];
          try {
            const raw = out[detectorSession.outputNames[0]];
            const ratio = Math.min(size / source.width, size / source.height);
            boxes = (
              job.settings.detector === "yolox"
                ? decodeYolox(raw.data as Float32Array, raw.dims, size, ratio)
                : decodeYolo(raw.data as Float32Array, raw.dims, ratio)
            )
              .filter((b) => {
                const x = ((b.x + b.width / 2) / source.width) * 100,
                  y = ((b.y + b.height / 2) / source.height) * 100,
                  r = job.settings.region;
                return (
                  b.width > 15 &&
                  b.height > 30 &&
                  x >= r.left &&
                  x <= r.right &&
                  y >= r.top &&
                  y <= r.bottom
                );
              })
              .slice(0, 12);
          } finally {
            Object.values(out).forEach((t) => t.dispose());
          }
          // Staged mode blocks only once `capacity` frames are queued or in
          // flight, so a slow pose stage throttles the decoder instead of
          // growing memory; in-place mode simply awaits the work.
          if (canOverlap()) await pipeline.push({ source, time, boxes });
          else await processFrame({ source, time, boxes });
        } catch (e) {
          source.dispose();
          throw e;
        }
      },
      // Progress follows *processed* frames (the pose stage), not decoded
      // ones: decoding runs ahead of the pipeline by design.
      () => {},
    );
    await pipeline.drain();
    send("status", "Recovering short gaps and segmenting stroke cycles…");
    const recovered = await timer.timeAsync("recover", async () =>
      recoverOcclusion(frames),
    );
    const strokes = await timer.timeAsync("strokes", async () =>
      segmentStrokes(recovered),
    );
    const seconds = (performance.now() - startedAt) / 1000;
    const performance_ = buildReport(timer, {
      frames: processed,
      seconds,
      posePasses,
      posePassesSaved,
    });
    notes.push(`Performance: ${performance_.summary}`);
    send("progress", 0.98);
    const session: Session = {
      version: 1,
      name: job.file.name,
      demo: false,
      width: meta.width,
      height: meta.height,
      duration: meta.duration,
      timestamps: timestamps.sort((a, b) => a - b),
      boat: job.boat,
      frames: recovered,
      strokes,
      performance: performance_,
      camera: estimateCamera(
        recovered.slice(0, job.boat.seats).map((f) => f.points[11]),
      ),
      corrections: [],
      warnings: [
        ...(timestamps.length !== meta.samples.length
          ? [
              `Decoder emitted ${timestamps.length} of ${meta.samples.length} encoded samples; missing frames remain absent. Timing resolution is limited by the actual decoded timestamps.`,
            ]
          : []),
        ...notes,
        ...(plan.crossOriginIsolated
          ? []
          : [
              "Cross-origin isolation is off: each WASM session runs single-threaded and parallelism comes from the pose-worker pool. Serving with COOP/COEP headers (on GitHub Pages the bundled isolation service worker does this after one reload) unlocks threaded WASM per session.",
            ]),
        "Experimental tracking: review seat identity and keypoints before coaching.",
        "Catch/finish events are hip-to-ankle cycle proxies, not directly observed blade entry/release.",
        "3D camera normalization, oar tracking, and depth-dependent biomechanics are not validated and are withheld.",
      ],
      model: `${job.detector.name} + ${job.pose.name}; ${job.settings.width}×${job.settings.height}; ${describeTta(job.settings.tta)}`,
      // Provider strings carry the adapter description when the GPU was used,
      // so an exported session records what actually ran where.
      provider: `${detectorLabel} detector / ${poseProvider} pose`,
    };
    send("result", session);
  } finally {
    pool?.dispose();
    activePool = null;
    timerRef = null;
    await d.session.release().catch(() => {});
    if (localPose) await localPose.session.release();
  }
}

self.onmessage = (event: MessageEvent<Job | { type: string }>) => {
  const data = event.data as { type?: string; index?: number; error?: string };
  if (data?.type === "pool-port") {
    const port = (event as MessageEvent).ports?.[0];
    if (port) poolPorts.push(port);
    return;
  }
  if (data?.type === "pool-error") {
    if (typeof data.index === "number")
      activePool?.handleError(
        data.index,
        new Error(data.error || "Pose worker error."),
      );
    return;
  }
  run(event.data as Job).catch((e) =>
    send("error", e instanceof Error ? e.message : String(e)),
  );
};
