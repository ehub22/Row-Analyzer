// Pose-only inference worker: one node of the parallel pose pool.
//
// The main thread spawns N of these and hands each a MessagePort connected
// to the analysis (coordinator) worker, so tensors travel worker→worker
// without bouncing through the UI thread. This worker owns exactly one WASM
// ONNX session for the whole-body pose model. WebGPU is deliberately not
// attempted here: the pool exists for the (common) case where WebGPU is
// unavailable or fails, and N CPU sessions scale across cores without any
// cross-origin-isolation requirement. If WebGPU does work, the coordinator
// keeps pose inference on the GPU itself and never initialises this pool.
// For that reason this worker imports the WASM-only entry point and pins the
// plain WASM build (no WebGPU bridge, no asyncify overhead) — see
// core/ortRuntime.ts for why the entry point and the WASM build must match.
//
// Two properties of this worker are what make the pool fast:
//
//  * **It rasterises its own crops.** The coordinator sends the frame's shared
//    ImageBitmap plus a crop box; drawing and normalizing a crop used to happen
//    serially on the coordinator (measured at ~20 ms per crop against
//    sub-millisecond postprocessing) while every worker waited for work.
//  * **It reuses its canvas and input buffer, and queues jobs.** One
//    whole-body pass per message would otherwise reallocate ~1.3 MB of tensor
//    per pass, and overlapping runs of one session on one buffer would corrupt
//    the input; jobs are therefore processed strictly one at a time.
//
// Protocol (over the MessagePort), implemented by core/pool.ts:
//   in   { type: "init", poseFile, format, width, height, wasmThreads }
//        { type: "pose", id, data, dims, crop, flip, format }
//        { type: "pose", id, bitmap, crop, flip, format }
//   out  { type: "ready" } | { type: "init-error", error }
//        { type: "pose-result", id, points, timings } | { type: "pose-error", id, error }

import * as ort from "onnxruntime-web/wasm";
import {
  configureOrtRuntime,
  ortWasmOnlyVariant,
} from "../core/ortRuntime";
import { decodeSimCC, decodeHeatmap, unflip } from "../core/pose";
import { Rasterizer } from "../core/preprocess";
import { checkPoseContract } from "../core/compat";
import type { Box, Point } from "../core/types";

interface InitMessage {
  type: "init";
  /** Blob reference; each worker reads its own copy of the model bytes. */
  poseFile: File;
  format: "simcc" | "heatmap";
  width: number;
  height: number;
  wasmThreads: number;
}
interface PoseMessage {
  type: "pose";
  id: number;
  /** Pre-rasterised tensor, when the coordinator could not share a bitmap. */
  data?: Float32Array;
  dims?: [number, number, number, number];
  /** Frame bitmap to rasterise here; the fast path. */
  bitmap?: ImageBitmap;
  crop: Box;
  flip: boolean;
  format: "simcc" | "heatmap";
}

/** Per-pass milliseconds, so the coordinator can report where time went. */
interface PassTimings {
  preprocess: number;
  infer: number;
  decode: number;
}

let session: ort.InferenceSession | null = null;
let port: MessagePort | null = null;
let rasterizer: Rasterizer | null = null;
let init: { format: "simcc" | "heatmap"; width: number; height: number } | null =
  null;

const reply = (message: unknown) => port?.postMessage(message);
const errText = (e: unknown) =>
  (e instanceof Error ? e.message : String(e)).slice(0, 500);

async function setup(msg: InitMessage): Promise<void> {
  try {
    // Threaded WASM only engages when the page is cross-origin isolated;
    // core/parallel.ts already caps this at 1 otherwise, and ORT would
    // silently clamp it anyway. Set before the first session is created.
    configureOrtRuntime(ort.env, ortWasmOnlyVariant, msg.wasmThreads);
    session = await ort.InferenceSession.create(
      await msg.poseFile.arrayBuffer(),
      { executionProviders: ["wasm"] },
    );
    const shape = (m: { isTensor: boolean; shape?: readonly (number | string)[] }) =>
      m.isTensor && m.shape ? m.shape : [];
    // Same fail-fast contract check the single-session path performs.
    checkPoseContract(
      shape(session.inputMetadata[0]),
      session.outputMetadata.map(shape),
      msg.format,
      msg.width,
      msg.height,
    );
    rasterizer = new Rasterizer(msg.width, msg.height, "pose");
    init = { format: msg.format, width: msg.width, height: msg.height };
    reply({ type: "ready" });
  } catch (e) {
    reply({ type: "init-error", error: errText(e) });
  }
}

function decodePoints(
  outputs: Record<string, ort.Tensor>,
  format: "simcc" | "heatmap",
  crop: Box,
): Point[] {
  const vals = session!.outputNames.map((n) => outputs[n]);
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
    return decodeSimCC(
      vals[0].data as Float32Array,
      vals[1].data as Float32Array,
      vals[0].dims[2],
      vals[1].dims[2],
      crop,
    );
  }
  if (vals[0].dims.length !== 4 || vals[0].dims[1] !== 133)
    throw Error("Expected [1,133,H,W] heatmaps.");
  return decodeHeatmap(
    vals[0].data as Float32Array,
    vals[0].dims[3],
    vals[0].dims[2],
    crop,
  );
}

async function runPose(msg: PoseMessage): Promise<void> {
  if (!session || !init) {
    reply({ type: "pose-error", id: msg.id, error: "Pose session not initialised." });
    return;
  }
  const timings: PassTimings = { preprocess: 0, infer: 0, decode: 0 };
  let input: ort.Tensor | null = null;
  try {
    let t0 = performance.now();
    if (msg.bitmap) {
      const data = rasterizer!.tensor(
        msg.bitmap,
        msg.bitmap.width,
        msg.bitmap.height,
        msg.crop,
        msg.flip,
      );
      input = new ort.Tensor("float32", data, [1, 3, init.height, init.width]);
    } else if (msg.data && msg.dims) {
      input = new ort.Tensor("float32", msg.data, msg.dims);
    } else {
      throw Error("Pose job carried neither a bitmap nor a tensor.");
    }
    timings.preprocess = performance.now() - t0;

    t0 = performance.now();
    const outputs = await session.run({ [session.inputNames[0]]: input });
    timings.infer = performance.now() - t0;
    try {
      t0 = performance.now();
      const points = decodePoints(outputs, msg.format, msg.crop);
      timings.decode = performance.now() - t0;
      reply({
        type: "pose-result",
        id: msg.id,
        points: msg.flip ? unflip(points, msg.crop) : points,
        timings,
      });
    } finally {
      Object.values(outputs).forEach((t) => t.dispose());
    }
  } catch (e) {
    reply({ type: "pose-error", id: msg.id, error: errText(e) });
  } finally {
    // The batch dimension of a transferred tensor is the caller's, but a
    // received bitmap handle is ours to release.
    msg.bitmap?.close();
    input?.dispose();
  }
}

// Jobs are serialised: one session, one canvas, one input buffer.
const queue: PoseMessage[] = [];
let busy = false;

async function drainQueue(): Promise<void> {
  if (busy) return;
  busy = true;
  try {
    while (queue.length) await runPose(queue.shift()!);
  } finally {
    busy = false;
  }
}

self.onmessage = (event: MessageEvent<{ type: string }>) => {
  // The MessagePort arrives as a transferred port (event.ports), not in data.
  const connected = event.ports?.[0];
  if (event.data?.type !== "connect" || !connected) return;
  port = connected;
  port.onmessage = (e: MessageEvent<InitMessage | PoseMessage>) => {
    const msg = e.data;
    if (msg.type === "init") void setup(msg);
    else if (msg.type === "pose") {
      queue.push(msg);
      void drainQueue();
    }
  };
  port.onmessageerror = () =>
    reply({ type: "init-error", error: "Pose worker port failed." });
};
