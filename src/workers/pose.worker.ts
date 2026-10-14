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
//
// Protocol (over the MessagePort), implemented by core/pool.ts:
//   in   { type: "init", poseBytes, format, width, height, wasmThreads }
//        { type: "pose", id, data, dims, crop, flip, format }
//   out  { type: "ready" } | { type: "init-error", error }
//        { type: "pose-result", id, points } | { type: "pose-error", id, error }

import * as ort from "onnxruntime-web/webgpu";
import wasmUrl from "../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.wasm?url";
import wasmModule from "../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.mjs?url";
import { decodeSimCC, decodeHeatmap, unflip } from "../core/pose";
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
  data: Float32Array;
  dims: [number, number, number, number];
  crop: Box;
  flip: boolean;
  format: "simcc" | "heatmap";
}

let session: ort.InferenceSession | null = null;
let port: MessagePort | null = null;

const reply = (message: unknown) => port?.postMessage(message);
const errText = (e: unknown) =>
  (e instanceof Error ? e.message : String(e)).slice(0, 500);

async function init(msg: InitMessage): Promise<void> {
  try {
    // Threaded WASM only engages when the page is cross-origin isolated;
    // core/parallel.ts already caps this at 1 otherwise, and ORT would
    // silently clamp it anyway. Set before the first session is created.
    ort.env.wasm.numThreads = Math.max(1, Math.floor(msg.wasmThreads) || 1);
    ort.env.wasm.wasmPaths = { wasm: wasmUrl, mjs: wasmModule };
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
    reply({ type: "ready" });
  } catch (e) {
    reply({ type: "init-error", error: errText(e) });
  }
}

async function runPose(msg: PoseMessage): Promise<void> {
  if (!session) {
    reply({ type: "pose-error", id: msg.id, error: "Pose session not initialised." });
    return;
  }
  const input = new ort.Tensor("float32", msg.data, msg.dims);
  try {
    const outputs = await session.run({ [session.inputNames[0]]: input });
    try {
      const vals = session.outputNames.map((n) => outputs[n]);
      let points: Point[];
      if (msg.format === "simcc") {
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
          msg.crop,
        );
      } else {
        if (vals[0].dims.length !== 4 || vals[0].dims[1] !== 133)
          throw Error("Expected [1,133,H,W] heatmaps.");
        points = decodeHeatmap(
          vals[0].data as Float32Array,
          vals[0].dims[3],
          vals[0].dims[2],
          msg.crop,
        );
      }
      reply({
        type: "pose-result",
        id: msg.id,
        points: msg.flip ? unflip(points, msg.crop) : points,
      });
    } finally {
      Object.values(outputs).forEach((t) => t.dispose());
    }
  } catch (e) {
    reply({ type: "pose-error", id: msg.id, error: errText(e) });
  } finally {
    input.dispose();
  }
}

self.onmessage = (event: MessageEvent<{ type: string }>) => {
  // The MessagePort arrives as a transferred port (event.ports), not in data.
  const connected = event.ports?.[0];
  if (event.data?.type !== "connect" || !connected) return;
  port = connected;
  port.onmessage = (e: MessageEvent<InitMessage | PoseMessage>) => {
    const msg = e.data;
    if (msg.type === "init") void init(msg);
    else if (msg.type === "pose") void runPose(msg);
  };
  port.onmessageerror = () =>
    reply({ type: "init-error", error: "Pose worker port failed." });
};
