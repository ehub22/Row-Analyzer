import * as ort from "onnxruntime-web/webgpu";
import wasmUrl from "../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.wasm?url";
import wasmModule from "../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.mjs?url";
import { decodeVideo } from "../core/video";
import { decodeYolo, decodeYolox, paddedCrop } from "../core/detection";
import { decodeSimCC, decodeHeatmap, unflip, fusePoses } from "../core/pose";
import { SeatTracker } from "../core/tracking";
import { recoverOcclusion } from "../core/occlusion";
import { segmentStrokes } from "../core/strokes";
import { estimateCamera } from "../core/geometry";
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
}
const send = (type: string, payload: unknown) => postMessage({ type, payload });
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = { wasm: wasmUrl, mjs: wasmModule };
async function model(file: File, forceWasm: boolean) {
  const bytes = await file.arrayBuffer();
  if (!forceWasm)
    try {
      return {
        session: await ort.InferenceSession.create(bytes, {
          executionProviders: ["webgpu"],
        }),
        provider: "WebGPU",
      };
    } catch (e) {
      send(
        "status",
        `WebGPU unavailable for ${file.name}; retrying WASM. ${String(e).slice(0, 100)}`,
      );
    }
  return {
    session: await ort.InferenceSession.create(bytes, {
      executionProviders: ["wasm"],
    }),
    provider: "WASM",
  };
}
const canvas = new OffscreenCanvas(640, 640);
const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
function tensor(
  frame: VideoFrame,
  w: number,
  h: number,
  crop: Box | null,
  flip: boolean,
  mode: "yolo" | "yolox" | "pose",
) {
  canvas.width = w;
  canvas.height = h;
  ctx.fillStyle = mode === "pose" ? "black" : "rgb(114,114,114)";
  ctx.fillRect(0, 0, w, h);
  ctx.save();
  if (flip) {
    ctx.translate(w, 0);
    ctx.scale(-1, 1);
  }
  if (crop) {
    ctx.drawImage(frame, crop.x, crop.y, crop.width, crop.height, 0, 0, w, h);
  } else {
    const ratio = Math.min(w / frame.displayWidth, h / frame.displayHeight);
    ctx.drawImage(
      frame,
      0,
      0,
      frame.displayWidth * ratio,
      frame.displayHeight * ratio,
    );
  }
  ctx.restore();
  const pixels = ctx.getImageData(0, 0, w, h).data,
    data = new Float32Array(3 * w * h);
  for (let i = 0; i < w * h; i++)
    for (let c = 0; c < 3; c++) {
      const p = pixels[i * 4 + (mode === "yolox" ? 2 - c : c)];
      data[c * w * h + i] =
        mode === "pose"
          ? (p - [123.675, 116.28, 103.53][c]) / [58.395, 57.12, 57.375][c]
          : mode === "yolo"
            ? p / 255
            : p;
    }
  return new ort.Tensor("float32", data, [1, 3, h, w]);
}
async function infer(session: ort.InferenceSession, input: ort.Tensor) {
  try {
    return await session.run({ [session.inputNames[0]]: input });
  } finally {
    input.dispose();
  }
}
async function run(job: Job) {
  send("status", "Loading detector and whole-body pose model…");
  const d = await model(job.detector, job.settings.wasm);
  let p: Awaited<ReturnType<typeof model>> | undefined;
  try {
    p = await model(job.pose, job.settings.wasm);
    const poseSession = p.session;
    const detectorMeta = d.session.inputMetadata[0];
    const shape = detectorMeta.isTensor ? detectorMeta.shape : [];
    const size = typeof shape[2] === "number" ? shape[2] : 640;
    if (size !== shape[3] && typeof shape[3] === "number")
      throw Error("Detector must have a square NCHW input.");
    const tracker = new SeatTracker(job.boat),
      frames: PoseFrame[] = [],
      timestamps: number[] = [];
    send(
      "status",
      "Decoding every frame · detecting people · estimating 133 joints",
    );
    const meta = await decodeVideo(
      job.file,
      async (frame, time) => {
        if (timestamps.length > 20000)
          throw Error(
            "Clip exceeds the 20,000-frame memory guard. Analyze shorter clips.",
          );
        const out = await infer(
          d.session,
          tensor(frame, size, size, null, false, job.settings.detector),
        );
        let boxes: Box[];
        try {
          const raw = out[d.session.outputNames[0]];
          const ratio = Math.min(
            size / frame.displayWidth,
            size / frame.displayHeight,
          );
          boxes = (
            job.settings.detector === "yolox"
              ? decodeYolox(raw.data as Float32Array, raw.dims, size, ratio)
              : decodeYolo(raw.data as Float32Array, raw.dims, ratio)
          )
            .filter((b) => {
              const x = ((b.x + b.width / 2) / frame.displayWidth) * 100,
                y = ((b.y + b.height / 2) / frame.displayHeight) * 100,
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
        const detections: { box: Box; points: Point[] }[] = [];
        for (const box of boxes) {
          const poses: Point[][] = [];
          for (const scale of job.settings.tta ? [1.25, 1.5] : [1.25])
            for (const flip of job.settings.tta ? [false, true] : [false]) {
              const crop = paddedCrop(
                box,
                job.settings.width,
                job.settings.height,
                scale,
              );
              const outputs = await infer(
                poseSession,
                tensor(
                  frame,
                  job.settings.width,
                  job.settings.height,
                  crop,
                  flip,
                  "pose",
                ),
              );
              try {
                const vals = poseSession.outputNames.map((n) => outputs[n]);
                let points: Point[];
                if (job.settings.format === "simcc") {
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
                poses.push(flip ? unflip(points, crop) : points);
              } finally {
                Object.values(outputs).forEach((t) => t.dispose());
              }
            }
          detections.push({ box, points: fusePoses(poses) });
        }
        frames.push(...tracker.update(detections, time));
        timestamps.push(time);
      },
      (v) => send("progress", v * 0.94),
    );
    send("status", "Recovering short gaps and segmenting stroke cycles…");
    const recovered = recoverOcclusion(frames);
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
      strokes: segmentStrokes(recovered),
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
        "Experimental tracking: review seat identity and keypoints before coaching.",
        "Catch/finish events are hip-to-ankle cycle proxies, not directly observed blade entry/release.",
        "3D camera normalization, oar tracking, and depth-dependent biomechanics are not validated and are withheld.",
      ],
      model: `${job.detector.name} + ${job.pose.name}; ${job.settings.width}×${job.settings.height}; ${job.settings.tta ? "flip + multi-scale TTA" : "single pass"}`,
      provider: `${d.provider} / ${p.provider}`,
    };
    send("result", session);
  } finally {
    await d.session.release();
    if (p) await p.session.release();
  }
}
self.onmessage = (event: MessageEvent<Job>) => {
  run(event.data).catch((e) =>
    send("error", e instanceof Error ? e.message : String(e)),
  );
};
