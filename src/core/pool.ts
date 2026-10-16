// Least-busy dispatcher for the pool of pose-inference workers.
//
// The coordinator (analysis worker) owns an array of endpoints — normally
// MessagePorts wired directly to pose workers by the main thread, so job
// payloads never bounce through the UI thread. This class only understands
// a minimal message protocol, which keeps it unit-testable with fakes:
//
//   outbound  { type: "init", ... }              (sent via init())
//             { type: "pose", id, ...job }        (sent via dispatch())
//   inbound   { type: "ready" }                   (answers "init")
//             { type: "init-error", error }       (answers "init")
//             { type: "pose-result", id, points } (answers "pose")
//             { type: "pose-error", id, error }   (answers "pose")
//
// Failure semantics: an endpoint that fails init or errors mid-job is
// marked dead and its in-flight jobs reject, so the caller can redo them on
// a surviving endpoint or on its own local session. One worker dying must
// never take the whole analysis down with it.

import type { Box, Point } from "./types";

interface PoseJobBase {
  crop: Box;
  flip: boolean;
  format: "simcc" | "heatmap";
}

/**
 * One crop/TTA pass, either already rasterised by the coordinator
 * (`data`, transferred) or handed over as a frame bitmap that the worker
 * rasterises itself.
 *
 * The bitmap form exists because rasterising a crop is expensive and was
 * previously done serially on the coordinator while every pooled worker sat
 * idle: profiling a 288×384 pose crop showed ~20 ms to draw a VideoFrame and
 * ~9 ms to draw the frame's shared ImageBitmap, against sub-millisecond pose
 * postprocessing. Sending the bitmap moves that cost into the pool, where it
 * runs in parallel and off the coordinator's critical path.
 */
export interface TensorPoseJob extends PoseJobBase {
  data: Float32Array;
  dims: readonly [number, number, number, number];
}

export interface BitmapPoseJob extends PoseJobBase {
  /** Frame bitmap or a pixel-identical subrectangle transferred to the worker. */
  bitmap: ImageBitmap;
  transfer: boolean;
  /** Crop coordinates in bitmap space; `crop` remains in full-frame space for decoding. */
  rasterCrop?: Box;
}

export type PoseJob = TensorPoseJob | BitmapPoseJob;

export interface PoolEndpoint {
  /** Sends a message, optionally transferring ownership of buffers. */
  send(message: unknown, transfer?: Transferable[]): void;
}

interface Pending {
  endpoint: number;
  resolve: (points: Point[]) => void;
  reject: (error: Error) => void;
}

/** Per-pass milliseconds measured inside a pose worker. */
export interface PassTimings {
  preprocess?: number;
  infer?: number;
  decode?: number;
}

export interface PoseWorkerPoolOptions {
  /**
   * Called for every completed pass with the worker's own timing breakdown.
   * Workers sum these independently, so the caller reports per-pass averages
   * rather than a share of wall time.
   */
  onPassTimings?: (timings: PassTimings) => void;
}

interface EndpointState {
  ready: boolean;
  dead: boolean;
  error?: string;
}

const toError = (v: unknown): Error =>
  v instanceof Error ? v : new Error(typeof v === "string" ? v : String(v));

/** Chrome's list of transferable types, checked without assuming ImageBitmap. */
function isTransferable(value: unknown): value is Transferable {
  return typeof ImageBitmap !== "undefined" && value instanceof ImageBitmap;
}

export class PoseWorkerPool {
  private endpoints: PoolEndpoint[];
  private state: EndpointState[];
  private pending = new Map<number, Pending>();
  private inflight: number[];
  private nextId = 1;
  private expectedJoints: 17 | 133 | null = null;
  private disposed = false;
  private readyWaiters: {
    resolve: () => void;
    reject: (e: Error) => void;
  }[] = [];
  private onPassTimings?: (timings: PassTimings) => void;

  constructor(endpoints: PoolEndpoint[], options: PoseWorkerPoolOptions = {}) {
    this.endpoints = endpoints;
    this.state = endpoints.map(() => ({ ready: false, dead: false }));
    this.inflight = endpoints.map(() => 0);
    this.onPassTimings = options.onPassTimings;
  }

  get size(): number {
    return this.endpoints.length;
  }

  /** Endpoints that finished init and have not failed since. */
  get aliveCount(): number {
    return this.state.filter((s) => s.ready && !s.dead).length;
  }

  /** Endpoints that failed init or died mid-run. */
  get deadCount(): number {
    return this.state.filter((s) => s.dead).length;
  }

  /** Endpoints whose fate is known (ready or dead). */
  get settledCount(): number {
    return this.aliveCount + this.deadCount;
  }

  /** Sends the init payload to every endpoint. Call exactly once. */
  init(payload: Record<string, unknown>): void {
    if (payload.joints === 17 || payload.joints === 133)
      this.expectedJoints = payload.joints;
    this.endpoints.forEach((ep, i) => {
      try {
        ep.send({ type: "init", ...payload });
      } catch (e) {
        this.markDead(i, toError(e));
      }
    });
  }

  /**
   * Resolves once at least one endpoint is ready. Rejects with the first
   * init error if every endpoint dies, or on dispose().
   */
  ready(): Promise<void> {
    if (this.aliveCount > 0) return Promise.resolve();
    if (this.state.length && this.state.every((s) => s.dead))
      return Promise.reject(
        new Error(
          `All pose workers failed to start: ${this.state.find((s) => s.error)?.error ?? "unknown error"}`,
        ),
      );
    if (this.disposed)
      return Promise.reject(new Error("Pose worker pool was disposed."));
    return new Promise((resolve, reject) =>
      this.readyWaiters.push({ resolve, reject }),
    );
  }

  /** Routes one pose job to the least-busy live endpoint. */
  dispatch(job: PoseJob): Promise<Point[]> {
    if (this.disposed)
      return Promise.reject(new Error("Pose worker pool was disposed."));
    const target = this.pickEndpoint();
    if (target < 0)
      return Promise.reject(
        new Error("No pose worker is available (all failed to start)."),
      );
    const id = this.nextId++;
    return new Promise<Point[]>((resolve, reject) => {
      this.pending.set(id, { endpoint: target, resolve, reject });
      this.inflight[target]++;
      const bitmapJob = "bitmap" in job ? job : null;
      const message = bitmapJob
        ? {
            type: "pose",
            id,
            bitmap: bitmapJob.bitmap,
            crop: job.crop,
            rasterCrop: bitmapJob.rasterCrop,
            flip: job.flip,
            format: job.format,
          }
        : {
            type: "pose",
            id,
            data: (job as TensorPoseJob).data,
            dims: [...(job as TensorPoseJob).dims],
            crop: job.crop,
            flip: job.flip,
            format: job.format,
          };
      // A newly created bitmap handle is transferred without cloning pixels;
      // serializing the original without transfer remains a compatibility
      // fallback. Tensor buffers are transferred as well.
      const transfer: Transferable[] = bitmapJob
        ? bitmapJob.transfer && isTransferable(bitmapJob.bitmap)
          ? [bitmapJob.bitmap]
          : []
        : [(job as TensorPoseJob).data.buffer as ArrayBuffer];
      try {
        this.endpoints[target].send(message, transfer);
      } catch (e) {
        this.pending.delete(id);
        this.inflight[target]--;
        reject(toError(e));
      }
    });
  }

  /** Feeds an inbound endpoint message into the pool. */
  handleMessage(index: number, message: unknown): void {
    const s = this.state[index];
    if (!s || typeof message !== "object" || message === null) return;
    const msg = message as {
      type?: string;
      id?: number;
      joints?: number;
      points?: Point[];
      timings?: PassTimings;
      error?: unknown;
    };
    if (msg.type === "ready") {
      if (!s.dead) {
        if (this.expectedJoints && msg.joints !== this.expectedJoints) {
          this.markDead(
            index,
            new Error(
              `Pose worker reported ${msg.joints ?? "unknown"} joints; expected ${this.expectedJoints}.`,
            ),
          );
          return;
        }
        s.ready = true;
        this.settleReadyWaiters();
      }
      return;
    }
    if (msg.type === "init-error") {
      s.error =
        typeof msg.error === "string" ? msg.error : String(msg.error ?? "");
      this.markDead(index, toError(msg.error));
      return;
    }
    if (msg.type === "pose-result" || msg.type === "pose-error") {
      const p =
        typeof msg.id === "number" ? this.pending.get(msg.id) : undefined;
      if (!p || p.endpoint !== index) return;
      this.pending.delete(msg.id!);
      this.inflight[index]--;
      if (msg.type === "pose-result") {
        if (msg.timings) this.onPassTimings?.(msg.timings);
        p.resolve(msg.points ?? []);
      } else p.reject(toError(msg.error));
    }
  }

  /** Feeds a transport-level endpoint failure (port/worker error event). */
  handleError(index: number, error: unknown): void {
    const s = this.state[index];
    if (s && !s.error) s.error = toError(error).message;
    this.markDead(index, toError(error));
  }

  /** Rejects everything outstanding and refuses further work. */
  dispose(): void {
    this.disposed = true;
    const error = new Error("Pose worker pool was disposed.");
    for (const p of this.pending.values()) p.reject(error);
    this.pending.clear();
    this.inflight = this.inflight.map(() => 0);
    this.settleReadyWaiters(error);
  }

  private pickEndpoint(): number {
    let best = -1;
    for (let i = 0; i < this.endpoints.length; i++) {
      if (!this.state[i].ready || this.state[i].dead) continue;
      if (best < 0 || this.inflight[i] < this.inflight[best]) best = i;
    }
    return best;
  }

  private markDead(index: number, error: Error): void {
    const s = this.state[index];
    if (!s || s.dead) return;
    s.dead = true;
    s.ready = false;
    if (!s.error) s.error = error.message;
    for (const [id, p] of this.pending) {
      if (p.endpoint === index) {
        this.pending.delete(id);
        p.reject(error);
      }
    }
    this.inflight[index] = 0;
    this.settleReadyWaiters();
  }

  private settleReadyWaiters(error?: Error): void {
    if (!this.readyWaiters.length) return;
    if (error) {
      for (const w of this.readyWaiters.splice(0)) w.reject(error);
      return;
    }
    if (this.aliveCount > 0) {
      for (const w of this.readyWaiters.splice(0)) w.resolve();
    } else if (this.state.every((s) => s.dead)) {
      const first = this.state.find((s) => s.error)?.error;
      for (const w of this.readyWaiters.splice(0))
        w.reject(
          new Error(
            `All pose workers failed to start: ${first ?? "unknown error"}`,
          ),
        );
    }
  }
}
