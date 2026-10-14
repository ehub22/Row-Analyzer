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

export interface PoseJob {
  /** CHW float tensor for one crop/TTA pass (transferred, not cloned). */
  data: Float32Array;
  dims: readonly [number, number, number, number];
  crop: Box;
  flip: boolean;
  format: "simcc" | "heatmap";
}

export interface PoolEndpoint {
  /** Sends a message, optionally transferring ownership of buffers. */
  send(message: unknown, transfer?: Transferable[]): void;
}

interface Pending {
  endpoint: number;
  resolve: (points: Point[]) => void;
  reject: (error: Error) => void;
}

interface EndpointState {
  ready: boolean;
  dead: boolean;
  error?: string;
}

const toError = (v: unknown): Error =>
  v instanceof Error ? v : new Error(typeof v === "string" ? v : String(v));

export class PoseWorkerPool {
  private endpoints: PoolEndpoint[];
  private state: EndpointState[];
  private pending = new Map<number, Pending>();
  private inflight: number[];
  private nextId = 1;
  private disposed = false;
  private readyWaiters: {
    resolve: () => void;
    reject: (e: Error) => void;
  }[] = [];

  constructor(endpoints: PoolEndpoint[]) {
    this.endpoints = endpoints;
    this.state = endpoints.map(() => ({ ready: false, dead: false }));
    this.inflight = endpoints.map(() => 0);
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
      try {
        this.endpoints[target].send(
          {
            type: "pose",
            id,
            data: job.data,
            dims: [...job.dims],
            crop: job.crop,
            flip: job.flip,
            format: job.format,
          },
          // Transfer the tensor backing store: zero-copy, and the caller
          // must build a fresh tensor per concurrent pass anyway.
          [job.data.buffer as ArrayBuffer],
        );
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
      points?: Point[];
      error?: unknown;
    };
    if (msg.type === "ready") {
      if (!s.dead) {
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
      const p = typeof msg.id === "number" ? this.pending.get(msg.id) : undefined;
      if (!p || p.endpoint !== index) return;
      this.pending.delete(msg.id!);
      this.inflight[index]--;
      if (msg.type === "pose-result") p.resolve(msg.points ?? []);
      else p.reject(toError(msg.error));
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
          new Error(`All pose workers failed to start: ${first ?? "unknown error"}`),
        );
    }
  }
}
