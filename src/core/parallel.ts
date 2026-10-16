// Parallelism planning for the ONNX pipeline.
//
// Two independent levels of parallelism are available in the browser:
//
// 1. A pool of pose Web Workers, each running its own WASM inference session.
//    Workers are separate OS threads and need no special page setup, so this
//    works on every browser (Chrome/Firefox/Safari) and on plain static
//    hosting such as GitHub Pages. The cost is memory: every pool worker
//    loads its own copy of the pose model (~134 MB for DWPose-l).
// 2. ONNX Runtime WASM intra-operator threads (`ort.env.wasm.numThreads`).
//    These need SharedArrayBuffer, which browsers only expose on
//    cross-origin-isolated pages (COOP + COEP response headers). Without
//    isolation ORT *silently* clamps numThreads back to 1, so this module
//    checks isolation explicitly and reports the cap instead of pretending.
//
// `planParallelism` is a pure function so the policy is unit-testable; the
// App and both workers all derive their configuration from the same plan.

export interface ParallelEnv {
  /** Logical CPU cores reported by the browser (may be capped for privacy). */
  hardwareConcurrency: number;
  /** True only when COOP/COEP isolation is in effect for this context. */
  crossOriginIsolated: boolean;
  /** SharedArrayBuffer constructor present (requires isolation in practice). */
  sharedArrayBuffer: boolean;
}

export interface ParallelPlan extends ParallelEnv {
  /** Number of pose worker sessions to run concurrently (always >= 1). */
  poseWorkers: number;
  /** WASM threads granted to each pose session. 1 unless isolated. */
  wasmThreads: number;
  /** WASM threads granted to the detector session; it may overlap pose work.
   * 1 unless isolated. */
  detectorThreads: number;
  /** True when the user picked the worker count explicitly. */
  workersExplicit: boolean;
}

export const MAX_POSE_WORKERS = 8;
/** Auto pool cap: each extra worker duplicates the pose model in memory. */
export const AUTO_POSE_WORKER_CAP = 4;

const clampInt = (v: number, lo: number, hi: number) =>
  Math.min(hi, Math.max(lo, Math.floor(v)));

/** Reads the live environment. Safe in window, worker, and test contexts. */
export function readParallelEnv(): ParallelEnv {
  const g = globalThis as {
    crossOriginIsolated?: boolean;
    SharedArrayBuffer?: unknown;
    navigator?: { hardwareConcurrency?: number };
  };
  let cores = 4;
  try {
    cores = g.navigator?.hardwareConcurrency ?? cores;
  } catch {
    /* navigator may be unavailable; keep the conservative default */
  }
  return {
    hardwareConcurrency:
      Number.isFinite(cores) && cores > 0 ? Math.floor(cores) : 4,
    crossOriginIsolated: g.crossOriginIsolated === true,
    sharedArrayBuffer: typeof g.SharedArrayBuffer === "function",
  };
}

/**
 * Decides pool size and per-session thread counts.
 * `requestedWorkers`: 0/undefined = automatic, 1..8 = explicit override.
 */
export function planParallelism(
  env: ParallelEnv,
  requestedWorkers = 0,
): ParallelPlan {
  const cores = clampInt(env.hardwareConcurrency || 4, 1, 64);
  // Threaded WASM needs SharedArrayBuffer; isolation without SAB (or SAB
  // without isolation, which browsers do not grant) both mean 1 thread.
  const threaded = env.crossOriginIsolated && env.sharedArrayBuffer;

  const explicit = requestedWorkers > 0;
  // Automatic sizing leaves a core for video decode + the coordinator's
  // detector/preprocessing work and the UI thread, and caps the pool so a
  // 16-core machine does not load eight copies of a 134 MB model by default.
  const auto = clampInt(cores - 1, 1, AUTO_POSE_WORKER_CAP);
  const poseWorkers = explicit
    ? clampInt(requestedWorkers, 1, MAX_POSE_WORKERS)
    : auto;

  // Detection and pose overlap on separate workers. Leave a core for decode/UI
  // and, on larger machines, one for the detector instead of granting every
  // logical core to pose and then oversubscribing it with detector threads.
  const reserve = cores >= 8 ? 2 : cores > 1 ? 1 : 0;
  const wasmThreads = threaded
    ? clampInt(Math.floor((cores - reserve) / poseWorkers), 1, cores)
    : 1;
  // Integer division can leave extra cores idle on large machines (20 cores,
  // 4 workers × 4 threads leaves four). Give the detector those spare cores
  // while retaining one for decode and UI instead of slowing every frame.
  const detectorThreads = threaded
    ? cores >= 16
      ? clampInt(cores - poseWorkers * wasmThreads - 1, 1, 4)
      : clampInt(reserve, 1, 4)
    : 1;

  return {
    hardwareConcurrency: cores,
    crossOriginIsolated: env.crossOriginIsolated,
    sharedArrayBuffer: env.sharedArrayBuffer,
    poseWorkers,
    wasmThreads,
    detectorThreads,
    workersExplicit: explicit,
  };
}

/** Approximate peak CPU threads the plan can keep busy during pose work. */
export function planThreadBudget(plan: ParallelPlan): number {
  return plan.poseWorkers * plan.wasmThreads;
}

/** One-line human summary, e.g. for status lines and the settings dialog. */
export function summarizePlan(plan: ParallelPlan): string {
  const workerPart =
    plan.poseWorkers === 1
      ? "1 pose worker"
      : `${plan.poseWorkers} pose workers`;
  const threadPart =
    plan.wasmThreads === 1
      ? "1 WASM thread each"
      : `${plan.wasmThreads} WASM threads each`;
  return `${workerPart} × ${threadPart} · detector ${plan.detectorThreads} WASM thread${plan.detectorThreads === 1 ? "" : "s"} · ${plan.hardwareConcurrency} logical cores · cross-origin isolation ${plan.crossOriginIsolated ? "on" : "off"}`;
}
