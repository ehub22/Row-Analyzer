// Per-stage timing for the analysis pipeline.
//
// "Pose tracking is slow" is only actionable once you know *which* stage is
// slow, and the answer differs per machine: on a CPU/worker-pool run pose
// inference usually dominates, a GPU run can be dominated by the detector or by
// rasterisation, and a run whose pool is starved is dominated by the
// coordinator's own preprocessing. So every stage is measured, the run reports
// a breakdown into the session (and the UI), and exports keep it for
// comparison.
//
// Stage names are dotted: the prefix is the top-level stage shown as a share of
// wall time, the rest is a per-pass average ("pose.infer" → "pose 61% · infer
// 84 ms/pass"). Worker-side stages are summed across workers, so they are
// reported per call rather than as a share of the wall clock.

export interface StageTiming {
  stage: string;
  ms: number;
  count: number;
}

export interface PerformanceReport {
  frames: number;
  /** Wall-clock time of the analysis stage, in seconds. */
  seconds: number;
  fps: number;
  /** Pose-model passes dispatched (1–4 per crop, × seats). */
  posePasses: number;
  /** Passes the adaptive TTA gate skipped compared with `always`. */
  posePassesSaved: number;
  stages: StageTiming[];
  /** One-line human summary, also shown in the workspace. */
  summary: string;
}

export interface StageTimerOptions {
  /** Injectable clock, for tests. */
  now?: () => number;
}

export class StageTimer {
  private totals = new Map<string, StageTiming>();
  private clock: () => number;

  constructor(options: StageTimerOptions = {}) {
    this.clock = options.now ?? (() => performance.now());
  }

  add(stage: string, ms: number, count = 1): void {
    const entry = this.totals.get(stage) ?? { stage, ms: 0, count: 0 };
    entry.ms += ms;
    entry.count += count;
    this.totals.set(stage, entry);
  }

  time<T>(stage: string, fn: () => T): T {
    const t0 = this.clock();
    try {
      return fn();
    } finally {
      this.add(stage, this.clock() - t0);
    }
  }

  async timeAsync<T>(stage: string, fn: () => Promise<T>): Promise<T> {
    const t0 = this.clock();
    try {
      return await fn();
    } finally {
      this.add(stage, this.clock() - t0);
    }
  }

  /** Total recorded milliseconds for a stage (0 when never recorded). */
  ms(stage: string): number {
    return this.totals.get(stage)?.ms ?? 0;
  }

  /** Milliseconds per recorded call, or 0 when the stage never ran. */
  perCall(stage: string): number {
    const entry = this.totals.get(stage);
    return entry && entry.count ? entry.ms / entry.count : 0;
  }

  get count(): number {
    let total = 0;
    for (const entry of this.totals.values()) total += entry.count;
    return total;
  }

  /** Every recorded stage, slowest first. */
  snapshot(): StageTiming[] {
    return [...this.totals.values()]
      .map((s) => ({ ...s }))
      .sort((a, b) => b.ms - a.ms || a.stage.localeCompare(b.stage));
  }
}

const pct = (part: number, whole: number) =>
  whole > 0 ? Math.round((part / whole) * 100) : 0;

/**
 * Sub-stages reported per call rather than as a share of wall time: they are
 * measured inside pose workers, so summing them across workers would describe
 * CPU time, not the run. Their averages answer "inference or preprocessing?".
 */
const PER_PASS_STAGES = [
  "pose.infer",
  "pose.preprocess",
  "pose.decode",
  "detector.infer",
  "detector.preprocess",
];

/** Formats one line: shares of wall time plus per-pass stage averages. */
export function formatReport(report: PerformanceReport): string {
  const wall = report.seconds * 1000;
  const top = report.stages.filter((s) => !s.stage.includes("."));
  const accounted = top.reduce((sum, s) => sum + s.ms, 0);
  const parts = top.map((s) => `${s.stage} ${pct(s.ms, wall)}%`);
  // Whatever no stage claimed is decode/pipeline waiting, which is real time
  // and the first thing to check when the model stages look fast.
  if (wall > 0 && wall - accounted > wall * 0.03)
    parts.push(`decode wait ${pct(wall - accounted, wall)}%`);
  const shared = report.stages
    .filter((s) => PER_PASS_STAGES.includes(s.stage) && s.count > 0)
    .map((s) => `${s.stage} ${(s.ms / s.count).toFixed(1)} ms/pass`);
  const total = report.seconds
    ? `${report.seconds.toFixed(1)} s for ${report.frames} frame${report.frames === 1 ? "" : "s"}`
    : `${report.frames} frames`;
  const speed = report.fps ? ` (${report.fps.toFixed(1)} fps)` : "";
  const passes = report.posePasses
    ? ` · ${report.posePasses} pose passes${
        report.posePassesSaved
          ? `, ${report.posePassesSaved} skipped by adaptive TTA`
          : ""
      }`
    : "";
  return [total + speed, ...parts, ...shared, passes.trim().replace(/^· /, "")]
    .filter(Boolean)
    .join(" · ");
}

export interface ReportInput {
  frames: number;
  /** Wall-clock seconds of the analysis pipeline. */
  seconds: number;
  posePasses?: number;
  posePassesSaved?: number;
}

export function buildReport(
  timer: StageTimer,
  input: ReportInput,
): PerformanceReport {
  const report: PerformanceReport = {
    frames: input.frames,
    seconds: input.seconds,
    fps: input.seconds > 0 ? input.frames / input.seconds : 0,
    posePasses: input.posePasses ?? 0,
    posePassesSaved: input.posePassesSaved ?? 0,
    stages: timer.snapshot(),
    summary: "",
  };
  report.summary = formatReport(report);
  return report;
}
