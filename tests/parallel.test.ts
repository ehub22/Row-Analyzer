import { describe, it, expect } from "vitest";
import {
  planParallelism,
  planThreadBudget,
  summarizePlan,
  MAX_POSE_WORKERS,
  type ParallelEnv,
} from "../src/core/parallel";

const env = (over: Partial<ParallelEnv> = {}): ParallelEnv => ({
  hardwareConcurrency: 8,
  crossOriginIsolated: false,
  sharedArrayBuffer: false,
  ...over,
});

describe("planParallelism", () => {
  it("parallelises across workers when threaded WASM is unavailable", () => {
    // The GitHub Pages / non-isolated case: ORT would silently clamp
    // numThreads to 1, so speed must come from the worker pool instead.
    const plan = planParallelism(env());
    expect(plan.poseWorkers).toBeGreaterThan(1);
    expect(plan.wasmThreads).toBe(1);
    expect(plan.detectorThreads).toBe(1);
    expect(planThreadBudget(plan)).toBe(plan.poseWorkers);
  });

  it("grants real WASM threads only with isolation AND SharedArrayBuffer", () => {
    const isolated = planParallelism(
      env({ crossOriginIsolated: true, sharedArrayBuffer: true }),
    );
    expect(isolated.wasmThreads).toBeGreaterThan(1);
    expect(planThreadBudget(isolated)).toBeLessThanOrEqual(
      isolated.hardwareConcurrency * 2,
    );
    // Isolation flag without SAB (or vice versa) must not claim threads.
    expect(
      planParallelism(env({ crossOriginIsolated: true })).wasmThreads,
    ).toBe(1);
    expect(
      planParallelism(env({ sharedArrayBuffer: true })).wasmThreads,
    ).toBe(1);
  });

  it("splits the core budget across the pool instead of oversubscribing", () => {
    const plan = planParallelism(
      env({
        hardwareConcurrency: 8,
        crossOriginIsolated: true,
        sharedArrayBuffer: true,
      }),
      2,
    );
    expect(plan.poseWorkers).toBe(2);
    expect(plan.wasmThreads).toBe(4);
    expect(planThreadBudget(plan)).toBe(8);
  });

  it("a single explicit worker on an isolated page gets the whole budget", () => {
    const plan = planParallelism(
      env({
        hardwareConcurrency: 8,
        crossOriginIsolated: true,
        sharedArrayBuffer: true,
      }),
      1,
    );
    expect(plan.poseWorkers).toBe(1);
    expect(plan.wasmThreads).toBe(8);
  });

  it("auto sizing leaves a core for decode/UI and caps model duplication", () => {
    expect(planParallelism(env({ hardwareConcurrency: 16 })).poseWorkers).toBe(4);
    expect(planParallelism(env({ hardwareConcurrency: 4 })).poseWorkers).toBe(3);
    expect(planParallelism(env({ hardwareConcurrency: 2 })).poseWorkers).toBe(1);
    expect(planParallelism(env({ hardwareConcurrency: 1 })).poseWorkers).toBe(1);
  });

  it("clamps explicit requests and tolerates bogus core counts", () => {
    expect(planParallelism(env(), 99).poseWorkers).toBe(MAX_POSE_WORKERS);
    expect(planParallelism(env(), -3).workersExplicit).toBe(false);
    const weird = planParallelism(env({ hardwareConcurrency: 0 }));
    expect(weird.poseWorkers).toBeGreaterThanOrEqual(1);
    expect(weird.wasmThreads).toBe(1);
  });

  it("summarises the plan for status lines", () => {
    const s = summarizePlan(planParallelism(env()));
    expect(s).toContain("pose workers");
    expect(s).toContain("cross-origin isolation off");
  });
});
