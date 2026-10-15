import { describe, it, expect } from "vitest";
import { StageTimer, buildReport, formatReport } from "../src/core/perf";

/** Clock that returns the next scripted value on every read. */
const scripted = (...values: number[]) => {
  let i = 0;
  return () => values[Math.min(i++, values.length - 1)];
};

describe("StageTimer", () => {
  it("accumulates totals and call counts per stage", () => {
    const timer = new StageTimer();
    timer.add("detector", 10);
    timer.add("detector", 20, 2);
    expect(timer.ms("detector")).toBe(30);
    expect(timer.count).toBe(3);
    expect(timer.perCall("detector")).toBe(10);
    expect(timer.ms("never-ran")).toBe(0);
    expect(timer.perCall("never-ran")).toBe(0);
  });

  it("times synchronous and asynchronous stages, including failures", async () => {
    const timer = new StageTimer({ now: scripted(100, 130, 200, 275) });
    expect(timer.time("sync", () => 1)).toBe(1);
    expect(timer.ms("sync")).toBe(30);
    await expect(
      timer.timeAsync("async", async () => {
        throw new Error("inference failed");
      }),
    ).rejects.toThrow("inference failed");
    // A failed stage still consumed time, and hiding it would misattribute the
    // run's cost.
    expect(timer.ms("async")).toBe(75);
  });

  it("orders the snapshot slowest first", () => {
    const timer = new StageTimer();
    timer.add("pose", 5);
    timer.add("detector", 50);
    timer.add("track", 1);
    expect(timer.snapshot().map((s) => s.stage)).toEqual([
      "detector",
      "pose",
      "track",
    ]);
  });
});

describe("formatReport", () => {
  const report = () => {
    const timer = new StageTimer();
    timer.add("detector", 4000);
    timer.add("detector.infer", 3200, 40);
    timer.add("detector.preprocess", 800, 40);
    timer.add("pose", 4000);
    timer.add("pose.infer", 8000, 80);
    timer.add("pose.preprocess", 400, 80);
    timer.add("track", 200);
    return buildReport(timer, {
      frames: 40,
      seconds: 10,
      posePasses: 80,
      posePassesSaved: 240,
    });
  };

  it("reports throughput, stage shares and per-pass averages", () => {
    const built = report();
    expect(built.fps).toBeCloseTo(4, 5);
    expect(built.summary).toContain("10.0 s for 40 frames");
    expect(built.summary).toContain("(4.0 fps)");
    expect(built.summary).toContain("detector 40%");
    expect(built.summary).toContain("pose 40%");
    expect(built.summary).toContain("detector.infer 80.0 ms/pass");
    expect(built.summary).toContain("pose.preprocess 5.0 ms/pass");
    expect(built.summary).toContain("80 pose passes");
    expect(built.summary).toContain("240 skipped by adaptive TTA");
  });

  it("names the unclaimed time as decode wait", () => {
    const timer = new StageTimer();
    timer.add("pose", 1000);
    const built = buildReport(timer, { frames: 10, seconds: 10 });
    expect(built.summary).toContain("decode wait 90%");
  });

  it("omits the pose-pass clause when no passes were dispatched", () => {
    const built = buildReport(new StageTimer(), { frames: 0, seconds: 0 });
    expect(built.summary).toBe("0 frames");
    expect(built.fps).toBe(0);
  });
});
