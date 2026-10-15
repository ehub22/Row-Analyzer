import { describe, it, expect } from "vitest";
import {
  assessPose,
  describeTta,
  passPlan,
  BODY_JOINTS,
  TTA_SCALES,
} from "../src/core/tta";
import { paddedCrop } from "../src/core/detection";
import type { Box, Point } from "../src/core/types";

const box: Box = { x: 100, y: 50, width: 80, height: 200, confidence: 0.9 };
const W = 288,
  H = 384;

const pose = (confidence: number | ((j: number) => number)): Point[] =>
  Array.from({ length: 133 }, (_, j) => ({
    x: j,
    y: j,
    confidence: typeof confidence === "function" ? confidence(j) : confidence,
    origin:
      (typeof confidence === "function" ? confidence(j) : confidence) >= 0.3
        ? ("observed" as const)
        : ("missing" as const),
  }));

describe("passPlan", () => {
  it("runs one pass when TTA is off", () => {
    const passes = passPlan("off", box, W, H);
    expect(passes).toHaveLength(1);
    expect(passes[0]).toMatchObject({ flip: false, scale: 1.25 });
  });

  it("runs the four-pass plan for always and adaptive", () => {
    for (const mode of ["always", "adaptive"] as const) {
      const passes = passPlan(mode, box, W, H);
      expect(passes).toHaveLength(4);
      expect(passes.map((p) => [p.scale, p.flip])).toEqual([
        [1.25, false],
        [1.25, true],
        [1.5, false],
        [1.5, true],
      ]);
      expect(TTA_SCALES).toEqual([1.25, 1.5]);
    }
  });

  it("keeps the existing padded-crop geometry for every pass", () => {
    const passes = passPlan("always", box, W, H);
    expect(passes[0].crop).toEqual(paddedCrop(box, W, H, 1.25));
    expect(passes[3].crop).toEqual(paddedCrop(box, W, H, 1.5));
  });

  it("makes adaptive start with the same pass always starts with", () => {
    // Otherwise a gate rejection would fuse a different pass set than `always`.
    const adaptive = passPlan("adaptive", box, W, H);
    const always = passPlan("always", box, W, H);
    expect(adaptive[0]).toEqual(always[0]);
    expect(adaptive.map((p) => p.crop)).toEqual(always.map((p) => p.crop));
  });
});

describe("assessPose", () => {
  it("accepts a clean single pass", () => {
    const verdict = assessPose(pose(0.9));
    expect(verdict.accepted).toBe(true);
    expect(verdict.weak).toBe(0);
    expect(verdict.mean).toBeCloseTo(0.9, 5);
  });

  it("rejects when a body joint falls below the observed threshold", () => {
    const verdict = assessPose(pose((j) => (j === 9 ? 0.2 : 0.9)));
    expect(verdict.accepted).toBe(false);
    expect(verdict.weak).toBe(1);
    expect(verdict.reason).toContain("1 of 17");
  });

  it("ignores weak face, hand and foot joints", () => {
    // Face/hands/feet are exactly the joints oars, water and the boat occlude,
    // and the stroke metrics do not use them; requiring them would make every
    // crop pay for the full fusion.
    const weakExtremities = pose((j) => (j >= BODY_JOINTS.length ? 0.05 : 0.9));
    expect(assessPose(weakExtremities).accepted).toBe(true);
  });

  it("rejects when every joint is only marginally above the threshold", () => {
    // 0.35 clears minJoint but not minMean: a pass that shaky does not get to
    // skip using the other three.
    expect(assessPose(pose(0.35)).accepted).toBe(false);
  });

  it("treats missing joints as zero confidence", () => {
    const points = pose(0.9);
    points[5] = { x: 0, y: 0, confidence: 0, origin: "missing" };
    const verdict = assessPose(points);
    expect(verdict.accepted).toBe(false);
    expect(verdict.weak).toBe(1);
  });
});

describe("describeTta", () => {
  it("names each policy for status lines and session exports", () => {
    expect(describeTta("off")).toBe("single pass");
    expect(describeTta("adaptive")).toContain("adaptive");
    expect(describeTta("always")).toContain("flip + multi-scale");
  });
});
