import { describe, expect, it } from "vitest";
import {
  needsFreshDetection,
  predictBoxes,
  usePredictedBoxes,
} from "../src/core/cadence";
import type { Point, PoseFrame } from "../src/core/types";

const point = (x: number, y: number): Point => ({
  x,
  y,
  confidence: 0.9,
  origin: "observed",
});
const frame = (time: number, x: number): PoseFrame => {
  const points = Array.from({ length: 133 }, () => point(x + 35, 55));
  return {
    time,
    seat: 1,
    box: { x, y: 10, width: 80, height: 120, confidence: 0.9 },
    points,
    identityConfidence: 0.9,
  };
};
const region = { left: 0, top: 0, right: 100, bottom: 100 };

describe("guarded detector cadence", () => {
  it("requires warm-up, a recent anchor, and a detector bottleneck", () => {
    const state = {
      frameIndex: 25,
      time: 25 / 30,
      lastDetectorFrame: 24,
      lastDetectorTime: 24 / 30,
      completedFrames: 25,
      detectorMs: 100,
      poseMs: 40,
      overlapping: true,
      recentRefreshes: 0,
    };
    expect(usePredictedBoxes(state)).toBe(true);
    expect(usePredictedBoxes({ ...state, detectorMs: 50 })).toBe(false);
    expect(usePredictedBoxes({ ...state, frameIndex: 27 })).toBe(false);
    expect(usePredictedBoxes({ ...state, completedFrames: 23 })).toBe(false);
    expect(usePredictedBoxes({ ...state, lastDetectorTime: 0 })).toBe(false);
    expect(usePredictedBoxes({ ...state, recentRefreshes: 3 })).toBe(false);
    expect(
      usePredictedBoxes({ ...state, overlapping: false, detectorMs: 20 }),
    ).toBe(true);
  });

  it("uses only recent observed pose motion and bounds the predicted crop", () => {
    const previous = frame(0, 30),
      latest = frame(1 / 30, 32);
    const prediction = predictBoxes(
      [latest],
      [previous],
      2 / 30,
      400,
      200,
      region,
      1,
    );
    expect(prediction).toHaveLength(1);
    expect(prediction![0]).toMatchObject({ x: 34, confidence: 0 });
    expect(
      predictBoxes([latest], [previous], 1, 400, 200, region, 1),
    ).toBeNull();
    expect(
      predictBoxes([latest], [previous], 2 / 30, 400, 200, region, 2),
    ).toBeNull();
    latest.points[11].origin = "missing";
    expect(
      predictBoxes([latest], [previous], 2 / 30, 400, 200, region, 1),
    ).toBeNull();
  });

  it("abstains near the detection region, image edge or another seat", () => {
    const a = frame(0, 30),
      b = frame(1 / 30, 32);
    expect(
      predictBoxes([b], [a], 2 / 30, 400, 200, { ...region, left: 20 }, 1),
    ).toBeNull();
    b.box.x = 345;
    expect(predictBoxes([b], [a], 2 / 30, 400, 200, region, 1)).toBeNull();
    b.box.x = 32;
    const a2 = { ...frame(0, 52), seat: 2 },
      b2 = { ...frame(1 / 30, 53), seat: 2 };
    expect(
      predictBoxes([b, b2], [a, a2], 2 / 30, 400, 200, region, 2),
    ).toBeNull();
  });

  it("forces same-frame detection when body evidence is weak or clipped", () => {
    const pose = frame(0, 30);
    expect(
      needsFreshDetection(pose.points, pose.box, 400, 200, [5, 6, 11, 12]),
    ).toBe(false);
    pose.points[11].confidence = 0.1;
    expect(
      needsFreshDetection(pose.points, pose.box, 400, 200, [5, 6, 11, 12]),
    ).toBe(true);
    pose.points[11] = point(-100, 30);
    expect(
      needsFreshDetection(pose.points, pose.box, 400, 200, [5, 6, 11, 12]),
    ).toBe(true);
  });
});
