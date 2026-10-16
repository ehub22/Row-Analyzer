import { describe, it, expect } from "vitest";
import {
  angle3D,
  rotateY,
  toBoatFrame,
  estimateCamera,
  scaleFromSegment,
} from "../src/core/geometry";
import { createDemo } from "../src/core/demo";
import { recoverOcclusion, applyCorrection } from "../src/core/occlusion";
import {
  segmentStrokes,
  crewTiming,
  crewMean,
  mean,
} from "../src/core/strokes";
import { SeatTracker } from "../src/core/tracking";
import { paddedCrop, nms, decodeYolox } from "../src/core/detection";
import { decodeSimCC, decodeHeatmap, unflip } from "../src/core/pose";
import { flipIndices } from "../src/core/keypoints";
import { defaultBoat, missing, type PoseFrame } from "../src/core/types";
import { validateSession } from "../src/core/export";
const demo = createDemo();
describe("boat frame geometry (known 3D, not monocular recovery)", () => {
  it.each([-85, -60, -30, 0, 15, 45, 75, 90])(
    "recovers calibrated joint angles at %s degrees",
    (azimuth) => {
      const pose = [
        { x: -0.3, y: 1, z: 0.1 },
        { x: 0, y: 0.5, z: 0 },
        { x: 0.4, y: 0, z: 0.2 },
      ] as const;
      const rendered = pose.map((p) => rotateY(p, azimuth));
      const normalized = rendered.map((p) => toBoatFrame(p, azimuth));
      expect(angle3D(normalized[0], normalized[1], normalized[2])).toBeCloseTo(
        angle3D(...pose)!,
        10,
      );
    },
  );
  it("rejects degenerate geometry and end-on scale", () => {
    expect(
      angle3D({ x: 0, y: 0, z: 0 }, { x: 0, y: 0, z: 0 }, { x: 1, y: 0, z: 0 }),
    ).toBeNull();
    expect(scaleFromSegment(10, 17, 90)).toBeNull();
    expect(scaleFromSegment(100, 10, 0)).toBe(0.1);
  });
  it("does not pretend a 2D shell axis determines azimuth", () => {
    expect(
      estimateCamera([
        { x: 1, y: 1, confidence: 1, origin: "observed" },
        { x: 200, y: 1, confidence: 1, origin: "observed" },
      ]).azimuth,
    ).toBeNull();
  });
});
describe("top-down pose math", () => {
  it("uses padded aspect-ratio-preserving crops", () => {
    const crop = paddedCrop(
      { x: 100, y: 100, width: 100, height: 100, confidence: 1 },
      288,
      384,
    );
    expect(crop.width / crop.height).toBe(0.75);
    expect(crop.x + crop.width / 2).toBe(150);
  });
  it("NMS removes overlaps but not distinct people", () => {
    const a = { x: 0, y: 0, width: 100, height: 100, confidence: 1 };
    expect(
      nms([a, { ...a, x: 1, confidence: 0.5 }, { ...a, x: 300 }]),
    ).toHaveLength(2);
  });
  it("requires documented detector output shape", () => {
    expect(() => decodeYolox(new Float32Array(2), [1, 6], 640, 1)).toThrow();
  });
  it("decodes the smaller YOLOX 416 raw-grid contract", () => {
    const data = new Float32Array(3549 * 85);
    const index = (10 * 52 + 10) * 85;
    data.set([0.5, 0.5, Math.log(40 / 8), Math.log(80 / 8), 0.9, 0.9], index);
    const boxes = decodeYolox(data, [1, 3549, 85], 416, 1);
    expect(boxes).toHaveLength(1);
    expect(boxes[0].x).toBeCloseTo(64);
    expect(boxes[0].y).toBeCloseTo(44);
    expect(boxes[0].width).toBeCloseTo(40);
    expect(boxes[0].height).toBeCloseTo(80);
  });
  it("decodes and subpixel-refines whole-body SimCC", () => {
    const x = new Float32Array(133 * 8),
      y = new Float32Array(133 * 8);
    for (let j = 0; j < 133; j++) {
      x[j * 8 + 3] = 0.7;
      x[j * 8 + 4] = 1;
      x[j * 8 + 5] = 0.9;
      y[j * 8 + 2] = 0.9;
    }
    const points = decodeSimCC(x, y, 8, 8, {
      x: 0,
      y: 0,
      width: 80,
      height: 80,
      confidence: 1,
    });
    expect(points).toHaveLength(133);
    expect(points[0].x).toBeCloseTo(42.5);
    expect(points[0].y).toBe(20);
    expect(points[0].confidence).toBeCloseTo(0.9);
  });
  it("supports heatmaps and rejects body-only outputs", () => {
    const heat = new Float32Array(133 * 16);
    heat[5] = 1;
    const p = decodeHeatmap(heat, 4, 4, {
      x: 0,
      y: 0,
      width: 40,
      height: 40,
      confidence: 1,
    });
    expect(p[0].x).toBe(10);
    expect(p[1].origin).toBe("missing");
    expect(() =>
      decodeHeatmap(new Float32Array(17 * 16), 4, 4, {
        x: 0,
        y: 0,
        width: 40,
        height: 40,
        confidence: 1,
      }),
    ).toThrow();
  });
  it("whole-body mirror map is an involution including face/hands", () => {
    flipIndices.forEach((j, i) => expect(flipIndices[j]).toBe(i));
    const pose = demo.frames[0].points,
      crop = demo.frames[0].box;
    const restored = unflip(unflip(pose, crop), crop);
    restored.forEach((p, i) => expect(p.x).toBeCloseTo(pose[i].x));
  });
});
describe("occlusion and corrections", () => {
  const seq = () =>
    structuredClone(demo.frames.filter((f) => f.seat === 1).slice(0, 25));
  it("fills short masked gaps bidirectionally, labels inference and measures error", () => {
    const frames = seq(),
      original = structuredClone(frames);
    for (let i = 7; i < 10; i++) frames[i].points[9] = missing();
    const recovered = recoverOcclusion(frames);
    for (let i = 7; i < 10; i++) {
      expect(recovered[i].points[9].origin).toBe("inferred");
      expect(recovered[i].points[9].confidence).toBeLessThanOrEqual(0.45);
      expect(
        Math.hypot(
          recovered[i].points[9].x - original[i].points[9].x,
          recovered[i].points[9].y - original[i].points[9].y,
        ),
      ).toBeLessThan(1);
    }
    expect(frames[7].points[9].origin).toBe("missing");
  });
  it("leaves long gaps missing", () => {
    const frames = seq();
    for (let i = 3; i < 20; i++) frames[i].points[9] = missing();
    expect(recoverOcclusion(frames)[10].points[9].origin).toBe("missing");
  });
  it("does not bridge low-confidence jumps or uncertain identities", () => {
    const frames = seq();
    frames[7].points[9] = { ...frames[7].points[9], x: 3000, confidence: 0.4 };
    expect(recoverOcclusion(frames)[7].points[9].flags).toContain(
      "Low-confidence jump; not bridged",
    );
    frames[8].points[9] = missing();
    frames[8].identityConfidence = 0;
    expect(recoverOcclusion(frames)[8].points[9].origin).toBe("missing");
  });
  it("preserves manual anchors and marks propagated changes inferred", () => {
    const frames = seq(),
      frame = frames[9],
      before = frame.points[9];
    const corrected = applyCorrection(frames, {
      seat: 1,
      time: frame.time,
      joint: 9,
      before,
      after: { ...before, x: before.x + 10, origin: "manual" },
    });
    expect(corrected[9].points[9].origin).toBe("manual");
    expect(corrected[8].points[9].origin).toBe("inferred");
    expect(corrected[0].points[9].origin).toBe("observed");
  });
});
describe("seat identity", () => {
  const detection = (x: number) => ({
    box: { x, y: 0, width: 30, height: 100, confidence: 1 },
    points: demo.frames[0].points,
  });
  const boat = { ...defaultBoat, seats: 2 };
  it("is invariant to detection order and preserves missing slots", () => {
    const t = new SeatTracker(boat);
    expect(
      t.update([detection(100), detection(0)], 0).map((f) => f.box.x),
    ).toEqual([0, 100]);
    expect(
      t.update([detection(105), detection(5)], 0.03).map((f) => f.box.x),
    ).toEqual([5, 105]);
    const result = t.update([detection(110)], 0.06);
    expect(result[0].points[0].origin).toBe("missing");
    expect(result[1].seat).toBe(2);
  });
  it("requires review when bystanders or a coxswain prevent safe initialization", () => {
    expect(() => new SeatTracker(boat).update([detection(0)], 0)).toThrow(
      "Found 1 people",
    );
  });
  it("flags ambiguous overlap instead of swapping identities", () => {
    const t = new SeatTracker(boat);
    t.update([detection(0), detection(100)], 0);
    expect(
      t
        .update([detection(49), detection(51)], 0.03)
        .every((f) => f.identityConfidence === 0),
    ).toBe(true);
  });
});
describe("cycle timing and evidence gates", () => {
  it("recovers generated 25 spm cycles", () => {
    const rows = demo.strokes.filter((s) => s.seat === 8);
    expect(rows.length).toBeGreaterThan(2);
    expect(mean(rows.map((s) => s.rate.value))).toBeCloseTo(25, 0);
    expect(mean(rows.map((s) => s.ratio.value))!).toBeGreaterThan(0.5);
    expect(mean(rows.map((s) => s.ratio.value))!).toBeLessThan(0.9);
  });
  it("withholds a crew-wide average when only one seat has reliable cycles", () => {
    const sparse = structuredClone(demo.strokes);
    sparse.forEach((stroke, i) => {
      if (i > 0) stroke.rate.value = null;
    });
    expect(crewMean(sparse, 8, (s) => s.rate.value)).toBeNull();
    const representative = demo.strokes.filter((s) => s.seat <= 4);
    expect(crewMean(representative, 8, (s) => s.rate.value)).toBeCloseTo(25, 0);
  });
  it("resolves crew timing only for matched cycles", () => {
    const timing = crewTiming(demo.strokes, 8);
    expect(timing.length).toBeGreaterThan(0);
    expect(timing.every((t) => Math.abs(t.catchMs) < 150)).toBe(true);
  });
  it("withholds timing when all limbs are inferred", () => {
    const frames = structuredClone(demo.frames.filter((f) => f.seat === 8));
    frames.forEach((f) => f.points.forEach((p) => (p.origin = "inferred")));
    const strokes = segmentStrokes(frames);
    expect(strokes.length).toBeGreaterThan(0);
    expect(strokes.every((s) => s.rate.value === null)).toBe(true);
  });
  it("does not produce strokes for stationary or missing data", () => {
    const frames: PoseFrame[] = Array.from({ length: 100 }, (_, i) => ({
      ...structuredClone(demo.frames[0]),
      time: i / 30,
    }));
    expect(segmentStrokes(frames)).toHaveLength(0);
  });
});
describe("session interchange", () => {
  it("accepts full sessions, rejects malformed input", () => {
    expect(validateSession(demo).version).toBe(1);
    expect(() => validateSession({ version: 1 })).toThrow();
    const bad = structuredClone(demo);
    bad.frames[0].points[0].x = NaN;
    expect(() => validateSession(bad)).toThrow();
  });
});
