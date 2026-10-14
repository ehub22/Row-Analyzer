import { describe, expect, it } from "vitest";
import { automaticCamera } from "../src/core/calibration";
import { liftSequence } from "../src/core/lifting";
import { biomechanics } from "../src/core/biomechanics";
import { imageEdges, OarTracker } from "../src/core/oars";
import { metricVisibility } from "../src/core/geometry";
import { createDemo } from "../src/core/demo";
import { defaultBoat, missing, type PoseFrame } from "../src/core/types";
import { validateSession } from "../src/core/export";
function frame(time = 0, seat = 1): PoseFrame {
  const points = Array.from({ length: 133 }, missing);
  for (const [j, x, y] of [
    [11, 100, 200],
    [12, 122, 200],
    [5, 95, 120],
    [6, 127, 120],
    [7, 80, 150],
    [8, 150, 150],
    [9, 65, 180],
    [10, 165, 180],
    [13, 155, 230],
    [14, 177, 230],
    [15, 180, 285],
    [16, 202, 285],
  ])
    points[j] = { x: x + seat * 250, y, confidence: 0.95, origin: "observed" };
  return {
    time,
    seat,
    points,
    box: {
      x: seat * 250 + 50,
      y: 100,
      width: 200,
      height: 200,
      confidence: 0.95,
    },
    identityConfidence: 0.95,
  };
}
describe("camera estimation is explicitly assumption-limited", () => {
  it("recovers a constructed 30 degree cue but never unlocks calibrated metrics", () => {
    const frames = Array.from({ length: 40 }, (_, i) => [
      frame(i / 30, 1),
      frame(i / 30, 2),
    ]).flat();
    const camera = automaticCamera(frames, { ...defaultBoat, seats: 2 });
    expect(camera.azimuth).toBeCloseTo(30);
    expect(camera.source).toBe("estimated");
    expect(camera.confidence).toBeLessThanOrEqual(0.55);
    expect(camera.uncertaintyDegrees).toBeGreaterThanOrEqual(20);
    expect(metricVisibility(camera).allowed).toBe(false);
  });
  it("abstains on single-seat, missing, or contradictory cues", () => {
    expect(
      automaticCamera([frame()], { ...defaultBoat, seats: 1 }).azimuth,
    ).toBeNull();
    const frames = Array.from({ length: 40 }, (_, i) => [
      frame(i / 30, 1),
      frame(i / 30, 2),
    ]).flat();
    frames.forEach((f) => (f.points[12].x = f.points[11].x));
    expect(automaticCamera(frames, defaultBoat).azimuth).toBeNull();
  });
});
describe("kinematic depth hypotheses", () => {
  it("preserves projected observed coordinates and labels depth ambiguous", () => {
    const frames = Array.from({ length: 20 }, (_, i) => frame(i / 30));
    const lifted = liftSequence(frames);
    const f = lifted[10];
    expect(f.lifted?.ambiguous).toBe(true);
    expect(f.lifted?.units).toBe("torso lengths");
    expect(f.lifted?.points.filter(Boolean).length).toBeGreaterThan(5);
    f.lifted?.points.forEach((p, j) => {
      if (p) {
        expect(p.confidence).toBeLessThanOrEqual(0.35);
        expect(p.x).toBeCloseTo(
          (f.points[j].x - f.points[11].x) / Math.hypot(80, 5),
        );
      }
    });
    expect(frames[0].lifted).toBeUndefined();
  });
  it("does not fabricate missing limbs or lift uncertain identities", () => {
    const frames = Array.from({ length: 20 }, (_, i) => frame(i / 30));
    frames[10].points[9] = missing();
    frames[11].identityConfidence = 0;
    const lifted = liftSequence(frames);
    expect(lifted[10].lifted?.points[9]).toBeNull();
    expect(lifted[11].lifted).toBeUndefined();
  });
});
describe("image-supported oar candidates", () => {
  it("abstains on blank images and missing wrists", () => {
    const tracker = new OarTracker(),
      image = imageEdges(new Uint8ClampedArray(320 * 200 * 4), 320, 200);
    expect(tracker.update(image, [frame()], 640, 400)[0].oars).toEqual([]);
  });
  it("tracks a long visible line without extrapolating after disappearance", () => {
    const w = 320,
      h = 200,
      data = new Uint8ClampedArray(w * h * 4);
    data.fill(255);
    for (let x = 75; x < 230; x++)
      for (let y = 97; y <= 100; y++) {
        const i = (y * w + x) * 4;
        data[i] = data[i + 1] = data[i + 2] = 0;
      }
    const make = (time: number) => {
      const f = frame(time);
      f.box = { x: 50, y: 50, width: 50, height: 60, confidence: 1 };
      f.points[9] = { x: 70, y: 98, confidence: 1, origin: "observed" };
      f.points[10] = missing();
      return f;
    };
    const tracker = new OarTracker(),
      image = imageEdges(data, w, h);
    const a = tracker.update(image, [make(0)], w, h)[0].oars!;
    expect(a).toHaveLength(1);
    tracker.update(image, [make(0.03)], w, h);
    expect(tracker.update(image, [make(0.06)], w, h)[0].oars![0].age).toBe(3);
    expect(
      tracker.update(
        imageEdges(new Uint8ClampedArray(data.length), w, h),
        [make(0.09)],
        w,
        h,
      )[0].oars,
    ).toHaveLength(0);
  });
});
describe("2D biomechanics evidence gates", () => {
  it("reports projected kinematics for synthetic observed cycles", () => {
    const rows = biomechanics(createDemo());
    expect(rows.length).toBeGreaterThan(0);
    expect(rows.some((r) => r.metrics["Knee range (2D)"].value !== null)).toBe(
      true,
    );
    expect(
      rows.every(
        (r) => r.metrics["Candidate shaft angle range (2D)"].value === null,
      ),
    ).toBe(true);
  });
  it("withholds everything for inferred-only limbs", () => {
    const s = createDemo();
    s.frames.forEach((f) => f.points.forEach((p) => (p.origin = "inferred")));
    expect(
      biomechanics(s).every((r) =>
        Object.values(r.metrics).every((m) => m.value === null),
      ),
    ).toBe(true);
  });
  it("recomputes imported depth and rejects malformed oar data", () => {
    const s = createDemo();
    s.frames[0].lifted = {
      method: "kinematic-weak-perspective-v1",
      points: [{ x: NaN, y: 0, z: 0, confidence: 1 }],
      units: "torso lengths",
      ambiguous: true,
    };
    expect(validateSession(s).frames[0].lifted?.points[0]?.x).not.toBeNaN();
    s.frames[0].oars = [
      {
        side: "left",
        start: { x: NaN, y: 0 },
        end: { x: 0, y: 0 },
        confidence: 1,
        support: 1,
        age: 1,
        source: "image-line-candidate",
      },
    ];
    expect(() => validateSession(s)).toThrow("oar");
  });
});
