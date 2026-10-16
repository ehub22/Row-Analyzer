import { describe, expect, it } from "vitest";
import { SeatTracker } from "../src/core/tracking";
import { defaultBoat, missing } from "../src/core/types";
import { createDemo } from "../src/core/demo";
import { validateSession } from "../src/core/export";

const points = () => Array.from({ length: 133 }, missing);
const detection = (
  x: number,
  source: "detected" | "predicted" = "detected",
  age = 0,
) => ({
  box: {
    x,
    y: 0,
    width: 40,
    height: 100,
    confidence: source === "detected" ? 0.9 : 0,
  },
  points: points(),
  boxSource: source,
  detectorAge: age,
});

describe("predicted box identity", () => {
  it("does not turn a held crop into a certain observed detection", () => {
    const tracker = new SeatTracker({ ...defaultBoat, seats: 1 });
    tracker.update([detection(20)], 0);
    const predicted = tracker.update(
      [detection(20, "predicted", 1)],
      1 / 30,
    )[0];
    expect(predicted.boxSource).toBe("predicted");
    expect(predicted.detectorAge).toBe(1);
    expect(predicted.identityConfidence).toBeLessThan(1);
    const older = tracker.update([detection(20, "predicted", 2)], 2 / 30)[0];
    expect(older.identityConfidence).toBeLessThan(predicted.identityConfidence);
  });

  it("previews an ambiguous assignment without changing committed tracks", () => {
    const tracker = new SeatTracker({ ...defaultBoat, seats: 2 });
    tracker.update([detection(0), detection(100)], 0);
    const preview = tracker.preview(
      [detection(49, "predicted", 1), detection(51, "predicted", 1)],
      1 / 30,
    );
    expect(preview.every((f) => f.identityConfidence === 0)).toBe(true);
    const next = tracker.update([detection(1), detection(101)], 1 / 30);
    expect(next.map((f) => f.box.x)).toEqual([1, 101]);
    expect(next.every((f) => f.boxSource === "detected")).toBe(true);
  });

  it("validates predicted identity on import without rejecting old sessions", () => {
    const legacy = createDemo();
    expect(validateSession(structuredClone(legacy)).version).toBe(1);
    const predicted = structuredClone(legacy);
    predicted.frames[0].boxSource = "predicted";
    predicted.frames[0].detectorAge = 1;
    predicted.frames[0].identityConfidence = 0.75;
    expect(
      validateSession(structuredClone(predicted)).frames[0].boxSource,
    ).toBe("predicted");
    predicted.frames[0].identityConfidence = 1;
    expect(() => validateSession(predicted)).toThrow();
  });
});
