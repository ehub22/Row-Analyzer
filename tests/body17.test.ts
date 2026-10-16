import { describe, expect, it } from "vitest";
import { checkPoseContract, checkPoseOutputDims } from "../src/core/compat";
import { decodeHeatmap, decodeSimCC, unflip } from "../src/core/pose";
import { MODEL_UNAVAILABLE_FLAG, padPose } from "../src/core/bodyLayout";
import {
  HIDDEN_FEET_FLAG,
  recoverOcclusion,
  suppressHiddenJoints,
} from "../src/core/occlusion";
import { createDemo } from "../src/core/demo";
import { validateSession } from "../src/core/export";

const box = { x: 20, y: 30, width: 80, height: 100, confidence: 1 };

describe("COCO-17 model boundary", () => {
  it("accepts only the selected joint count for both SimCC axes and heatmaps", () => {
    const input = [1, 3, 256, 192];
    expect(() =>
      checkPoseContract(
        input,
        [
          [1, 17, 384],
          [1, 17, 512],
        ],
        "simcc",
        192,
        256,
        17,
      ),
    ).not.toThrow();
    expect(() =>
      checkPoseContract(
        input,
        [
          [1, 133, 384],
          [1, 133, 512],
        ],
        "simcc",
        192,
        256,
        17,
      ),
    ).toThrow();
    expect(() =>
      checkPoseContract(
        input,
        [
          [1, 17, 384],
          [1, 133, 512],
        ],
        "simcc",
        192,
        256,
        17,
      ),
    ).toThrow();
    expect(() =>
      checkPoseContract(input, [[1, 17, 64, 48]], "heatmap", 192, 256, 17),
    ).not.toThrow();
    expect(() =>
      checkPoseContract(input, [[1, 16, 64, 48]], "heatmap", 192, 256, 17),
    ).toThrow();
    expect(() =>
      checkPoseContract(input, [[1, 133, 64, 48]], "heatmap", 192, 256, 133),
    ).not.toThrow();
  });

  it("verifies symbolic metadata against actual runtime dimensions", () => {
    checkPoseContract(
      ["batch", 3, 256, 192],
      [
        ["batch", "joints", "x"],
        ["batch", "joints", "y"],
      ],
      "simcc",
      192,
      256,
      17,
    );
    expect(() =>
      checkPoseOutputDims(
        [
          [1, 17, 384],
          [1, 17, 512],
        ],
        "simcc",
        192,
        256,
        17,
      ),
    ).not.toThrow();
    expect(() =>
      checkPoseOutputDims(
        [
          [1, 133, 384],
          [1, 133, 512],
        ],
        "simcc",
        192,
        256,
        17,
      ),
    ).toThrow();
    expect(() =>
      checkPoseOutputDims(
        [
          [1, 17, 512],
          [1, 17, 384],
        ],
        "simcc",
        192,
        256,
        17,
      ),
    ).toThrow();
    expect(() =>
      checkPoseOutputDims(
        [
          [1, 17, 0],
          [1, 17, 512],
        ],
        "simcc",
        192,
        256,
        17,
      ),
    ).toThrow();
  });

  it("decodes native body joints and pads unmodeled joints without inventing observations", () => {
    const x = new Float32Array(17 * 8),
      y = new Float32Array(17 * 8);
    for (let joint = 0; joint < 17; joint++) {
      x[joint * 8 + 3] = 0.9;
      y[joint * 8 + 4] = 0.9;
    }
    const native = decodeSimCC(x, y, 8, 8, box, 17);
    expect(native).toHaveLength(17);
    expect(native[11].origin).toBe("observed");
    const padded = padPose(native, 17);
    expect(padded).toHaveLength(133);
    expect(padded[16].origin).toBe("observed");
    expect(
      padded
        .slice(17)
        .every(
          (p) =>
            p.origin === "missing" &&
            p.confidence === 0 &&
            p.flags?.includes(MODEL_UNAVAILABLE_FLAG),
        ),
    ).toBe(true);
    expect(padPose(padded, 133)).toBe(padded);
  });

  it("unflips body pairs before padding and keeps unavailable points missing", () => {
    const points = Array.from({ length: 17 }, (_, i) => ({
      x: box.x + i,
      y: box.y + i,
      confidence: 1,
      origin: "observed" as const,
    }));
    const restored = unflip(unflip(points, box), box);
    expect(restored).toHaveLength(17);
    restored.forEach((p, i) => expect(p.x).toBeCloseTo(points[i].x));
    const padded = padPose(unflip(points, box), 17);
    expect(padded[17].x).toBe(0);
    expect(padded[17].origin).toBe("missing");
  });

  it("decodes 17 heatmaps while rejecting wrong tensor lengths", () => {
    const heat = new Float32Array(17 * 16);
    heat[5] = 0.95;
    const points = decodeHeatmap(heat, 4, 4, box, 17);
    expect(points).toHaveLength(17);
    expect(points[0].origin).toBe("observed");
    expect(() =>
      decodeHeatmap(new Float32Array(16 * 16), 4, 4, box, 17),
    ).toThrow();
  });

  it("preserves model-unavailable provenance through hidden-feet suppression and recovery", () => {
    const body = Array.from({ length: 17 }, () => ({
      x: 50,
      y: 50,
      confidence: 0.9,
      origin: "observed" as const,
    }));
    const padded = padPose(body, 17);
    const hidden = suppressHiddenJoints(
      padded,
      [15, 16, 17, 18, 19, 20, 21, 22],
    );
    expect(hidden[17].flags).toContain(MODEL_UNAVAILABLE_FLAG);
    expect(hidden[17].flags).toContain(HIDDEN_FEET_FLAG);
    const frames = [0, 0.05, 0.1].map((time) => ({
      time,
      seat: 1,
      box,
      points: structuredClone(hidden),
      identityConfidence: 0.9,
    }));
    frames[0].points[23] = { x: 50, y: 50, confidence: 1, origin: "manual" };
    frames[2].points[23] = { x: 52, y: 50, confidence: 1, origin: "manual" };
    const recovered = recoverOcclusion(frames);
    expect(recovered[1].points[23].origin).toBe("missing");
    expect(recovered[1].points[23].flags).toContain(MODEL_UNAVAILABLE_FLAG);
  });

  it("round-trips body-only provenance while preserving legacy sessions and manual extras", () => {
    const session = structuredClone(createDemo());
    expect(
      validateSession(structuredClone(session)).poseJoints,
    ).toBeUndefined();
    session.poseJoints = 17;
    session.frames = session.frames.map((frame) => ({
      ...frame,
      points: padPose(frame.points, 17),
    }));
    expect(
      validateSession(structuredClone(session)).frames[0].points,
    ).toHaveLength(133);
    session.frames[0].points[23] = {
      x: 2,
      y: 3,
      confidence: 1,
      origin: "manual",
    };
    expect(
      validateSession(structuredClone(session)).frames[0].points[23].origin,
    ).toBe("manual");
    session.frames[1].points[24].flags = [];
    expect(() => validateSession(session)).toThrow();
  });
});
