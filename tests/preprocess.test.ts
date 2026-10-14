import { describe, it, expect } from "vitest";
import {
  drawPlan,
  fillTensor,
  POSE_MEAN,
  POSE_STD,
  type RasterMode,
} from "../src/core/preprocess";

const rgba = (pixels: number[][]) => new Uint8ClampedArray(pixels.flat());

describe("drawPlan", () => {
  const crop = { x: 40, y: 20, width: 100, height: 200, confidence: 1 };

  it("stretches a crop across the whole model input", () => {
    expect(drawPlan(640, 360, 288, 384, crop)).toEqual({
      sx: 40,
      sy: 20,
      sw: 100,
      sh: 200,
      dw: 288,
      dh: 384,
    });
  });

  it("letterboxes the whole frame without a crop", () => {
    // 5-argument drawImage semantics: the *source* stays the full frame and the
    // destination shrinks — cropping the source instead would feed the model a
    // different image entirely.
    expect(drawPlan(1920, 1080, 640, 640, null)).toEqual({
      sx: 0,
      sy: 0,
      sw: 1920,
      sh: 1080,
      dw: 640,
      dh: 360,
    });
    expect(drawPlan(1080, 1920, 640, 640, undefined)).toEqual({
      sx: 0,
      sy: 0,
      sw: 1080,
      sh: 1920,
      dw: 360,
      dh: 640,
    });
  });

  it("passes out-of-frame crops through so the canvas clips them as before", () => {
    const plan = drawPlan(640, 360, 288, 384, {
      x: -50,
      y: -10,
      width: 100,
      height: 500,
      confidence: 1,
    });
    expect(plan.sx).toBe(-50);
    expect(plan.sy).toBe(-10);
  });
});

describe("fillTensor", () => {
  // Two pixels: (10, 20, 30) and (200, 100, 50).
  const pixels = rgba([
    [10, 20, 30, 255],
    [200, 100, 50, 255],
  ]);

  it("normalizes RGB with the ImageNet statistics whole-body models expect", () => {
    const out = fillTensor(pixels, 2, 1, "pose", new Float32Array(6));
    expect(out[0]).toBeCloseTo((10 - POSE_MEAN[0]) / POSE_STD[0], 5);
    expect(out[1]).toBeCloseTo((200 - POSE_MEAN[0]) / POSE_STD[0], 5);
    expect(out[2]).toBeCloseTo((20 - POSE_MEAN[1]) / POSE_STD[1], 5);
    expect(out[4]).toBeCloseTo((30 - POSE_MEAN[2]) / POSE_STD[2], 5);
  });

  it("converts to BGR for YOLOX and keeps 0..255 range", () => {
    const out = fillTensor(pixels, 2, 1, "yolox", new Float32Array(6));
    expect(Array.from(out)).toEqual([30, 50, 20, 100, 10, 200]);
  });

  it("scales to 0..1 RGB for ultralytics YOLO", () => {
    const out = fillTensor(pixels, 2, 1, "yolo", new Float32Array(6));
    // Planar RGB: both R values, then both G, then both B.
    expect(Array.from(out).map((v) => Math.round(v * 255))).toEqual([
      10, 200, 20, 100, 30, 50,
    ]);
  });

  it("writes into the caller's buffer instead of allocating", () => {
    const out = new Float32Array(6);
    expect(fillTensor(pixels, 2, 1, "pose", out)).toBe(out);
  });

  it("refuses a buffer of the wrong size", () => {
    for (const mode of ["pose", "yolox", "yolo"] as RasterMode[])
      expect(() => fillTensor(pixels, 2, 1, mode, new Float32Array(3))).toThrow(
        /expected 6/,
      );
  });
});
