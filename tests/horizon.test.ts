import { describe, it, expect } from "vitest";
import {
  detectHorizon,
  lockHorizon,
  grayFromRgba,
  downscale,
  type GrayImage,
} from "../src/core/horizon";

const W = 640,
  H = 360;

/** Deterministic PRNG so the fixtures are reproducible. */
function rng(seed: number) {
  let s = seed >>> 0;
  return () => (s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32;
}

/**
 * A water-and-sky frame with a tilted horizon, sensor noise, vertical oar-like
 * stripes, and optional moving blobs (rowers/bystanders) that do not stay put.
 */
function scene(opts: {
  horizonY?: number;
  tiltDeg?: number;
  seed: number;
  blobs?: number;
  stripes?: boolean;
  noSky?: boolean;
}): GrayImage {
  const rand = rng(opts.seed);
  const data = new Float32Array(W * H);
  const yc = opts.horizonY ?? 150;
  const slope = Math.tan(((opts.tiltDeg ?? 0) * Math.PI) / 180);
  const blobs = Array.from({ length: opts.blobs ?? 0 }, () => ({
    x: rand() * W,
    y: 170 + rand() * 150,
    r: 6 + rand() * 6,
    v: 0.1 + rand() * 0.1,
  }));
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      const hy = yc + slope * (x - W / 2);
      let v = opts.noSky ? 0.45 : y < hy ? 0.82 : 0.32;
      if (opts.stripes && x % 97 < 3 && y > 200) v = 0.1;
      v += (rand() - 0.5) * 0.04;
      data[y * W + x] = v;
    }
  for (const b of blobs) {
    for (let y = Math.max(0, b.y - b.r); y < Math.min(H, b.y + b.r); y++)
      for (let x = Math.max(0, b.x - b.r); x < Math.min(W, b.x + b.r); x++)
        if ((x - b.x) ** 2 + (y - b.y) ** 2 < b.r * b.r) data[y * W + x] = 0.95;
    b.x = (b.x + b.v * 40) % W;
  }
  return { data, width: W, height: H };
}

/** Frame-to-frame: same horizon, different clutter (what a static camera sees). */
function sequence(n: number, base: Omit<Parameters<typeof scene>[0], "seed">) {
  return Array.from({ length: n }, (_, i) =>
    scene({ ...base, seed: 1000 + i, blobs: 3 }),
  );
}

/** Expected horizon height at the image centre for (yc, tilt). */
const yAtCentre = (yc: number, tiltDeg: number) =>
  yc + Math.tan((tiltDeg * Math.PI) / 180) * 0;

describe("automatic horizon detection", () => {
  it("locks onto a tilted horizon in a single frame", () => {
    const img = scene({
      horizonY: 150,
      tiltDeg: 4,
      seed: 1,
      blobs: 3,
      stripes: true,
    });
    const det = detectHorizon(img);
    expect(det.accepted).toBe(true);
    expect(det.tiltDeg).toBeCloseTo(4, 0);
    expect(det.yAtCenter).toBeCloseTo(yAtCentre(150, 4), 0);
    expect(det.support).toBeGreaterThan(0.2);
  });

  it("detects a level horizon with a different height", () => {
    const det = detectHorizon(
      scene({ horizonY: 230, tiltDeg: 0, seed: 2, stripes: true }),
    );
    expect(det.accepted).toBe(true);
    expect(Math.abs(det.tiltDeg)).toBeLessThan(0.3);
    expect(det.yAtCenter).toBeCloseTo(230, 0);
  });

  it("locks the horizon across frames where clutter moves (static camera)", () => {
    const frames = sequence(8, { horizonY: 160, tiltDeg: -3, stripes: true });
    const det = lockHorizon(frames);
    expect(det.accepted).toBe(true);
    expect(det.tiltDeg).toBeCloseTo(-3, 0);
    expect(det.yAtCenter).toBeCloseTo(160, 0);
  });

  it("abstains on a featureless frame", () => {
    const det = detectHorizon(scene({ seed: 3, noSky: true }));
    expect(det.accepted).toBe(false);
  });

  it("does not claim a horizon from pure noise", () => {
    const rand = rng(9);
    const data = Float32Array.from({ length: W * H }, () => rand());
    const det = detectHorizon({ data, width: W, height: H });
    expect(det.accepted).toBe(false);
  });

  it("downscale keeps the horizon at the same relative height", () => {
    const img = scene({ horizonY: 180, tiltDeg: 0, seed: 4 });
    const { image, factor } = downscale(img, 320);
    expect(factor).toBeCloseTo(2, 6);
    expect(image.width).toBe(320);
    // Half-size rows 89 and 90 cover original rows 178–181, straddling the horizon at 180.
    expect(image.data[89 * 320 + 160]).toBeGreaterThan(0.6);
    expect(image.data[90 * 320 + 160]).toBeLessThan(0.5);
  });

  it("converts RGBA bytes to luminance", () => {
    const rgba = new Uint8ClampedArray([255, 255, 255, 255, 0, 0, 0, 255]);
    const g = grayFromRgba(rgba, 2, 1);
    expect(g.data[0]).toBeCloseTo(1, 6);
    expect(g.data[1]).toBeCloseTo(0, 6);
  });
});
