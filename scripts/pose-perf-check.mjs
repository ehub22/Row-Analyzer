// Rasterisation equivalence + stage-cost check (developer tool, not part of the
// app). Run against the dev server, which serves the TypeScript modules:
//
//   npm run dev
//   node scripts/pose-perf-check.mjs                  # default http://localhost:5173
//   URL=http://localhost:5173 node scripts/pose-perf-check.mjs
//
// Two questions this answers, on *your* machine and browser:
//
//  1. Does drawing a crop from the frame's shared ImageBitmap produce exactly
//     the same tensor as drawing the VideoFrame per pass? If it ever does not,
//     the fast path has changed what the model sees and must not be used.
//     Reported as the maximum per-channel difference (must be 0).
//  2. What does one pass cost where? Coordinator-side VideoFrame rasterisation,
//     bitmap rasterisation, the normalization loop, and pose post-processing
//     are timed separately so a slow run can be attributed instead of guessed.
//
// The numbers are machine-specific by design; the equivalence check is the part
// that must hold everywhere.

import { chromium } from "playwright";
import { resolveChromium } from "./lib/chromium.mjs";

const url = process.env.URL ?? "http://localhost:5173";
const { executablePath, args, env } = await resolveChromium();
const browser = await chromium.launch({
  executablePath,
  args,
  env,
  headless: true,
});
const page = await browser.newPage();
page.on("pageerror", (e) => console.log("[pageerror]", e.message));
await page.goto(url);

const report = await page.evaluate(async () => {
  const { Rasterizer, FrameSource, drawPlan, fillTensor } =
    await import("/src/core/preprocess.ts");
  const { decodeSimCC, fusePoses } = await import("/src/core/pose.ts");
  const { SeatTracker } = await import("/src/core/tracking.ts");
  const { defaultBoat } = await import("/src/core/types.ts");
  const { assessPose, passPlan } = await import("/src/core/tta.ts");

  // A 1920x1080 synthetic frame stands in for decoded video.
  const canvas = new OffscreenCanvas(1920, 1080);
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#6f7d8c";
  ctx.fillRect(0, 0, 1920, 1080);
  for (let i = 0; i < 600; i++) {
    ctx.fillStyle = `rgb(${(i * 37) % 255}, ${(i * 91) % 255}, ${(i * 53) % 255})`;
    ctx.fillRect((i * 37) % 1860, (i * 71) % 1020, 60, 60);
  }
  const frame = new VideoFrame(canvas, { timestamp: 0 });

  const W = 288,
    H = 384;
  const crop = { x: 400, y: 200, width: 300, height: 420, confidence: 0.9 };
  const time = (label, n, fn) => {
    fn();
    const t0 = performance.now();
    for (let i = 0; i < n; i++) fn();
    return { label, msPerCall: +((performance.now() - t0) / n).toFixed(2) };
  };

  // Reference path: the original implementation, per-pass VideoFrame draw.
  const reference = new OffscreenCanvas(W, H);
  const referenceCtx = reference.getContext("2d", { willReadFrequently: true });
  const legacyTensor = (flip) => {
    referenceCtx.fillStyle = "black";
    referenceCtx.fillRect(0, 0, W, H);
    referenceCtx.save();
    if (flip) {
      referenceCtx.translate(W, 0);
      referenceCtx.scale(-1, 1);
    }
    referenceCtx.drawImage(
      frame,
      crop.x,
      crop.y,
      crop.width,
      crop.height,
      0,
      0,
      W,
      H,
    );
    referenceCtx.restore();
    const pixels = referenceCtx.getImageData(0, 0, W, H).data;
    const out = new Float32Array(3 * W * H);
    return fillTensor(pixels, W, H, "pose", out);
  };

  // Fast path: one bitmap per frame, every pass drawn from it.
  const source = await FrameSource.create(frame);
  const rasterizer = new Rasterizer(W, H, "pose");
  const fastTensor = (flip) =>
    rasterizer.tensor(source.image, source.width, source.height, crop, flip);

  const diff = (a, b) => {
    let max = 0;
    for (let i = 0; i < a.length; i++)
      max = Math.max(max, Math.abs(a[i] - b[i]));
    return max;
  };
  const plainDiff = diff(legacyTensor(false), fastTensor(false));
  const flipDiff = diff(legacyTensor(true), fastTensor(true));

  // Detector geometry: letterboxed full frame, the other rasterisation shape.
  const detectRaster = new Rasterizer(640, 640, "yolox");
  const legacyDetect = (() => {
    const c = new OffscreenCanvas(640, 640);
    const g = c.getContext("2d", { willReadFrequently: true });
    g.fillStyle = "rgb(114,114,114)";
    g.fillRect(0, 0, 640, 640);
    const ratio = Math.min(640 / frame.displayWidth, 640 / frame.displayHeight);
    g.drawImage(
      frame,
      0,
      0,
      frame.displayWidth * ratio,
      frame.displayHeight * ratio,
    );
    return fillTensor(
      g.getImageData(0, 0, 640, 640).data,
      640,
      640,
      "yolox",
      new Float32Array(3 * 640 * 640),
    );
  })();
  const fastDetect = detectRaster.tensor(
    source.image,
    source.width,
    source.height,
    null,
    false,
  );

  const timings = [
    time("VideoFrame crop -> tensor (per pass, previous path)", 30, () =>
      legacyTensor(false),
    ),
    time("shared bitmap crop -> tensor (per pass)", 60, () =>
      fastTensor(false),
    ),
    time("shared bitmap crop -> tensor (flipped)", 60, () => fastTensor(true)),
    time("shared bitmap letterbox -> 640 tensor (detector)", 40, () =>
      detectRaster.tensor(
        source.image,
        source.width,
        source.height,
        null,
        false,
      ),
    ),
  ];

  // Pose post-processing, for scale: what the fused pipeline adds per crop.
  const nx = 576,
    ny = 768;
  const sx = new Float32Array(133 * nx),
    sy = new Float32Array(133 * ny);
  for (let j = 0; j < 133; j++) {
    sx[j * nx + 250 + (j % 20)] = 0.95;
    sy[j * ny + 300 + (j % 30)] = 0.95;
  }
  timings.push(
    time("SimCC decode (133 joints, both axes)", 40, () =>
      decodeSimCC(sx, sy, nx, ny, crop),
    ),
  );
  const poses = Array.from({ length: 4 }, () =>
    decodeSimCC(sx, sy, nx, ny, crop),
  );
  timings.push(time("fusePoses (4 passes)", 40, () => fusePoses(poses)));
  const tracker = new SeatTracker({ ...defaultBoat, seats: 8 });
  const detections = Array.from({ length: 8 }, (_, i) => ({
    box: { x: 20 + i * 90, y: 100, width: 96, height: 280, confidence: 0.9 },
    points: poses[0],
  }));
  timings.push(
    time("SeatTracker.update (8 seats)", 40, () =>
      tracker.update(detections, 0),
    ),
  );

  return {
    equivalence: {
      "crop tensor, unflipped": plainDiff,
      "crop tensor, flipped": flipDiff,
      "letterboxed detector tensor": diff(legacyDetect, fastDetect),
    },
    timings,
    plan: {
      off: passPlan("off", detections[0].box, W, H).length,
      adaptiveAllPasses: passPlan("adaptive", detections[0].box, W, H).length,
      always: passPlan("always", detections[0].box, W, H).length,
      gateOnCleanPose: assessPose(poses[0]).accepted,
      drawPlanLetterbox: drawPlan(1920, 1080, 640, 640, null),
    },
    cores: navigator.hardwareConcurrency,
  };
});

const identical = Object.values(report.equivalence).every((d) => d === 0);
console.log(JSON.stringify(report, null, 1));
console.log(
  identical
    ? "OK: the bitmap path is bit-identical to the per-pass VideoFrame path."
    : "FAIL: rasterisation differs from the reference path; the fast path must not ship.",
);
await browser.close();
process.exit(identical ? 0 : 1);
