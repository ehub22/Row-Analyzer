// Verify the worker's cropped-bitmap input against the full-frame rasterizer
// on actual decoded video pixels. No footage, frames or screenshots are saved.
// Run with the Vite dev server: VIDEO=5.mov node scripts/crop-equivalence-check.mjs
import { chromium } from "playwright";
import { resolveChromium } from "./lib/chromium.mjs";

const { executablePath, args, env } = await resolveChromium();
const browser = await chromium.launch({
  executablePath,
  args,
  env,
  headless: true,
});
try {
  const page = await browser.newPage();
  await page.goto(process.env.URL ?? "http://localhost:5173");
  await page.evaluate(() => {
    const input = document.createElement("input");
    input.type = "file";
    input.id = "crop-check-file";
    document.body.append(input);
  });
  await page
    .locator("#crop-check-file")
    .setInputFiles(process.env.VIDEO ?? "5.mov");
  const result = await page.evaluate(async () => {
    const { decodeVideo } = await import("/src/core/video.ts");
    const { Rasterizer, transferCrop } =
      await import("/src/core/preprocess.ts");
    const { paddedCrop } = await import("/src/core/detection.ts");
    const file = document.querySelector("#crop-check-file").files[0];
    const boxes = [
      { x: 550, y: 460, width: 180, height: 220, confidence: 1 },
      { x: 900, y: 450, width: 180, height: 220, confidence: 1 },
      { x: 1470, y: 470, width: 180, height: 220, confidence: 1 },
      { x: -20, y: 200, width: 170, height: 240, confidence: 1 },
      { x: 1810, y: 960, width: 170, height: 200, confidence: 1 },
    ];
    const whole = new Rasterizer(288, 384, "pose");
    const cropped = new Rasterizer(288, 384, "pose");
    let frames = 0,
      cases = 0,
      different = 0,
      maxDifference = 0;
    const stop = new Error("Enough frames for the crop check");
    try {
      await decodeVideo(
        file,
        async (frame) => {
          if (frames >= 5) throw stop;
          const bitmap = await createImageBitmap(frame);
          try {
            for (const box of boxes) {
              const crop = paddedCrop(box, 288, 384, 1.25);
              const bounds = transferCrop(crop, bitmap.width, bitmap.height);
              if (!bounds) throw Error("Test crop has no source pixels.");
              const cut = await createImageBitmap(
                bitmap,
                bounds.sx,
                bounds.sy,
                bounds.sw,
                bounds.sh,
              );
              try {
                for (const flip of [false, true]) {
                  const reference = whole.tensor(
                    bitmap,
                    bitmap.width,
                    bitmap.height,
                    crop,
                    flip,
                  );
                  const candidate = cropped.tensor(
                    cut,
                    cut.width,
                    cut.height,
                    bounds.localCrop,
                    flip,
                  );
                  for (let i = 0; i < reference.length; i++) {
                    if (reference[i] !== candidate[i]) {
                      different++;
                      maxDifference = Math.max(
                        maxDifference,
                        Math.abs(reference[i] - candidate[i]),
                      );
                    }
                  }
                  cases++;
                }
              } finally {
                cut.close();
              }
            }
            frames++;
          } finally {
            bitmap.close();
          }
        },
        () => {},
      );
    } catch (error) {
      if (error !== stop) throw error;
    }
    return { frames, cases, different, maxDifference };
  });
  console.log(JSON.stringify(result));
  if (result.frames !== 5 || result.cases !== 50 || result.different !== 0)
    process.exitCode = 1;
} finally {
  await browser.close();
}
