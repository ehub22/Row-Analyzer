import { chromium } from "playwright";
import { resolveChromium } from "./lib/chromium.mjs";
const { executablePath, args, env } = await resolveChromium();
const browser = await chromium.launch({ executablePath, args, env, headless: true });
const page = await browser.newPage();
await page.goto("http://localhost:5173");
for (const name of ["1.mov", "3.mov"]) {
  const result = await page.evaluate(async (name) => {
    const { decodeVideo } = await import("/src/core/video.ts");
    const response = await fetch(`/google_drive/${name}`);
    const file = new File([await response.blob()], name);
    const times = [];
    let count = 0,
      first = Infinity,
      last = -Infinity,
      previous = -Infinity,
      monotonic = true;
    try {
      const meta = await decodeVideo(
        file,
        async (frame, t) => {
          count++;
          times.push(t);
          first = Math.min(first, t);
          last = Math.max(last, t);
          if (t < previous) monotonic = false;
          previous = t;
        },
        () => {},
      );
      return {
        name,
        count,
        first,
        last,
        monotonic,
        width: meta.width,
        height: meta.height,
        codec: meta.config.codec,
        samples: meta.samples.length,
        missing: meta.samples
          .filter(
            (s) =>
              !times.some(
                (t) =>
                  Math.abs(t - (s.cts / s.timescale - meta.origin)) < 0.0001,
              ),
          )
          .map((s) => s.cts / s.timescale),
        firstSamples: meta.samples
          .slice(0, 4)
          .map((s) => ({ cts: s.cts, dts: s.dts, sync: s.is_sync })),
        duration: meta.duration,
      };
    } catch (e) {
      return { name, error: String(e), count };
    }
  }, name);
  console.log(JSON.stringify(result));
}
await browser.close();
