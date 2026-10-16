// Developer-only end-to-end browser benchmark. The default models are CONSTANT
// TEST-DOUBLE graphs: they exercise decode → detect → pose → track, but do not
// measure provided-weight inference speed or validate detection/pose accuracy.
// Run a local Vite server first (npm run dev), then for example:
//   GPU=0 FORCE_WASM=1 VIDEO=5.mov BOAT_CLASS=Quad REGION=0,0,100,100 \
//     DETECTOR_CADENCE=every-frame TTA_MODE=off WORKERS=2 \
//     node scripts/gpu-analysis-check.mjs
// The footage is a quad: use make_test_models.py --people 4 for a fixture run.
// For provided-weight measurements, pass DETECTOR and POSE paths explicitly;
// this harness does not authenticate their provenance or validate accuracy.
// No screenshots or copies of the source video are produced.
import { readFile } from "node:fs/promises";
import { performance } from "node:perf_hooks";
import { chromium } from "playwright";
import { resolveChromium } from "./lib/chromium.mjs";

const url = process.env.URL ?? "http://localhost:5173";
const wantGpu = process.env.GPU !== "0";
const forceWasm = process.env.FORCE_WASM === "1";
const video = process.env.VIDEO ?? "5.mov";
const detector =
  process.env.DETECTOR ?? "private/test-models/TEST-ONLY-detector4.onnx";
const pose = process.env.POSE ?? "private/test-models/TEST-ONLY-pose.onnx";
const boatClass = process.env.BOAT_CLASS ?? "Quad";
const cadence = process.env.DETECTOR_CADENCE ?? "every-frame";
const ttaMode = process.env.TTA_MODE ?? "adaptive";
const poseJoints = Number(process.env.POSE_JOINTS ?? 133);
const poseFormat = process.env.POSE_FORMAT ?? "simcc";
const poseWidth = Number(
  process.env.POSE_WIDTH ?? (poseJoints === 17 ? 192 : 288),
);
const poseHeight = Number(
  process.env.POSE_HEIGHT ?? (poseJoints === 17 ? 256 : 384),
);
const workers = Number(process.env.WORKERS ?? 0);
const timeout = Number(process.env.TIMEOUT ?? 1_200_000);
const regionText = process.env.REGION ?? "0,0,100,100";
const region = regionText.split(",").map((part) => Number(part.trim()));
if (
  region.length !== 4 ||
  region.some((value) => !Number.isFinite(value) || value < 0 || value > 100) ||
  region[0] >= region[2] ||
  region[1] >= region[3]
)
  throw Error(
    "REGION must be left,top,right,bottom percentages, with left < right and top < bottom",
  );
if (!["every-frame", "auto"].includes(cadence))
  throw Error("DETECTOR_CADENCE must be every-frame or auto");
if (!["off", "adaptive", "always"].includes(ttaMode))
  throw Error("TTA_MODE must be off, adaptive, or always");
if (![17, 133].includes(poseJoints))
  throw Error("POSE_JOINTS must be 17 or 133");
if (!["simcc", "heatmap"].includes(poseFormat))
  throw Error("POSE_FORMAT must be simcc or heatmap");
if (![poseWidth, poseHeight].every((n) => Number.isInteger(n) && n >= 128))
  throw Error("POSE_WIDTH/POSE_HEIGHT must be integers at least 128");
if (![0, 1, 2, 3, 4, 6, 8].includes(workers))
  throw Error("WORKERS must be one of the UI choices: 0, 1, 2, 3, 4, 6, 8");
if (!Number.isFinite(timeout) || timeout <= 0)
  throw Error("TIMEOUT must be positive milliseconds");

const config = {
  url,
  video,
  detector,
  pose,
  boatClass,
  region: {
    left: region[0],
    top: region[1],
    right: region[2],
    bottom: region[3],
  },
  cadence,
  ttaMode,
  poseJoints,
  poseFormat,
  poseWidth,
  poseHeight,
  workers,
  gpuRequested: wantGpu,
  forceWasm,
  timeoutMs: timeout,
};
const errors = [];
const wasmAssets = new Map();
let browser;
let statusTicker;
let summary = {
  config,
  testDoubleBenchmark: /TEST-ONLY/.test(detector) && /TEST-ONLY/.test(pose),
  outcome: "error",
  errors,
};
try {
  const { executablePath, args, env } = await resolveChromium();
  const launchArgs = wantGpu
    ? args
    : args.filter((arg) => arg !== "--enable-unsafe-webgpu");
  browser = await chromium.launch({
    executablePath,
    args: launchArgs,
    env,
    headless: true,
  });
  summary.browserVersion = browser.version();
  summary.nodeVersion = process.version;
  const page = await browser.newPage({ acceptDownloads: true });
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(`[console] ${message.text()}`);
  });
  page.on("pageerror", (error) =>
    errors.push(`[pageerror] ${error.stack ?? error.message}`),
  );
  page.on("requestfailed", (request) =>
    errors.push(
      `[requestfailed] ${request.url()} ${request.failure()?.errorText}`,
    ),
  );
  page.on("request", (request) => {
    const asset = request.url();
    if (/ort-wasm[^/]*\.(wasm|mjs)/.test(asset))
      wasmAssets.set(
        asset.replace(/^https?:\/\/[^/]+/, ""),
        request.resourceType(),
      );
  });

  await page.goto(url);
  const gpu = await page.evaluate(async () => {
    const out = {
      hasGPU: !!navigator.gpu,
      secure: isSecureContext,
      adapter: null,
      error: null,
    };
    if (!navigator.gpu) return out;
    try {
      const adapter = await navigator.gpu.requestAdapter();
      out.adapter = adapter
        ? { vendor: adapter.info?.vendor, arch: adapter.info?.architecture }
        : null;
    } catch (error) {
      out.error = String(error);
    }
    return out;
  });
  summary.gpu = gpu;
  console.log("PAGE WEBGPU:", JSON.stringify(gpu));

  await page.locator("input[type=file]").first().setInputFiles(video);
  await page.getByLabel("Boat class").selectOption(boatClass);
  await page.getByRole("button", { name: "Models & settings" }).click();
  await page
    .getByLabel("Pose joint layout", { exact: true })
    .selectOption(String(poseJoints));
  await page.getByLabel("Pose output").selectOption(poseFormat);
  await page
    .getByLabel("Detector model", { exact: true })
    .setInputFiles(detector);
  await page.getByLabel("Pose model", { exact: true }).setInputFiles(pose);
  await page.getByLabel("Input width", { exact: true }).fill(String(poseWidth));
  await page
    .getByLabel("Input height", { exact: true })
    .fill(String(poseHeight));
  const cadenceControl = page.getByLabel("Detector cadence", { exact: true });
  if (await cadenceControl.count()) {
    await cadenceControl.selectOption(cadence);
    summary.cadenceControl = "selected";
  } else if (cadence === "every-frame") {
    // Pre-cadence builds always detect each decoded frame. Never silently
    // accept an absent control for an auto benchmark.
    summary.cadenceControl = "absent (legacy every-frame behavior)";
  } else {
    throw Error(
      "DETECTOR_CADENCE=auto requested but Detector cadence control is absent",
    );
  }
  const ttaControl = page.getByLabel("Test-time augmentation", { exact: true });
  const workersControl = page.getByLabel("Parallel pose workers", {
    exact: true,
  });
  await ttaControl.selectOption(ttaMode);
  await workersControl.selectOption(String(workers));
  await page.locator("details.region-settings summary").click();
  for (const [key, value] of Object.entries(config.region))
    await page.getByLabel(`${key} (%)`, { exact: true }).fill(String(value));
  const wasmControl = page.getByRole("checkbox", {
    name: "Force WASM compatibility mode",
  });
  await wasmControl.setChecked(forceWasm);
  // Capture what the UI actually selected rather than inferring it from env vars.
  summary.selectedSettings = {
    detectorExport: await page.getByLabel("Detector export").inputValue(),
    poseOutput: await page.getByLabel("Pose output").inputValue(),
    poseJoints: Number(await page.getByLabel("Pose joint layout").inputValue()),
    inputWidth: Number(await page.getByLabel("Input width").inputValue()),
    inputHeight: Number(await page.getByLabel("Input height").inputValue()),
    cadence: (await cadenceControl.count())
      ? await cadenceControl.inputValue()
      : "every-frame",
    ttaMode: await ttaControl.inputValue(),
    workers: Number(await workersControl.inputValue()),
    parallelPlan: await page
      .getByLabel("Parallel processing plan")
      .inputValue(),
    forceWasm: await wasmControl.isChecked(),
  };
  if (
    summary.selectedSettings.ttaMode !== ttaMode ||
    summary.selectedSettings.poseOutput !== poseFormat ||
    summary.selectedSettings.poseJoints !== poseJoints ||
    summary.selectedSettings.workers !== workers ||
    summary.selectedSettings.forceWasm !== forceWasm ||
    summary.selectedSettings.cadence !== cadence
  )
    throw Error("Selected processing settings do not match requested settings");
  await page.getByText("Save processing settings").click();

  const analyzeStart = performance.now();
  await page
    .getByRole("button", { name: "Analyze video", exact: true })
    .click();
  statusTicker = setInterval(async () => {
    try {
      const status = await page.locator(".progress-block").innerText();
      console.log("[status]", status.replace(/\n/g, " | "));
    } catch {
      /* page may already be complete or closing */
    }
  }, 15_000);
  try {
    await Promise.race([
      page.waitForFunction(
        () =>
          document.querySelector(".alert.error, .analysis-notes, .demo-notice"),
        null,
        { timeout },
      ),
      new Promise((_, reject) => page.once("pageerror", reject)),
      new Promise((_, reject) =>
        page.once("crash", () => reject(Error("Benchmark page crashed"))),
      ),
    ]);
  } catch (error) {
    if (error?.name !== "TimeoutError") throw error;
    const status = await page
      .locator(".progress-block")
      .innerText()
      .catch(() => "none");
    throw Error(`Analysis timed out after ${timeout} ms; progress: ${status}`, {
      cause: error,
    });
  }
  summary.analyzeToResultSeconds = +(
    (performance.now() - analyzeStart) /
    1000
  ).toFixed(3);
  if (await page.locator(".alert.error").count())
    throw Error(await page.locator(".alert.error").innerText());
  if (!(await page.locator(".analysis-notes").count()))
    throw Error("Unexpected demo result instead of video analysis");

  // Export only the analysis JSON to Playwright's temporary download file;
  // never save a screenshot or the private source footage.
  await page.getByRole("button", { name: "Export", exact: true }).click();
  const pending = page.waitForEvent("download");
  await page.getByRole("button", { name: "JSON session", exact: true }).click();
  const download = await pending;
  const session = JSON.parse(await readFile(await download.path(), "utf8"));
  if (session.poseJoints !== poseJoints)
    throw Error(
      `Exported pose layout ${session.poseJoints} differs from selected ${poseJoints}`,
    );
  const first = session.frames[0];
  if (!first || first.points.length !== 133)
    throw Error("Exported session lost its 133-slot point layout");
  const unavailable = first.points
    .slice(17)
    .filter(
      (point) =>
        point.origin === "missing" &&
        point.confidence === 0 &&
        point.flags?.includes("Not provided by the selected body-only model"),
    ).length;
  if (poseJoints === 17 && unavailable !== 116)
    throw Error(
      `Expected 116 explicitly unavailable joints, found ${unavailable}`,
    );
  const perf = session.performance;
  if (!perf || !Number.isFinite(perf.frames) || perf.frames < 1)
    throw Error("Analysis result lacks measured processed frame count");
  const detectorStage = perf.stages.find(({ stage }) => stage === "detector");
  const decodeWarning = session.warnings.find((warning) =>
    warning.startsWith("Decoder emitted "),
  );
  const encodedMatch = decodeWarning?.match(
    /^Decoder emitted \d+ of (\d+) encoded samples/,
  );
  // This report is under a collapsed <details> by default; expand it as a
  // viewer would before reading the actually displayed text.
  await page.locator(".analysis-notes summary").click();
  const displayedSummary = await page.locator(".performance-line").innerText();
  summary = {
    ...summary,
    outcome: "success",
    provider: session.provider,
    model: session.model,
    layoutCheck: {
      poseJoints: session.poseJoints,
      slots: first.points.length,
      unavailable,
    },
    userVisibleTotal: displayedSummary.replace(/\s+/g, " ").trim(),
    runtimeLine: (await page.locator(".runtime-line").innerText())
      .replace(/\s+/g, " ")
      .trim(),
    videoSeconds: session.duration,
    decodedCounts: {
      encodedSamples: encodedMatch
        ? Number(encodedMatch[1])
        : session.timestamps.length,
      timestamps: session.timestamps.length,
      processedFrames: perf.frames,
      seatFrames: session.frames.length,
      detectorCalls: detectorStage?.count ?? null,
    },
    performance: perf,
    warnings: session.warnings,
    wasmRequests: [...wasmAssets].sort(([a], [b]) => a.localeCompare(b)),
  };
  console.log("DONE. provider:", session.provider);
  console.log("DONE. user-visible total:", summary.userVisibleTotal);
  console.log("DONE. decoded counts:", JSON.stringify(summary.decodedCounts));
} catch (error) {
  summary.error = error instanceof Error ? error.message : String(error);
  if (!summary.analyzeToResultSeconds) summary.analyzeToResultSeconds = null;
  process.exitCode = 1;
} finally {
  if (statusTicker) clearInterval(statusTicker);
  await browser?.close();
  console.log("BENCHMARK_JSON", JSON.stringify(summary, null, 2));
}
