// Headless run of the real analysis pipeline with the generated TEST-DOUBLE
// models, to inspect which execution provider each model ends up on and how
// the GPU path fails. Developer diagnostic only — not part of the app.
//
//   URL=http://localhost:5173 GPU=1 node scripts/gpu-analysis-check.mjs
//     GPU=1  launch Chromium with a (software) WebGPU adapter available
//     GPU=0  launch without it, reproducing "no adapter on this machine"
//     FORCE_WASM=1  tick "Force WASM compatibility mode" in the UI
import { chromium } from "playwright";
import { resolveChromium } from "./lib/chromium.mjs";

const url = process.env.URL ?? "http://localhost:5173";
const wantGpu = process.env.GPU !== "0";
const forceWasm = process.env.FORCE_WASM === "1";
const video = process.env.VIDEO ?? "private/test-models/short.mp4";
const detector = process.env.DETECTOR ?? "private/test-models/TEST-ONLY-detector.onnx";
const pose = process.env.POSE ?? "private/test-models/TEST-ONLY-pose.onnx";

const { executablePath, args, env } = await resolveChromium();
const launchArgs = wantGpu ? args : args.filter((a) => a !== "--enable-unsafe-webgpu");
const browser = await chromium.launch({ executablePath, args: launchArgs, env, headless: true });
const page = await browser.newPage();
const logs = [];
page.on("console", (m) => logs.push(`[console.${m.type()}] ${m.text()}`));
page.on("pageerror", (e) => logs.push(`[pageerror] ${e.message}`));
page.on("requestfailed", (r) => logs.push(`[requestfailed] ${r.url()} ${r.failure()?.errorText}`));
const wasmAssets = new Map();
page.on("request", (r) => {
  const url = r.url();
  if (/ort-wasm[^/]*\.(wasm|mjs)/.test(url))
    wasmAssets.set(url.replace(/^https?:\/\/[^/]+/, ""), r.resourceType());
});

await page.goto(url);
const gpu = await page.evaluate(`(${async () => {
  const out = { hasGPU: !!navigator.gpu, secure: isSecureContext, adapter: null, err: null };
  if (!navigator.gpu) return out;
  try {
    const a = await navigator.gpu.requestAdapter();
    out.adapter = a ? { vendor: a.info?.vendor, arch: a.info?.architecture } : null;
  } catch (e) {
    out.err = String(e);
  }
  return out;
}})()`);
console.log("PAGE WEBGPU:", JSON.stringify(gpu));

await page.locator("input[type=file]").first().setInputFiles(video);
await page.getByLabel("Boat class").selectOption("Single");
await page.getByRole("button", { name: "Models & settings" }).click();
await page.getByLabel("Detector model", { exact: true }).setInputFiles(detector);
await page.getByLabel("Pose model", { exact: true }).setInputFiles(pose);
if (forceWasm) await page.getByText("Force WASM compatibility mode", { exact: true }).click();
await page.getByText("Save processing settings").click();
await page.getByRole("button", { name: "Analyze video", exact: true }).click();

const statusTicker = setInterval(async () => {
  try {
    const status = await page.locator(".progress-block").innerText();
    console.log("[status]", status.replace(/\n/g, " | "));
  } catch {
    console.log("[status] (no progress block)");
  }
}, 3000);

await page
  .waitForFunction(
    () =>
      document.querySelector(".alert.error") ||
      document.querySelector(".analysis-notes") ||
      document.querySelector(".demo-notice"),
    {},
    { timeout: Number(process.env.TIMEOUT ?? 300000) },
  )
  .catch(async () => {
    console.log("TIMEOUT — page state:");
    console.log("  progress:", await page.locator(".progress-block").innerText().catch(() => "none"));
    console.log("  buttons:", await page.locator("button").allInnerTexts());
  });
clearInterval(statusTicker);

const error = await page.locator(".alert.error").count();
if (error) console.log("ERROR:", await page.locator(".alert.error").innerText());
else {
  await page.getByRole("button", { name: "Export", exact: true }).click();
  const pending = page.waitForEvent("download");
  await page.getByRole("button", { name: "JSON session", exact: true }).click();
  const file = await pending;
  const path = await file.path();
  const session = JSON.parse(await (await import("node:fs/promises")).readFile(path, "utf8"));
  console.log("DONE. provider:", session.provider);
  console.log(
    "DONE. workspace runtime line:",
    (await page.locator(".runtime-line").innerText().catch(() => "(absent)")).replace(/\n/g, " "),
  );
  console.log("DONE. model:", session.model);
  console.log("Warnings:");
  session.warnings.forEach((w) => console.log("  -", w));
}
console.log("ORT WASM REQUESTS:");
for (const [url, type] of [...wasmAssets].sort())
  console.log(`   ${type.padEnd(10)} ${url}`);
console.log("LOGS:");
logs.forEach((l) => console.log("  ", l));
await browser.close();
