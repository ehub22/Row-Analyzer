import { chromium } from "playwright";
import chrome from "@sparticuz/chromium";
const browser = await chromium.launch({
  executablePath: await chrome.executablePath(),
  args: chrome.args,
  headless: true,
});
const page = await browser.newPage();
page.on("requestfailed", (r) =>
  console.log("REQUEST FAILED", r.url(), r.failure()),
);
page.on("pageerror", (e) => console.log("ERROR", e.message));
page.on("console", (m) => {
  if (m.type() === "error") console.log("CONSOLE", m.text());
});
await page.goto("http://localhost:4173");
await page
  .locator("input[type=file]")
  .first()
  .setInputFiles("private/test-models/short.mp4");
await page.getByLabel("Boat class").selectOption("Single");
await page.getByRole("button", { name: "Configure whole-body models" }).click();
await page
  .getByLabel("Detector model", { exact: true })
  .setInputFiles("private/test-models/TEST-ONLY-detector.onnx");
await page
  .getByLabel("Pose model", { exact: true })
  .setInputFiles("private/test-models/TEST-ONLY-pose.onnx");
await page.getByText("Force WASM compatibility mode", { exact: true }).click();
await page.getByText("Save processing settings").click();
await page.getByRole("button", { name: "Analyze video", exact: true }).click();
await page
  .waitForFunction(
    () =>
      document.querySelector(".alert.error") ||
      document.querySelector(".stage-badges"),
    {},
    { timeout: 60000 },
  )
  .catch(async () => {
    console.log(
      "TIMEOUT STATUS",
      await page.locator(".progress-block").innerText(),
    );
    await page.screenshot({ path: ".cache/worker.png" });
  });
console.log(
  (await page.locator(".alert.error").count())
    ? await page.locator(".alert.error").innerText()
    : (await page.locator(".stage-badges").count())
      ? "SUCCESS: full decode → WASM detector/pose test doubles → seat tracking → report; no genuine tracking validation"
      : "INCOMPLETE",
);
console.log(await page.locator(".metrics-row").innerText());
await browser.close();
