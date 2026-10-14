import { chromium } from "playwright";
import { resolveChromium } from "./lib/chromium.mjs";
const { executablePath, args, env } = await resolveChromium();
const browser = await chromium.launch({ executablePath, args, env, headless: true });
const page = await browser.newPage({ viewport: { width: 1440, height: 1050 } });
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));
page.on("console", (m) => {
  if (m.type() === "error") console.log("CONSOLE", m.text());
});
await page.goto("http://localhost:5173");
await page.screenshot({ path: ".cache/workspace.png", fullPage: true });
await page.getByText("Explore a synthetic demo").click();
await page.waitForTimeout(500);
console.log("demo text", await page.locator(".metrics-row").innerText());
await page.screenshot({ path: ".cache/demo.png", fullPage: true });
await page.getByRole("button", { name: "Crew timing", exact: true }).click();
console.log("crew rows", await page.locator("tbody tr").count());
await page
  .getByRole("button", { name: "Keyframe labels", exact: true })
  .click();
await page.getByRole("button", { name: "Export", exact: true }).click();
const d = page.waitForEvent("download");
await page.getByRole("button", { name: "PDF coaching report" }).click();
console.log("download", (await d).suggestedFilename());
console.log("errors", errors);
await browser.close();
