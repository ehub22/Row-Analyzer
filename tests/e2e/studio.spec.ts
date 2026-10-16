import { test, expect } from "@playwright/test";
test("empty workspace does not fabricate results", async ({ page }) => {
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "Every stroke, understood." }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Export", exact: true }),
  ).toBeDisabled();
  await expect(page.locator(".metric-card").first()).toContainText(
    "Awaiting analysis",
  );
  await page.getByRole("button", { name: "Models & settings" }).click();
  await expect(page.getByRole("dialog")).toContainText(
    "Model weights are not bundled",
  );
  await page.getByRole("button", { name: "Close dialog" }).click();
});
test("synthetic demo, per-seat review and exact frame navigation", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByText("Explore a synthetic demo").click();
  await expect(page.locator(".demo-notice")).toContainText(
    "not a tracking accuracy benchmark",
  );
  await expect(page.locator(".metric-card").first()).toContainText("25.0");
  await page.getByRole("button", { name: "Next frame", exact: true }).click();
  await expect(page.locator(".timecode")).toContainText("00:00.03");
  await page.locator(".seat-strip button").nth(3).click();
  await expect(page.locator(".stage-badges")).toContainText("SEAT 4");
  await page.getByRole("button", { name: "Crew timing", exact: true }).click();
  await expect(page.locator("tbody tr")).toHaveCount(21);
});
test("manual correction produces labels without labeling inferred neighbors as ground truth", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByText("Explore a synthetic demo").click();
  await page
    .getByRole("button", { name: "Keyframe labels", exact: true })
    .click();
  await page.locator("canvas").click({ position: { x: 200, y: 150 } });
  await expect(
    page.getByRole("button", { name: "Export labels", exact: true }),
  ).toBeEnabled();
  const pending = page.waitForEvent("download");
  await page
    .getByRole("button", { name: "Export labels", exact: true })
    .click();
  expect((await pending).suggestedFilename()).toBe("rowing-labels.json");
  await page
    .getByRole("button", { name: "Reset corrections", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "Export labels", exact: true }),
  ).toBeDisabled();
});
test("exports reports and compares imported sessions", async ({
  page,
}, info) => {
  await page.goto("/");
  await page.getByText("Explore a synthetic demo").click();
  await page.getByRole("button", { name: "Export", exact: true }).click();
  const pdf = page.waitForEvent("download");
  await page
    .getByRole("button", { name: "PDF coaching report", exact: true })
    .click();
  expect((await pdf).suggestedFilename()).toBe("row-analysis-report.pdf");
  await page.getByRole("button", { name: "Export", exact: true }).click();
  const json = page.waitForEvent("download");
  await page.getByRole("button", { name: "JSON session", exact: true }).click();
  const path = info.outputPath("session.json");
  await (await json).saveAs(path);
  await page.locator("input[type=file]").nth(2).setInputFiles(path);
  await expect(page.locator(".compare-page")).toContainText(
    "Rate difference: 0.0 spm",
  );
});
test("malformed sessions fail without crashing the workspace", async ({
  page,
}) => {
  await page.goto("/");
  await page
    .locator("input[type=file]")
    .nth(1)
    .setInputFiles({
      name: "bad.json",
      mimeType: "application/json",
      buffer: Buffer.from('{"version":1}'),
    });
  await expect(page.getByRole("alert")).toContainText(
    "Invalid Row Analyzer session",
  );
  await expect(
    page.getByRole("heading", { name: "Every stroke, understood." }),
  ).toBeVisible();
});
test("layout remains usable at a narrow viewport", async ({ page }) => {
  await page.setViewportSize({ width: 600, height: 900 });
  await page.goto("/");
  await expect(
    page.getByRole("button", { name: "Import rowing video", exact: true }),
  ).toBeVisible();
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth),
  ).toBeLessThanOrEqual(600);
});
test("models dialog exposes parallel worker controls and the live plan", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Models & settings" }).click();
  const workers = page.getByLabel("Parallel pose workers");
  await expect(workers).toBeVisible();
  await expect(workers).toHaveValue("0"); // Auto by default
  const plan = page.getByLabel("Parallel processing plan");
  await expect(plan).toHaveValue(/pose workers?/);
  await expect(plan).toHaveValue(/logical cores/);
  await workers.selectOption("2");
  await expect(plan).toHaveValue(/2 pose workers/);
  const cadence = page.getByLabel("Detector cadence");
  await expect(cadence).toHaveValue("every-frame");
  await cadence.selectOption("auto");
  await expect(cadence).toHaveValue("auto");
  const layout = page.getByLabel("Pose joint layout");
  await expect(layout).toHaveValue("133");
  await layout.selectOption("17");
  await expect(page.getByLabel("Input width")).toHaveValue("192");
  await expect(page.getByLabel("Input height")).toHaveValue("256");
  await expect(page.getByText(/COCO-17 requires a local file/)).toBeVisible();
  await expect(
    page.getByText(/Detector and pose inference overlap/),
  ).toBeVisible();
  await page.getByRole("button", { name: "Close dialog" }).click();
});
test("model cache panel reports no-URL builds and keeps local import as fallback", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Models & settings" }).click();
  const panel = page.getByLabel("Automatic model cache");
  await expect(panel).toContainText("YOLOX-L person detector");
  await expect(panel).toContainText("No download URL in this build");
  await expect(panel.getByRole("button", { name: "Download now" })).toHaveCount(
    0,
  );
  await expect(
    page.getByLabel("Detector model", { exact: true }),
  ).toBeAttached();
  await expect(page.getByLabel("Pose model", { exact: true })).toBeAttached();
  await page.getByRole("button", { name: "Clear cached models" }).click();
  await expect(page.getByText("Cached model files removed.")).toBeVisible();
  await page.getByRole("button", { name: "Close dialog" }).click();
});
