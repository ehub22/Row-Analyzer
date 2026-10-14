import { test, expect } from "@playwright/test";
import { readFile, writeFile } from "node:fs/promises";
import type { Session } from "../../src/core/types";

test("authentic weights: download, WASM inference, reference tracking and offline cache", async ({
  page,
}, info) => {
  test.skip(
    process.env.REAL_MODEL_TEST !== "1",
    "Requires pinned weights and the transient public reference fixture.",
  );
  test.setTimeout(600000);
  const truth = JSON.parse(
    await readFile("private/reference/truth.json", "utf8"),
  );
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await page
    .locator("input[type=file]")
    .first()
    .setInputFiles("private/reference/reference.mp4");
  await page.getByLabel("Boat class").selectOption("Single");
  await page.getByRole("button", { name: "Models & settings" }).click();
  await page
    .getByText("Force WASM compatibility mode", { exact: true })
    .click();
  await page
    .getByText("Flip + multi-scale test-time augmentation", { exact: true })
    .click();
  await page.getByRole("button", { name: "Download / restore models" }).click();
  await expect(page.locator(".model-downloads")).toContainText(
    "Weights ready",
    { timeout: 120000 },
  );
  await page.getByText("Save processing settings").click();
  await page
    .getByRole("button", { name: "Analyze video", exact: true })
    .click();
  await expect(
    page.locator(".stage-badges").or(page.locator(".alert.error")),
  ).toBeVisible({ timeout: 400000 });
  expect(await page.locator(".alert.error").allTextContents()).toEqual([]);
  await page.getByRole("button", { name: "Export", exact: true }).click();
  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "JSON session", exact: true }).click();
  const file = info.outputPath("reference-session.json");
  await (await download).saveAs(file);
  const session: Session = JSON.parse(await readFile(file, "utf8"));
  expect(session.demo).toBe(false);
  expect(session.provider).toBe("WASM / WASM");
  expect(session.frames.length).toBe(8);
  expect(
    session.frames.every((f) => f.seat === 1 && f.identityConfidence > 0.7),
  ).toBe(true);
  expect(session.strokes).toHaveLength(0); // Translating a still image must not invent rowing strokes.
  const distances: number[] = [];
  let visible = 0,
    observed = 0;
  for (const f of session.frames) {
    const index = Math.round(f.time * 10),
      offset = -24 + index * 3;
    for (let j = 0; j < 17; j++) {
      const [x, y, v] = truth.annotation.keypoints.slice(j * 3, j * 3 + 3);
      if (v !== 2) continue;
      visible++;
      const p = f.points[j];
      if (p.origin !== "observed") {
        distances.push(Infinity);
        continue;
      }
      observed++;
      distances.push(
        Math.hypot(p.x - (x + offset), p.y - y) /
          Math.max(truth.annotation.bbox[2], truth.annotation.bbox[3]),
      );
    }
  }
  const pck = distances.filter((d) => d <= 0.1).length / distances.length;
  const report = {
    scope: truth.scope,
    frames: session.frames.length,
    visibleLabels: visible,
    observedLabels: observed,
    pckAt01Bbox: pck,
    meanNormalizedError: distances.every(Number.isFinite)
      ? distances.reduce((a, b) => a + b, 0) / distances.length
      : null,
    identitySwitches: 0,
    provider: session.provider,
    rowAccuracyValidated: false,
  };
  await writeFile(
    "test-results/reference-validation.json",
    JSON.stringify(report),
  );
  expect(pck).toBeGreaterThanOrEqual(0.8);
  expect(observed / visible).toBeGreaterThanOrEqual(0.8);
  expect(errors).toEqual([]);
  // New page state, model network deliberately blocked: only a real persistent cache can restore.
  await page.reload();
  await page.route("**/models/*.onnx", (route) => route.abort());
  await page.getByRole("button", { name: "Models & settings" }).click();
  await page.getByRole("button", { name: "Download / restore models" }).click();
  await expect(page.locator(".model-downloads")).toContainText(
    "Weights ready",
    { timeout: 120000 },
  );
  await page.getByRole("button", { name: "Clear model cache" }).click();
  await expect(page.locator(".model-downloads")).toContainText(
    "Cached weights removed",
  );
});
