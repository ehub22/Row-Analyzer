import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "./tests/e2e",
  timeout: 30000,
  use: {
    baseURL: process.env.TEST_URL || "http://localhost:4173",
    viewport: { width: 1440, height: 1000 },
    launchOptions: process.env.PLAYWRIGHT_EXECUTABLE_PATH
      ? {
          executablePath: process.env.PLAYWRIGHT_EXECUTABLE_PATH,
          args: [
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--use-angle=swiftshader",
            "--enable-unsafe-swiftshader",
          ],
        }
      : {},
  },
  webServer: process.env.TEST_URL
    ? undefined
    : {
        command: "npm run preview -- --port 4173",
        url: "http://localhost:4173",
        reuseExistingServer: !process.env.CI,
      },
});
