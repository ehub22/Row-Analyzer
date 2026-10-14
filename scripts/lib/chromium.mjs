// Chromium launcher for the local dev scripts (scripts/*.mjs).
//
// CI installs its own Chromium (`npx playwright install --with-deps chromium`)
// and passes PLAYWRIGHT_EXECUTABLE_PATH. In minimal containers there is no
// system Chromium, no apt, and the only available build is the one bundled
// with @sparticuz/chromium — which needs (a) the Amazon Linux 2023 NSS/NSPR
// shared libraries and (b) SwiftShader unpacked next to it. Both ship inside
// the npm package as brotli-compressed tarballs, so they can be extracted
// offline into .cache/ (gitignored).
import { createRequire } from "node:module";
import { execFileSync } from "node:child_process";
import { brotliDecompressSync } from "node:zlib";
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..", "..");
const cacheRoot = join(root, ".cache", "chromium");
const libDir = join(cacheRoot, "lib");
const shaderDir = join(cacheRoot, "swiftshader");

function unpack(archive, into, marker) {
  if (existsSync(join(into, marker))) return;
  mkdirSync(into, { recursive: true });
  const tar = join(into, "payload.tar");
  writeFileSync(tar, brotliDecompressSync(readFileSync(archive)));
  try {
    execFileSync("tar", ["xf", tar, "-C", into]);
  } finally {
    rmSync(tar, { force: true });
  }
}

/** @returns {Promise<{executablePath: string, args: string[], env: NodeJS.ProcessEnv}>} */
export async function resolveChromium() {
  const require = createRequire(join(root, "package.json"));
  const override = process.env.PLAYWRIGHT_EXECUTABLE_PATH;
  const chromeMod = override ? null : require("@sparticuz/chromium");
  const chrome = chromeMod ? (chromeMod.default ?? chromeMod) : null;
  const executablePath = override ?? (await chrome.executablePath());
  if (override) return { executablePath, args: [], env: process.env };

  const pkg = join(root, "node_modules", "@sparticuz", "chromium");
  unpack(join(pkg, "bin", "al2023.tar.br"), cacheRoot, "lib/libnss3.so");
  unpack(join(pkg, "bin", "swiftshader.tar.br"), shaderDir, "vk_swiftshader_icd.json");

  return {
    executablePath,
    args: [
      "--no-sandbox",
      "--disable-dev-shm-usage",
      "--disable-setuid-sandbox",
      // SwiftShader gives this container a software GL/Vulkan adapter so the
      // WebGPU code paths can be exercised at all.
      "--use-angle=swiftshader",
      "--enable-unsafe-swiftshader",
      "--use-gl=angle",
      "--enable-unsafe-webgpu",
    ],
    env: {
      ...process.env,
      LD_LIBRARY_PATH: [libDir, shaderDir, process.env.LD_LIBRARY_PATH]
        .filter(Boolean)
        .join(":"),
      VK_ICD_FILENAMES: join(shaderDir, "vk_swiftshader_icd.json"),
    },
  };
}
