import { describe, it, expect, afterEach } from "vitest";
import { readFileSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  ortWebGpuVariant,
  ortWasmOnlyVariant,
  probeWebGpuAdapter,
  webGpuProviderLabel,
  readGpu,
  type GpuAdapterLike,
} from "../src/core/ortRuntime";

const dist = join(
  dirname(fileURLToPath(import.meta.url)),
  "..",
  "node_modules",
  "onnxruntime-web",
  "dist",
);

/** Resolves the ESM entry onnxruntime-web exposes for a subpath. */
function entryFile(subpath: "webgpu" | "wasm"): string {
  const pkg = JSON.parse(
    readFileSync(join(dist, "..", "package.json"), "utf8"),
  ) as {
    exports: Record<string, { import?: { default?: string } | string }>;
  };
  const entry = pkg.exports[`./${subpath}`]!.import;
  const file = typeof entry === "string" ? entry : entry!.default!;
  return join(dist, "..", file);
}

/** Glue file names, keyed by build id. */
const glueFor = {
  asyncify: "ort-wasm-simd-threaded.asyncify.mjs",
  plain: "ort-wasm-simd-threaded.mjs",
  jsep: "ort-wasm-simd-threaded.jsep.mjs",
  jspi: "ort-wasm-simd-threaded.jspi.mjs",
} as const;
const binaryFor = (id: keyof typeof glueFor) =>
  glueFor[id].replace(/\.mjs$/, ".wasm");

// The pairing is compiled into each entry point: it names the WASM module it
// will load when no path override is supplied. Pinning `env.wasm.wasmPaths` to
// a different variant is what made every WebGPU session fail with
// "webgpuInit is not a function" on every browser and device, so these tests
// re-derive the expected pairing from the installed package and fail loudly
// when a dependency upgrade changes it.
describe("ONNX Runtime WASM build pairing (the WebGPU regression)", () => {
  it.each([
    ["webgpu", ortWebGpuVariant],
    ["wasm", ortWasmOnlyVariant],
  ] as const)(
    "pins the WASM module that the %s entry point asks for",
    (subpath, variant) => {
      const entry = entryFile(subpath);
      const source = readFileSync(entry, "utf8");
      const referenced = new Set(
        [...source.matchAll(/ort-wasm-simd-threaded[a-z.-]*\.mjs/g)].map(
          (m) => m[0],
        ),
      );
      // Exactly one WASM glue per entry point; anything else means the entry
      // changed and the pin has to be revisited.
      expect([...referenced]).toHaveLength(1);
      expect([...referenced][0]).toBe(glueFor[variant.id]);
    },
  );

  it("only the pinned builds that expose the WebGPU bridge are used for WebGPU", () => {
    // webgpuInit is defined by the asyncify/jspi builds. If the WebGPU entry
    // ever expects a glue without it, WebGPU cannot work at all.
    const asyncify = readFileSync(
      join(dist, "ort-wasm-simd-threaded.asyncify.mjs"),
      "utf8",
    );
    const plain = readFileSync(join(dist, "ort-wasm-simd-threaded.mjs"), "utf8");
    expect(asyncify).toContain("webgpuInit");
    expect(plain).not.toContain("webgpuInit");
    expect(ortWebGpuVariant.id).toBe("asyncify");
    expect(ortWasmOnlyVariant.id).toBe("plain");
  });

  it("ships both pinned files, and the hook itself", () => {
    for (const variant of [ortWebGpuVariant, ortWasmOnlyVariant]) {
      // `?url` imports are rewritten by Vite; in the test runner they are the
      // asset paths, so assert on the shipped dist files instead.
      expect(variant.glue).toMatch(/ort-wasm-simd-threaded[.\w-]*\.mjs/);
      expect(variant.wasm).toMatch(/ort-wasm-simd-threaded[.\w-]*\.wasm/);
      expect(existsSync(join(dist, binaryFor(variant.id)))).toBe(true);
    }
  });
});

const adapter = (over: Partial<GpuAdapterLike> = {}): GpuAdapterLike => ({
  limits: {},
  features: new Set<string>(),
  requestDevice: async () => ({}),
  info: { vendor: "acme", architecture: "gpu-9000" },
  ...over,
});

const withGpu = (gpu: unknown) => {
  Object.defineProperty(globalThis, "navigator", {
    value: { gpu },
    configurable: true,
  });
};

afterEach(() => {
  Object.defineProperty(globalThis, "navigator", {
    value: undefined,
    configurable: true,
  });
});

describe("WebGPU adapter probe", () => {
  it("reports a missing API instead of assuming no GPU exists", async () => {
    withGpu(undefined);
    expect(readGpu()).toBeNull();
    const probe = await probeWebGpuAdapter();
    expect(probe.kind).toBe("no-api");
    expect(probe.reason).toContain("navigator.gpu");
  });

  it("returns the adapter and its label when one is handed out", async () => {
    withGpu({
      requestAdapter: async () => adapter(),
    });
    const probe = await probeWebGpuAdapter();
    expect(probe.kind).toBe("adapter");
    expect(probe.label).toBe("acme · gpu-9000");
    expect(probe.softwareOnly).toBe(false);
    expect(webGpuProviderLabel(probe)).toBe("WebGPU (acme · gpu-9000)");
  });

  it("retries without a power preference before giving up", async () => {
    const calls: unknown[] = [];
    withGpu({
      requestAdapter: async (options?: unknown) => {
        calls.push(options);
        return calls.length === 1 ? null : adapter();
      },
    });
    const probe = await probeWebGpuAdapter();
    expect(probe.kind).toBe("adapter");
    expect(calls[0]).toEqual({ powerPreference: "high-performance" });
    expect(calls[1]).toEqual({});
  });

  it("flags a software-only adapter and does not use it by default", async () => {
    withGpu({
      requestAdapter: async (options?: { forceFallbackAdapter?: boolean }) =>
        options?.forceFallbackAdapter
          ? adapter({ info: { vendor: "google", architecture: "swiftshader" } })
          : null,
    });
    const probe = await probeWebGpuAdapter();
    expect(probe.kind).toBe("no-adapter");
    expect(probe.reason).toContain("software fallback adapter");
    const optedIn = await probeWebGpuAdapter(true);
    expect(optedIn.kind).toBe("adapter");
    expect(optedIn.softwareOnly).toBe(true);
  });

  it("labels an emulated adapter as software when Chrome hands it out", async () => {
    withGpu({
      requestAdapter: async () =>
        adapter({ info: { vendor: "google", architecture: "swiftshader" } }),
    });
    const probe = await probeWebGpuAdapter();
    expect(probe.kind).toBe("adapter");
    expect(probe.softwareOnly).toBe(true);
    expect(probe.reason).toContain("software adapter");
  });

  it("carries the browser's own error text into the reason", async () => {
    withGpu({
      requestAdapter: async () => {
        throw new Error("Adapter request blocked by policy");
      },
    });
    const probe = await probeWebGpuAdapter();
    expect(probe.kind).toBe("no-adapter");
    expect(probe.reason).toContain("Adapter request blocked by policy");
  });
});
