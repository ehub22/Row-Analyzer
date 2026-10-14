// ONNX Runtime Web runtime wiring: WASM build selection and the WebGPU probe.
//
// WHY THIS FILE EXISTS
//
// onnxruntime-web ships one JavaScript entry point per execution-provider set,
// and **each entry expects exactly one of the prebuilt WASM modules**. The
// pairing is compiled into the entry (it is not chosen at runtime), so mixing
// them silently breaks execution providers:
//
//   entry                       pinned WASM module                 needs
//   --------------------------  --------------------------------  -----------
//   onnxruntime-web/webgpu      ort-wasm-simd-threaded.asyncify   webgpuInit
//   onnxruntime-web (all EPs)   ort-wasm-simd-threaded.jsep       jsepInit
//   onnxruntime-web/wasm        ort-wasm-simd-threaded            —
//   onnxruntime-web/jspi        ort-wasm-simd-threaded.jspi       jspi
//
// The asyncify/jspi modules are the ones that expose the native WebGPU bridge
// (`webgpuInit`); the jsep and plain modules do not define it at all. Pointing
// `env.wasm.wasmPaths` at the jsep module while the WebGPU entry asks for the
// asyncify one produced, on every browser and device:
//
//   WebGPU failed for <model> (no available backend found.
//   ERR: [webgpu] TypeError: ...webgpuInit is not a function); using WASM.
//
// which is a static file mismatch, not a hardware limitation. The bundled
// model files are referenced with `?url` so the app serves exactly the pair it
// pinned (offline, same origin, no CDN) instead of relying on the bundler to
// guess. `tests/ortRuntime.test.ts` re-derives the table above from the
// installed package and fails if a dependency upgrade changes the pairing.
//
// WebGPU itself is also probed here rather than assumed from `navigator.gpu`:
// exposing the API and being able to hand out an adapter are different things
// (Linux Chrome, Firefox, Safari, headless, blocklisted drivers, remote
// desktops), and users need to be told which one failed.

import asyncifyGlueUrl from "../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.mjs?url";
import asyncifyWasmUrl from "../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.wasm?url";
import plainGlueUrl from "../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs?url";
import plainWasmUrl from "../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm?url";

export interface OrtWasmVariant {
  /** Which prebuilt WASM module this is. */
  id: "asyncify" | "plain" | "jsep" | "jspi";
  /** JS glue module (loads the binary, owns the emscripten runtime). */
  glue: string;
  /** WebAssembly binary consumed by that glue. */
  wasm: string;
  /** Entry points in onnxruntime-web that require this exact variant. */
  forEntry: string[];
}

/** Pair with `import ... from "onnxruntime-web/webgpu"`. */
export const ortWebGpuVariant: OrtWasmVariant = {
  id: "asyncify",
  glue: asyncifyGlueUrl,
  wasm: asyncifyWasmUrl,
  forEntry: ["onnxruntime-web/webgpu"],
};

/** Pair with `import ... from "onnxruntime-web/wasm"` (no GPU code at all). */
export const ortWasmOnlyVariant: OrtWasmVariant = {
  id: "plain",
  glue: plainGlueUrl,
  wasm: plainWasmUrl,
  forEntry: ["onnxruntime-web/wasm"],
};

/** Minimal shape of the ORT env we touch; avoids a second ORT import. */
export interface OrtEnvLike {
  wasm: {
    numThreads?: number;
    wasmPaths?: unknown;
  };
  webgpu: {
    adapter?: unknown;
    device?: unknown;
  };
}

/**
 * Points ONNX Runtime at the WASM build that matches the imported entry point
 * and fixes the thread count. Must run before the first session is created in
 * this JavaScript context (ORT initialises the WASM runtime once).
 */
export function configureOrtRuntime(
  env: OrtEnvLike,
  variant: OrtWasmVariant,
  numThreads: number,
): void {
  env.wasm.numThreads = Math.max(1, Math.floor(numThreads) || 1);
  env.wasm.wasmPaths = { wasm: variant.wasm, mjs: variant.glue };
}

/** The WebGPU objects we use, typed locally so no @webgpu/types is required. */
export interface GpuAdapterLike {
  limits?: unknown;
  features?: unknown;
  requestDevice?: () => Promise<unknown>;
  info?: {
    vendor?: string;
    architecture?: string;
    device?: string;
    description?: string;
  };
}
export interface GpuLike {
  requestAdapter(options?: {
    powerPreference?: "low-power" | "high-performance";
    forceFallbackAdapter?: boolean;
  }): Promise<GpuAdapterLike | null>;
}

/** Reads navigator.gpu without throwing in non-browser contexts. */
export function readGpu(): GpuLike | null {
  try {
    const gpu = (globalThis as { navigator?: { gpu?: GpuLike } }).navigator?.gpu;
    return gpu && typeof gpu.requestAdapter === "function" ? gpu : null;
  } catch {
    return null;
  }
}

export type GpuProbeKind = "adapter" | "no-api" | "no-adapter" | "error";

export interface GpuProbe {
  kind: GpuProbeKind;
  /** Usable adapter, when one was handed out. */
  adapter: GpuAdapterLike | null;
  /** Short human label, e.g. "google · swiftshader". */
  label: string;
  /** Full explanation for the session notes. */
  reason: string;
  /** True when only a software (SwiftShader-style) adapter exists. */
  softwareOnly: boolean;
}

const describeAdapter = (adapter: GpuAdapterLike): string => {
  const info = adapter.info;
  const parts = [info?.vendor, info?.architecture || info?.description || info?.device]
    .map((p) => (typeof p === "string" ? p.trim() : ""))
    .filter((p) => p.length > 0);
  return parts.length ? parts.join(" · ") : "adapter without identifying info";
};

/**
 * Asks WebGPU for an adapter and reports precisely what happened. A
 * `requestAdapter()` that returns null is the common failure (Chrome on Linux
 * without its WebGPU flag, Firefox/Safari builds, blocklisted or virtualised
 * GPUs, headless), and it is not the same problem as a missing API — the
 * analysis notes have to say which one it was.
 *
 * Software adapters are reported but not used unless `allowSoftware` is set:
 * an emulated GPU is usually slower than the parallel WASM pool.
 */
export async function probeWebGpuAdapter(allowSoftware = false): Promise<GpuProbe> {
  const gpu = readGpu();
  if (!gpu)
    return {
      kind: "no-api",
      adapter: null,
      label: "",
      softwareOnly: false,
      reason:
        "This browser exposes no navigator.gpu, so WebGPU is unavailable here (some browsers disable it outside a secure context, in private windows, or without a page reload after a flag change).",
    };

  let failure = "";
  for (const options of [
    { powerPreference: "high-performance" as const },
    {},
  ]) {
    try {
      const adapter = await gpu.requestAdapter(options);
      if (adapter) {
        const softwareOnly =
          typeof adapter.info?.architecture === "string" &&
          /swiftshader|software|llvmpipe|cpu/i.test(adapter.info.architecture);
        return {
          kind: "adapter",
          adapter,
          label: describeAdapter(adapter),
          softwareOnly,
          reason: softwareOnly
            ? `WebGPU adapter found (${describeAdapter(adapter)}) — a software adapter; GPU inference is optional and may be slower than the WASM pool.`
            : `WebGPU adapter found (${describeAdapter(adapter)}).`,
        };
      }
    } catch (e) {
      failure = e instanceof Error ? e.message : String(e);
    }
  }

  // Distinguish "no hardware GPU at all" from "software fallback exists", so
  // the message can be actionable instead of a generic failure.
  let software: GpuAdapterLike | null = null;
  try {
    software = await gpu.requestAdapter({ forceFallbackAdapter: true });
  } catch {
    /* ignore: probing only */
  }
  const tail = failure ? ` (last error: ${failure.slice(0, 160)})` : "";
  if (software && allowSoftware)
    return {
      kind: "adapter",
      adapter: software,
      label: `${describeAdapter(software)} · software`,
      softwareOnly: true,
      reason: `No hardware adapter; using the software WebGPU adapter (${describeAdapter(software)}). Expect WASM-like speed.`,
    };
  return {
    kind: "no-adapter",
    adapter: null,
    label: "",
    softwareOnly: false,
    reason: software
      ? `WebGPU is present but handed out no hardware adapter${tail}; only a software fallback adapter is available, which this app does not use (the parallel WASM pool is faster). Check chrome://gpu, the OS/driver GPU acceleration setting, and the browser's WebGPU support.`
      : `WebGPU is present but handed out no adapter at all${tail}. This is usually the browser or driver: check chrome://gpu, enable hardware acceleration, update the GPU driver, or use a browser build with WebGPU support (some platforms require an explicit WebGPU/Vulkan flag).`,
  };
}

/** One-line provider label for the session, e.g. "WebGPU (google · swiftshader)". */
export function webGpuProviderLabel(probe: GpuProbe): string {
  return probe.kind === "adapter" && probe.label
    ? `WebGPU (${probe.label})`
    : "WebGPU";
}
