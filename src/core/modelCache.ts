// Automatic model download + persistent browser cache (Cache Storage).
//
// Weights are NEVER committed to Git or bundled into the Pages build. Each model
// has a remote URL supplied at build time (VITE_YOLOX_URL / VITE_DWPOSE_URL).
// Resolution order for a model file:
//   1. a local file chosen by the user in the Models dialog (always wins),
//   2. the Cache Storage copy from an earlier download (size/hash re-checked),
//   3. a fresh streamed download from the configured URL, then cached,
//   4. otherwise an error that tells the user to import the file locally.

export type ModelId = "detector" | "pose";

export interface ModelSpec {
  id: ModelId;
  label: string;
  /** File name as stored in Drive / expected by the compatibility checks. */
  fileName: string;
  /** Remote source. Empty string means "no download configured": local file only. */
  url: string;
  /** Exact expected byte length (from the source file metadata). 0 = unknown. */
  bytes: number;
  /** Optional lowercase hex SHA-256 of the file. Checked when present. */
  sha256?: string;
}

export const CACHE_NAME = "row-analyzer-models-v1";
const CACHE_PREFIX = "./__models__/";

/** Build-time override for the expected size; "0" disables the exact-size check. */
function sizeOverride(raw: string | undefined, fallback: number) {
  if (raw === undefined || raw === "") return fallback;
  const n = Number(raw);
  return Number.isFinite(n) && n >= 0 ? Math.floor(n) : fallback;
}

export const modelSpecs: Record<ModelId, ModelSpec> = {
  detector: {
    id: "detector",
    label: "YOLOX-L person detector",
    fileName: "yolox_l.onnx",
    url: import.meta.env.VITE_YOLOX_URL ?? "",
    bytes: sizeOverride(import.meta.env.VITE_YOLOX_BYTES, 216746733),
  },
  pose: {
    id: "pose",
    label: "DWPose whole-body (dw-ll_ucoco_384)",
    fileName: "dw-ll_ucoco_384.onnx",
    url: import.meta.env.VITE_DWPOSE_URL ?? "",
    bytes: sizeOverride(import.meta.env.VITE_DWPOSE_BYTES, 134399116),
  },
};

export type ProgressFn = (received: number, total: number) => void;

export class ModelLoadError extends Error {}

function cacheKey(spec: ModelSpec) {
  return new URL(
    CACHE_PREFIX + spec.fileName,
    globalThis.location?.href ?? "https://local.invalid/",
  ).href;
}

/** ONNX models are protobuf ModelProtos; field 1 (ir_version) is serialized first. */
export function looksLikeOnnx(bytes: Uint8Array) {
  return bytes.length > 8 && bytes[0] === 0x08;
}

export async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", bytes as BufferSource);
  return Array.from(new Uint8Array(digest), (b) =>
    b.toString(16).padStart(2, "0"),
  ).join("");
}

/** Validates a complete model buffer against the spec. Throws ModelLoadError. */
export async function verifyModel(spec: ModelSpec, bytes: Uint8Array) {
  if (spec.bytes && bytes.byteLength !== spec.bytes)
    throw new ModelLoadError(
      `${spec.fileName}: expected ${spec.bytes} bytes, received ${bytes.byteLength}. The file is truncated or is a different export.`,
    );
  if (!looksLikeOnnx(bytes))
    throw new ModelLoadError(`${spec.fileName} is not an ONNX protobuf file.`);
  if (spec.sha256) {
    const actual = await sha256Hex(bytes);
    if (actual !== spec.sha256.toLowerCase())
      throw new ModelLoadError(
        `${spec.fileName}: SHA-256 mismatch (${actual.slice(0, 12)}… ≠ ${spec.sha256.slice(0, 12)}…).`,
      );
  }
}

async function openCache(): Promise<Cache | null> {
  try {
    return globalThis.caches ? await caches.open(CACHE_NAME) : null;
  } catch {
    return null; // e.g. private browsing or insecure context: download per session
  }
}

/** Returns the cached file if present and still valid; invalid entries are evicted. */
export async function getCachedModel(spec: ModelSpec): Promise<File | null> {
  const cache = await openCache();
  if (!cache) return null;
  const hit = await cache.match(cacheKey(spec));
  if (!hit) return null;
  const bytes = new Uint8Array(await hit.arrayBuffer());
  try {
    await verifyModel(spec, bytes);
  } catch {
    await cache.delete(cacheKey(spec));
    return null;
  }
  return new File([bytes], spec.fileName, { type: "application/octet-stream" });
}

/** Streams a download, reporting progress, then verifies and stores it in the cache. */
export async function downloadModel(
  spec: ModelSpec,
  onProgress: ProgressFn,
  options: { fetchImpl?: typeof fetch; signal?: AbortSignal } = {},
): Promise<File> {
  if (!spec.url)
    throw new ModelLoadError(
      `No download URL is configured for ${spec.fileName}. Import the file locally.`,
    );
  const fetchImpl = options.fetchImpl ?? fetch;
  let response: Response;
  try {
    response = await fetchImpl(spec.url, {
      signal: options.signal,
      cache: "no-store",
    });
  } catch (e) {
    throw new ModelLoadError(
      `Could not download ${spec.fileName} (${e instanceof Error ? e.message : String(e)}). Check the connection or import the file locally.`,
    );
  }
  if (!response.ok || !response.body)
    throw new ModelLoadError(
      `Download of ${spec.fileName} failed with HTTP ${response.status}.`,
    );

  const declared = Number(response.headers.get("content-length")) || 0;
  const total = spec.bytes || declared;
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let received = 0;
  onProgress(0, total);
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.byteLength;
    onProgress(received, total);
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const c of chunks) {
    bytes.set(c, offset);
    offset += c.byteLength;
  }
  await verifyModel(spec, bytes);

  const cache = await openCache();
  if (cache) {
    try {
      await cache.put(
        cacheKey(spec),
        new Response(bytes, {
          headers: {
            "content-type": "application/octet-stream",
            "content-length": String(bytes.byteLength),
            "x-model-source": spec.url,
          },
        }),
      );
    } catch {
      // Quota or storage error: the model still works for this session.
    }
  }
  return new File([bytes], spec.fileName, { type: "application/octet-stream" });
}

/** Cache first, then download. Never silently substitutes a different model. */
export async function ensureModel(
  spec: ModelSpec,
  onProgress: ProgressFn,
  options: { fetchImpl?: typeof fetch; signal?: AbortSignal } = {},
): Promise<{ file: File; source: "cache" | "download" }> {
  const cached = await getCachedModel(spec);
  if (cached) return { file: cached, source: "cache" };
  return {
    file: await downloadModel(spec, onProgress, options),
    source: "download",
  };
}

export async function isModelCached(spec: ModelSpec): Promise<boolean> {
  const cache = await openCache();
  return cache ? Boolean(await cache.match(cacheKey(spec))) : false;
}

export async function clearModelCache(): Promise<void> {
  if (globalThis.caches) await caches.delete(CACHE_NAME);
}
