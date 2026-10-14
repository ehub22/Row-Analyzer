import manifest from "../../models/manifest.json";

export type ModelKind = "detector" | "pose";
export const modelManifest = manifest.models;
export interface ModelProgress {
  kind: ModelKind;
  phase: "cache" | "download" | "verify" | "ready";
  loaded: number;
  total: number;
}
const database = "row-analyzer-models-v1";
const limit = 512 * 1024 * 1024;
function db(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(database, 1);
    request.onupgradeneeded = () => request.result.createObjectStore("weights");
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
    request.onblocked = () =>
      reject(Error("Model cache is blocked by another tab."));
  });
}
async function transaction<T>(
  mode: IDBTransactionMode,
  action: (s: IDBObjectStore) => IDBRequest<T>,
): Promise<T> {
  const connection = await db();
  try {
    return await new Promise<T>((resolve, reject) => {
      const tx = connection.transaction("weights", mode);
      const request = action(tx.objectStore("weights"));
      tx.oncomplete = () => resolve(request.result);
      tx.onerror = tx.onabort = () =>
        reject(tx.error || request.error || Error("Cache transaction failed."));
    });
  } finally {
    connection.close();
  }
}
export async function clearModels() {
  await transaction("readwrite", (s) => s.clear());
}
export async function verifyModel(
  file: Blob,
  expected?: (typeof modelManifest)[number],
) {
  if (file.size < 8 || file.size > limit)
    throw Error("Expected a standalone ONNX file between 8 bytes and 512 MiB.");
  const bytes = await file.arrayBuffer();
  const head = new TextDecoder()
    .decode(bytes.slice(0, 100))
    .trim()
    .toLowerCase();
  if (
    head.startsWith("<") ||
    head.startsWith("version https://git-lfs") ||
    head.startsWith("{")
  )
    throw Error(
      "This is an HTML/JSON response or Git LFS pointer, not an ONNX model. Import the downloaded .onnx file.",
    );
  const hash = [...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))]
    .map((x) => x.toString(16).padStart(2, "0"))
    .join("");
  if (expected && (file.size !== expected.bytes || hash !== expected.sha256))
    throw Error(
      `${expected.name}: size or SHA-256 mismatch. The incomplete/incorrect file was not cached.`,
    );
  return hash;
}
export async function importModel(
  kind: ModelKind,
  file: File,
): Promise<{ file: File; warning?: string }> {
  // Unknown local exports remain supported; graph/inference validation happens before analysis.
  const hash = await verifyModel(file);
  const expected = modelManifest.find((m) => m.id === kind)!;
  if (hash !== expected.sha256)
    return {
      file,
      warning:
        "Custom local model: runtime compatibility will be checked before analysis. Only the pinned reference weights are persistently cached.",
    };
  try {
    await transaction("readwrite", (s) => s.put(file, expected.sha256));
    return { file };
  } catch {
    return {
      file,
      warning:
        "Model ready in memory. Browser storage is unavailable/full; reimport next time.",
    };
  }
}
export async function loadModel(
  kind: ModelKind,
  progress: (p: ModelProgress) => void,
  signal: AbortSignal,
): Promise<{ file: File; warning?: string }> {
  const spec = modelManifest.find((m) => m.id === kind)!;
  const emit = (phase: ModelProgress["phase"], loaded = 0) =>
    progress({ kind, phase, loaded, total: spec.bytes });
  let cacheWarning: string | undefined;
  emit("cache");
  try {
    const cached = await transaction<Blob | undefined>("readonly", (s) =>
      s.get(spec.sha256),
    );
    signal.throwIfAborted();
    if (cached) {
      try {
        emit("verify", cached.size);
        await verifyModel(cached, spec);
        signal.throwIfAborted();
        emit("ready", cached.size);
        return { file: new File([cached], spec.name) };
      } catch (e) {
        signal.throwIfAborted();
        await transaction("readwrite", (s) => s.delete(spec.sha256));
        cacheWarning = `Corrupt cached ${spec.name} removed; downloading a fresh copy.`;
      }
    }
  } catch {
    signal.throwIfAborted();
    cacheWarning =
      "Browser model cache is unavailable. Models will work in memory only.";
  }
  emit("download");
  let response: Response;
  try {
    response = await fetch(
      new URL(
        `models/${spec.name}`,
        new URL(import.meta.env.BASE_URL, location.href),
      ),
      { signal },
    );
  } catch (e) {
    signal.throwIfAborted();
    throw Error(
      `Could not download ${spec.name}. Check your connection or import the local ONNX file. ${String(e)}`,
    );
  }
  if (!response.ok || !response.body)
    throw Error(
      `${spec.name}: download failed (HTTP ${response.status}). Use the local-file fallback.`,
    );
  const declared = Number(response.headers.get("content-length"));
  if (declared && declared !== spec.bytes)
    throw Error(
      `${spec.name}: server returned an unexpected file size. Use local import.`,
    );
  const reader = response.body.getReader();
  const chunks: ArrayBuffer[] = [];
  let loaded = 0;
  try {
    while (true) {
      signal.throwIfAborted();
      const { done, value } = await reader.read();
      if (done) break;
      loaded += value.byteLength;
      if (loaded > spec.bytes)
        throw Error(`${spec.name}: download exceeded the pinned size.`);
      chunks.push(value.slice().buffer);
      emit("download", loaded);
    }
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
  const file = new File(chunks, spec.name, {
    type: "application/octet-stream",
  });
  emit("verify", loaded);
  await verifyModel(file, spec);
  signal.throwIfAborted();
  try {
    await transaction("readwrite", (s) => s.put(file, spec.sha256));
  } catch {
    cacheWarning =
      "Model ready in memory, but storage quota/private browsing prevented caching. Reimport or download next time.";
  }
  signal.throwIfAborted();
  emit("ready", loaded);
  return { file, warning: cacheWarning };
}
