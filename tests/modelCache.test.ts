import { describe, it, expect, beforeEach, afterEach } from "vitest";
import {
  downloadModel,
  ensureModel,
  getCachedModel,
  isModelCached,
  clearModelCache,
  verifyModel,
  sha256Hex,
  ModelLoadError,
  CACHE_NAME,
  type ModelSpec,
} from "../src/core/modelCache";
import { checkDetectorContract, checkPoseContract } from "../src/core/compat";

// Minimal in-memory Cache Storage stand-in (keys are URLs, values are Responses).
function fakeCaches() {
  const stores = new Map<string, Map<string, Response>>();
  return {
    async open(name: string) {
      if (!stores.has(name)) stores.set(name, new Map());
      const m = stores.get(name)!;
      return {
        async match(key: string) {
          const r = m.get(key);
          return r ? r.clone() : undefined;
        },
        async put(key: string, r: Response) {
          m.set(key, r.clone());
        },
        async delete(key: string) {
          return m.delete(key);
        },
      };
    },
    async delete(name: string) {
      return stores.delete(name);
    },
    stores,
  };
}

const onnxBytes = (n = 64) => {
  const b = new Uint8Array(n);
  b[0] = 0x08; // ir_version field, as serialized by ONNX
  for (let i = 1; i < n; i++) b[i] = (i * 37) & 0xff;
  return b;
};

function streamResponse(
  bytes: Uint8Array,
  chunk = 16,
  headers: Record<string, string> = {},
) {
  const parts: Uint8Array[] = [];
  for (let i = 0; i < bytes.length; i += chunk)
    parts.push(bytes.slice(i, i + chunk));
  const body = new ReadableStream<Uint8Array>({
    start(c) {
      for (const p of parts) c.enqueue(p);
      c.close();
    },
  });
  return new Response(body, {
    status: 200,
    headers: { "content-length": String(bytes.length), ...headers },
  });
}

const spec = (over: Partial<ModelSpec> = {}): ModelSpec => ({
  id: "detector",
  label: "test",
  fileName: "test.onnx",
  url: "https://example.test/test.onnx",
  bytes: 64,
  ...over,
});

let fake: ReturnType<typeof fakeCaches>;
beforeEach(() => {
  fake = fakeCaches();
  (globalThis as any).caches = fake;
});
afterEach(() => {
  delete (globalThis as any).caches;
});

describe("model download progress and verification", () => {
  it("streams progress up to the exact byte count and caches the file", async () => {
    const bytes = onnxBytes(64);
    const progress: number[] = [];
    const file = await downloadModel(
      spec(),
      (received, total) => {
        expect(total).toBe(64);
        progress.push(received);
      },
      { fetchImpl: async () => streamResponse(bytes) },
    );
    expect(file.name).toBe("test.onnx");
    expect(file.size).toBe(64);
    expect(progress[0]).toBe(0);
    expect(progress.at(-1)).toBe(64);
    expect(progress.length).toBeGreaterThan(2);
    expect(await isModelCached(spec())).toBe(true);
  });

  it("rejects truncated downloads and does not cache them", async () => {
    await expect(
      downloadModel(spec(), () => {}, {
        fetchImpl: async () => streamResponse(onnxBytes(40)),
      }),
    ).rejects.toThrow(/expected 64 bytes, received 40/);
    expect(await isModelCached(spec())).toBe(false);
  });

  it("rejects non-ONNX payloads such as an HTML error page", async () => {
    const html = new TextEncoder().encode(
      "<html>quota exceeded</html>".padEnd(64, " "),
    );
    await expect(
      downloadModel(spec(), () => {}, {
        fetchImpl: async () => streamResponse(html),
      }),
    ).rejects.toThrow(/not an ONNX protobuf/);
  });

  it("enforces a SHA-256 when one is provided", async () => {
    const bytes = onnxBytes(64);
    const good = await sha256Hex(bytes);
    await expect(
      downloadModel(spec({ sha256: good }), () => {}, {
        fetchImpl: async () => streamResponse(bytes),
      }),
    ).resolves.toBeInstanceOf(File);
    await expect(
      downloadModel(spec({ sha256: "0".repeat(64) }), () => {}, {
        fetchImpl: async () => streamResponse(bytes),
      }),
    ).rejects.toThrow(/SHA-256 mismatch/);
  });

  it("reports HTTP and network failures as actionable errors", async () => {
    await expect(
      downloadModel(spec(), () => {}, {
        fetchImpl: async () => new Response("nope", { status: 404 }),
      }),
    ).rejects.toThrow(/HTTP 404/);
    await expect(
      downloadModel(spec(), () => {}, {
        fetchImpl: async () => {
          throw new TypeError("Failed to fetch");
        },
      }),
    ).rejects.toThrow(/import the file locally/i);
  });

  it("refuses to download when no URL is configured (local-file fallback)", async () => {
    await expect(
      downloadModel(spec({ url: "" }), () => {}),
    ).rejects.toBeInstanceOf(ModelLoadError);
  });

  it("can be aborted mid-stream", async () => {
    const controller = new AbortController();
    const body = new ReadableStream<Uint8Array>({
      pull(c) {
        controller.abort();
        c.error(new DOMException("aborted", "AbortError"));
      },
    });
    await expect(
      downloadModel(spec(), () => {}, {
        signal: controller.signal,
        fetchImpl: async () => new Response(body, { status: 200 }),
      }),
    ).rejects.toThrow();
  });
});

describe("cache behaviour", () => {
  it("serves the second load from cache without fetching", async () => {
    let fetches = 0;
    const bytes = onnxBytes(64);
    const fetchImpl = async () => {
      fetches++;
      return streamResponse(bytes);
    };
    const first = await ensureModel(spec(), () => {}, { fetchImpl });
    const second = await ensureModel(spec(), () => {}, { fetchImpl });
    expect(first.source).toBe("download");
    expect(second.source).toBe("cache");
    expect(fetches).toBe(1);
    expect(new Uint8Array(await second.file.arrayBuffer())).toEqual(bytes);
  });

  it("evicts a corrupted cache entry and falls back to downloading", async () => {
    const s = spec();
    await ensureModel(s, () => {}, {
      fetchImpl: async () => streamResponse(onnxBytes(64)),
    });
    const cache = await caches.open(CACHE_NAME);
    const key = [...fake.stores.get(CACHE_NAME)!.keys()][0];
    await cache.put(key, new Response(new Uint8Array(10)));
    expect(await getCachedModel(s)).toBeNull();
    expect(await isModelCached(s)).toBe(false);
  });

  it("clearModelCache removes stored models", async () => {
    await ensureModel(spec(), () => {}, {
      fetchImpl: async () => streamResponse(onnxBytes(64)),
    });
    await clearModelCache();
    expect(await isModelCached(spec())).toBe(false);
  });

  it("still works without Cache Storage (session-only memory)", async () => {
    delete (globalThis as any).caches;
    const r = await ensureModel(spec(), () => {}, {
      fetchImpl: async () => streamResponse(onnxBytes(64)),
    });
    expect(r.file.size).toBe(64);
    expect(await isModelCached(spec())).toBe(false);
  });
});

describe("verifyModel", () => {
  it("accepts an unknown size and any hash when none is configured", async () => {
    await expect(
      verifyModel(spec({ bytes: 0 }), onnxBytes(12)),
    ).resolves.toBeUndefined();
  });
});

describe("ONNX I/O compatibility contract", () => {
  const yoloxIn = [1, 3, 640, 640];
  it("accepts the YOLOX-L raw grid [1,8400,85]", () => {
    expect(() =>
      checkDetectorContract(yoloxIn, [1, 8400, 85], "yolox"),
    ).not.toThrow();
  });
  it("rejects YOLOX graphs with decoded or wrong channel layouts", () => {
    expect(() => checkDetectorContract(yoloxIn, [1, 8400, 6], "yolox")).toThrow(
      /85/,
    );
    expect(() =>
      checkDetectorContract(yoloxIn, [1, 84, 8400], "yolo"),
    ).not.toThrow();
    expect(() => checkDetectorContract(yoloxIn, [1, 8400, 85], "yolo")).toThrow(
      /84/,
    );
    expect(() => checkDetectorContract(yoloxIn, [1, 8400, 6], "yolo")).toThrow(
      /84/,
    );
  });
  it("rejects non-square detector input", () => {
    expect(() =>
      checkDetectorContract([1, 3, 640, 480], [1, 8400, 85], "yolox"),
    ).toThrow(/square/);
  });
  it("accepts DWPose SimCC 133-joint outputs at 288×384", () => {
    expect(() =>
      checkPoseContract(
        [1, 3, 384, 288],
        [
          [1, 133, 576],
          [1, 133, 768],
        ],
        "simcc",
        288,
        384,
      ),
    ).not.toThrow();
  });
  it("rejects COCO-17 body models and mismatched input sizes", () => {
    expect(() =>
      checkPoseContract(
        [1, 3, 384, 288],
        [
          [1, 17, 576],
          [1, 17, 768],
        ],
        "simcc",
        288,
        384,
      ),
    ).toThrow(/133/);
    expect(() =>
      checkPoseContract(
        [1, 3, 256, 192],
        [
          [1, 133, 384],
          [1, 133, 512],
        ],
        "simcc",
        288,
        384,
      ),
    ).toThrow(/settings say 384/);
  });
  it("accepts a 133-joint heatmap export and rejects a SimCC pair for heatmap mode", () => {
    expect(() =>
      checkPoseContract(
        [1, 3, 384, 288],
        [[1, 133, 96, 72]],
        "heatmap",
        288,
        384,
      ),
    ).not.toThrow();
    expect(() =>
      checkPoseContract(
        [1, 3, 384, 288],
        [
          [1, 133, 576],
          [1, 133, 768],
        ],
        "heatmap",
        288,
        384,
      ),
    ).toThrow(/heatmap/);
  });
});
