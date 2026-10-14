import "fake-indexeddb/auto";
import { beforeEach, afterEach, describe, expect, it, vi } from "vitest";
vi.mock("../models/manifest.json", () => ({
  default: {
    models: [
      {
        id: "detector",
        name: "test.onnx",
        bytes: 8,
        sha256:
          "ef797c8118f02dfb649607dd5d3f8c7623048c9c063d532cc95c5ed7a898a64f",
      },
    ],
  },
}));
import { clearModels, loadModel, importModel } from "../src/core/models";
beforeEach(async () => {
  await clearModels();
  vi.stubEnv("BASE_URL", "./");
  vi.stubGlobal("location", { href: "https://example.test/Row-Analyzer/" });
});
afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});
describe("atomic persistent model caching", () => {
  it("streams progress, verifies, then restores offline without fetching", async () => {
    const fetch = vi.fn(
      async (_url: URL) =>
        new Response("12345678", { headers: { "content-length": "8" } }),
    );
    vi.stubGlobal("fetch", fetch);
    const phases: string[] = [];
    const first = await loadModel(
      "detector",
      (p) => phases.push(p.phase),
      new AbortController().signal,
    );
    expect(first.file.size).toBe(8);
    expect(phases).toContain("download");
    expect(phases.at(-1)).toBe("ready");
    const url = String(fetch.mock.calls[0]?.[0]);
    expect(url).toBe("https://example.test/Row-Analyzer/models/test.onnx");
    fetch.mockImplementation(async () => {
      throw Error("offline");
    });
    const second = await loadModel(
      "detector",
      () => {},
      new AbortController().signal,
    );
    expect(await second.file.text()).toBe("12345678");
    expect(fetch).toHaveBeenCalledTimes(1);
  });
  it("does not cache HTTP errors, corrupt bytes or partial downloads; retry succeeds", async () => {
    const fetch = vi
      .fn()
      .mockResolvedValueOnce(new Response("no", { status: 403 }))
      .mockResolvedValueOnce(new Response("badbytes"))
      .mockResolvedValueOnce(new Response("short"))
      .mockResolvedValueOnce(new Response("12345678"));
    vi.stubGlobal("fetch", fetch);
    for (let i = 0; i < 3; i++)
      await expect(
        loadModel("detector", () => {}, new AbortController().signal),
      ).rejects.toThrow();
    await expect(
      loadModel("detector", () => {}, new AbortController().signal),
    ).resolves.toHaveProperty("file");
    expect(fetch).toHaveBeenCalledTimes(4);
  });
  it("cancellation never leaves a usable partial cache", async () => {
    const controller = new AbortController();
    vi.stubGlobal(
      "fetch",
      vi.fn(
        async () =>
          new Response(
            new ReadableStream({
              start(c) {
                c.enqueue(new TextEncoder().encode("1234"));
                controller.abort();
                c.close();
              },
            }),
          ),
      ),
    );
    await expect(
      loadModel("detector", () => {}, controller.signal),
    ).rejects.toThrow();
    const fetch = vi.fn(async () => new Response("12345678"));
    vi.stubGlobal("fetch", fetch);
    await loadModel("detector", () => {}, new AbortController().signal);
    expect(fetch).toHaveBeenCalledTimes(1);
  });
  it("persists only reference local files; clear removes cached weights", async () => {
    await importModel("detector", new File(["12345678"], "test.onnx"));
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw Error("offline");
      }),
    );
    await expect(
      loadModel("detector", () => {}, new AbortController().signal),
    ).resolves.toHaveProperty("file");
    await clearModels();
    await expect(
      loadModel("detector", () => {}, new AbortController().signal),
    ).rejects.toThrow("local");
    expect(
      (await importModel("detector", new File(["custom00"], "custom.onnx")))
        .warning,
    ).toContain("Custom");
  });
  it("continues in memory and warns when IndexedDB is unavailable", async () => {
    vi.stubGlobal("indexedDB", {
      open: () => {
        throw Error("quota");
      },
    });
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response("12345678")),
    );
    const result = await loadModel(
      "detector",
      () => {},
      new AbortController().signal,
    );
    expect(result.file.size).toBe(8);
    expect(result.warning).toContain("storage");
  });
});
