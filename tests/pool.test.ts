import { describe, it, expect } from "vitest";
import { PoseWorkerPool, type PoseJob } from "../src/core/pool";
import type { Point } from "../src/core/types";

interface Sent {
  message: Record<string, unknown>;
  transfer?: Transferable[];
}

class FakeEndpoint {
  sent: Sent[] = [];
  send(message: unknown, transfer?: Transferable[]) {
    this.sent.push({ message: message as Record<string, unknown>, transfer });
  }
  get jobs() {
    return this.sent.filter((s) => s.message.type === "pose");
  }
}

const job = (seed = 0): PoseJob => ({
  data: new Float32Array([seed]),
  dims: [1, 3, 2, 2],
  crop: { x: 0, y: 0, width: 10, height: 10, confidence: 1 },
  flip: false,
  format: "simcc",
});

const points = (x: number): Point[] => [
  { x, y: 0, confidence: 1, origin: "observed" },
];

function readyPool(n = 2) {
  const endpoints = Array.from({ length: n }, () => new FakeEndpoint());
  const pool = new PoseWorkerPool(endpoints);
  pool.init({ poseBytes: new ArrayBuffer(4) });
  endpoints.forEach((_, i) => pool.handleMessage(i, { type: "ready" }));
  return { pool, endpoints };
}

describe("PoseWorkerPool", () => {
  it("sends init to every endpoint and reports readiness", async () => {
    const endpoints = [new FakeEndpoint(), new FakeEndpoint()];
    const pool = new PoseWorkerPool(endpoints);
    const ready = pool.ready();
    pool.init({ wasmThreads: 1 });
    expect(endpoints.every((e) => e.sent[0].message.type === "init")).toBe(true);
    pool.handleMessage(1, { type: "ready" });
    await expect(ready).resolves.toBeUndefined();
    expect(pool.aliveCount).toBe(1);
  });

  it("rejects readiness when every worker fails init", async () => {
    const pool = new PoseWorkerPool([new FakeEndpoint()]);
    const ready = pool.ready();
    pool.init({});
    pool.handleMessage(0, { type: "init-error", error: "bad model contract" });
    await expect(ready).rejects.toThrow("bad model contract");
    await expect(pool.dispatch(job())).rejects.toThrow("No pose worker");
  });

  it("balances jobs across endpoints and resolves out of order", async () => {
    const { pool, endpoints } = readyPool(2);
    const a = pool.dispatch(job(1));
    const b = pool.dispatch(job(2));
    const c = pool.dispatch(job(3));
    // Least-busy routing: 2 jobs on one endpoint, 1 on the other.
    expect(endpoints.map((e) => e.jobs.length).sort()).toEqual([1, 2]);
    const busy = endpoints.findIndex((e) => e.jobs.length === 2);
    const firstId = endpoints[busy].jobs[0].message.id as number;
    const secondId = endpoints[busy].jobs[1].message.id as number;
    // Answer the later job first; promises must still match by id.
    pool.handleMessage(busy, { type: "pose-result", id: secondId, points: points(2) });
    pool.handleMessage(busy, { type: "pose-result", id: firstId, points: points(1) });
    const other = endpoints[1 - busy];
    pool.handleMessage(1 - busy, {
      type: "pose-result",
      id: other.jobs[0].message.id,
      points: points(3),
    });
    await expect(Promise.all([a, b, c])).resolves.toHaveLength(3);
  });

  it("transfers the tensor buffer instead of cloning it", async () => {
    const { pool, endpoints } = readyPool(1);
    const p = pool.dispatch(job());
    const sent = endpoints[0].jobs[0];
    expect(sent.transfer).toHaveLength(1);
    expect(sent.transfer?.[0]).toBeInstanceOf(ArrayBuffer);
    pool.handleMessage(0, {
      type: "pose-result",
      id: sent.message.id,
      points: points(0),
    });
    await p;
  });

  it("a pose-error rejects only that job, not the pool", async () => {
    const { pool, endpoints } = readyPool(1);
    const bad = pool.dispatch(job());
    pool.handleMessage(0, {
      type: "pose-error",
      id: endpoints[0].jobs[0].message.id,
      error: "unsupported operator",
    });
    await expect(bad).rejects.toThrow("unsupported operator");
    const good = pool.dispatch(job());
    pool.handleMessage(0, {
      type: "pose-result",
      id: endpoints[0].jobs[1].message.id,
      points: points(7),
    });
    await expect(good).resolves.toEqual(points(7));
  });

  it("a dead endpoint rejects its in-flight jobs and is never reused", async () => {
    const { pool, endpoints } = readyPool(2);
    const a = pool.dispatch(job(1)); // least-busy routing: endpoint 0
    const b = pool.dispatch(job(2)); // then endpoint 1
    pool.handleError(0, new Error("worker crashed"));
    await expect(a).rejects.toThrow("worker crashed");
    expect(pool.deadCount).toBe(1);
    expect(pool.settledCount).toBe(2);
    // The surviving endpoint still completes its job.
    pool.handleMessage(1, {
      type: "pose-result",
      id: endpoints[1].jobs[0].message.id,
      points: points(5),
    });
    await expect(b).resolves.toEqual(points(5));
    // New work goes only to the survivor.
    const before = endpoints[0].jobs.length;
    const c = pool.dispatch(job(3));
    expect(endpoints[0].jobs.length).toBe(before);
    pool.handleMessage(1, {
      type: "pose-result",
      id: endpoints[1].jobs.at(-1)!.message.id,
      points: points(6),
    });
    await expect(c).resolves.toEqual(points(6));
  });

  it("dispose rejects pending jobs", async () => {
    const { pool } = readyPool(1);
    const p = pool.dispatch(job());
    pool.dispose();
    await expect(p).rejects.toThrow("disposed");
    await expect(pool.dispatch(job())).rejects.toThrow("disposed");
  });
});
