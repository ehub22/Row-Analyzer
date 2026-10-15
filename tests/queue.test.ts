import { describe, it, expect } from "vitest";
import { SequentialPipeline } from "../src/core/queue";

const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

describe("SequentialPipeline", () => {
  it("handles items one at a time, in order", async () => {
    const seen: number[] = [];
    const pipeline = new SequentialPipeline<number>({
      handler: async (item) => {
        await tick();
        seen.push(item);
      },
    });
    for (const item of [1, 2, 3, 4, 5]) await pipeline.push(item);
    await pipeline.drain();
    expect(seen).toEqual([1, 2, 3, 4, 5]);
    expect(pipeline.pending).toBe(0);
  });

  it("never runs two handlers concurrently", async () => {
    let active = 0,
      maxActive = 0;
    const pipeline = new SequentialPipeline<number>({
      handler: async () => {
        active++;
        maxActive = Math.max(maxActive, active);
        await tick();
        active--;
      },
    });
    await Promise.all([1, 2, 3].map((i) => pipeline.push(i)));
    await pipeline.drain();
    expect(maxActive).toBe(1);
  });

  it("applies backpressure once capacity is reached", async () => {
    const gates: Array<() => void> = [];
    const pipeline = new SequentialPipeline<number>({
      capacity: 2,
      handler: () => new Promise<void>((resolve) => gates.push(resolve)),
    });
    // First item starts the handler and leaves room, so the producer proceeds.
    await pipeline.push(1);
    // Second item fills the pipeline: the producer is held before it can queue
    // a third, so at most `capacity` frames are ever outstanding.
    let freed = false;
    const second = pipeline.push(2).then(() => (freed = true));
    await tick();
    expect(freed).toBe(false);
    // A finished handler frees the producer while item 2 runs.
    gates.shift()!();
    await tick();
    expect(freed).toBe(true);
    gates.shift()!();
    await second;
    await pipeline.drain();
    expect(pipeline.pending).toBe(0);
  });

  it("latches the first handler error and stops consuming", async () => {
    const seen: number[] = [];
    const pipeline = new SequentialPipeline<number>({
      handler: async (item) => {
        seen.push(item);
        if (item === 2) throw new Error("stage failed");
      },
    });
    await pipeline.push(1);
    const second = pipeline.push(2);
    await second.catch(() => {});
    await tick();
    expect(pipeline.failed).toBe(true);
    expect(pipeline.errorValue).toBeInstanceOf(Error);
    // New work is refused rather than silently dropped.
    await expect(pipeline.push(3)).rejects.toThrow("stage failed");
    // drain() surfaces the same error for the whole run.
    await expect(pipeline.drain()).rejects.toThrow("stage failed");
    expect(seen).toEqual([1, 2]);
  });

  it("drops queued items with cleanup after a failure", async () => {
    const dropped: number[] = [];
    const pipeline = new SequentialPipeline<number>({
      capacity: 8,
      handler: async (item) => {
        if (item === 1) throw new Error("boom");
        await tick();
      },
      onDrop: (item) => dropped.push(item),
    });
    const pushes = [1, 2, 3, 4].map((i) => pipeline.push(i).catch(() => {}));
    await Promise.all(pushes);
    await pipeline.drain().catch(() => {});
    // Items that never ran must be released, not leaked (they hold frame
    // clones in the real pipeline).
    expect(dropped.sort()).toEqual([2, 3, 4]);
  });

  it("rejects a producer waiting for room when the pipeline fails", async () => {
    const gates: Array<() => void> = [];
    const pipeline = new SequentialPipeline<number>({
      capacity: 2,
      handler: () =>
        new Promise<void>((_, reject) =>
          gates.push(() => reject(new Error("boom"))),
        ),
    });
    await pipeline.push(1);
    const blocked = pipeline.push(2);
    const queued = pipeline.push(3);
    await tick();
    gates.shift()!();
    await expect(blocked).rejects.toThrow("boom");
    await expect(queued).rejects.toThrow("boom");
    await expect(pipeline.drain()).rejects.toThrow("boom");
    expect(pipeline.pending).toBe(0);
  });
});
