// Bounded, in-order pipeline stage.
//
// The analysis of a frame is two stages that use different hardware:
// detection runs on the coordinator's own session, pose passes run on the
// worker pool (or on the GPU session). Awaiting them one after another leaves
// half the machine idle — the pool sits idle during every detector run, and
// the coordinator sits idle while the pool works — so frames are handed to a
// consumer that keeps exactly one frame's pose work in flight while the next
// frame's detection starts.
//
// Order is preserved (a FIFO of one consumer), which the seat tracker requires:
// it matches detections against the previous frame, so frames have to reach it
// in decode order. Backpressure is the producer's `await push(item)`: it
// resolves immediately while there is room and only blocks once `capacity`
// frames are queued or in flight, so a slow pose stage throttles the decoder
// instead of buffering the whole clip in memory.
//
// Failure semantics: the first handler error is latched. It stops the consumer,
// drops (and cleans up) everything still queued, rejects pushes, and is
// rethrown by drain() so the caller sees one error for the run.

export interface SequentialPipelineOptions<T> {
  handler: (item: T) => Promise<void>;
  /** Frames allowed to be queued or in flight before push() blocks. */
  capacity?: number;
  /** Releases resources of items dropped after a failure. */
  onDrop?: (item: T) => void;
  /** Test hook: resolve the current task queue between handler calls. */
  yieldBetween?: () => Promise<void>;
}

export class SequentialPipeline<T> {
  private queue: T[] = [];
  private running = false;
  private error: unknown = null;
  private waiters: Array<{
    resolve: () => void;
    reject: (e: unknown) => void;
  }> = [];
  private readonly handler: (item: T) => Promise<void>;
  private readonly capacity: number;
  private readonly onDrop?: (item: T) => void;
  private readonly yieldBetween?: () => Promise<void>;

  constructor(options: SequentialPipelineOptions<T>) {
    this.handler = options.handler;
    this.capacity = Math.max(1, options.capacity ?? 3);
    this.onDrop = options.onDrop;
    this.yieldBetween = options.yieldBetween;
  }

  /** True once a handler error has stopped the pipeline. */
  get failed(): boolean {
    return this.error !== null;
  }

  get errorValue(): unknown {
    return this.error;
  }

  /** Items queued or in flight. */
  get pending(): number {
    return this.queue.length + (this.running ? 1 : 0);
  }

  /**
   * Queues one item. Resolves at once while there is room; otherwise resolves
   * when the consumer has caught up. Rejects once the pipeline has failed.
   */
  push(item: T): Promise<void> {
    if (this.error) return Promise.reject(this.error);
    this.queue.push(item);
    void this.pump();
    if (this.error) return Promise.reject(this.error);
    if (this.pending < this.capacity) return Promise.resolve();
    return new Promise((resolve, reject) =>
      this.waiters.push({ resolve, reject }),
    );
  }

  /** Waits for everything queued to finish, then rethrows the first error. */
  async drain(): Promise<void> {
    while (this.running || this.queue.length) {
      await new Promise((resolve) => setTimeout(resolve, 1));
    }
    if (this.error) throw this.error;
  }

  private async pump(): Promise<void> {
    if (this.running) return;
    this.running = true;
    try {
      while (this.queue.length) {
        const item = this.queue.shift()!;
        // Taking the item out of the queue is what frees a slot: the producer
        // is released as soon as fewer than `capacity` items are queued or in
        // flight, not only when a handler finishes.
        this.release();
        try {
          await this.handler(item);
        } catch (e) {
          this.error ??= e;
        }
        if (this.error) {
          this.fail();
          return;
        }
        if (this.yieldBetween) await this.yieldBetween();
      }
    } finally {
      this.running = false;
    }
  }

  private release(): void {
    if (this.pending < this.capacity) {
      for (const waiter of this.waiters.splice(0)) waiter.resolve();
    }
  }

  private fail(): void {
    for (const item of this.queue.splice(0)) this.onDrop?.(item);
    // A producer waiting for room must learn about the failure here rather than
    // on its next push, so the decode loop can stop promptly.
    for (const waiter of this.waiters.splice(0)) waiter.reject(this.error);
  }
}
