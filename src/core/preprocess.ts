// Frame rasterisation: decoded VideoFrames → the NCHW float tensors the ONNX
// models consume. This is the CPU-side hot path of the whole pipeline, and two
// properties of it are deliberate:
//
// 1. **One bitmap per frame.** Drawing a VideoFrame into a canvas costs a
//    readback/format conversion that does *not* get cheaper when the crop is
//    small (profiled in the reference container: ~20 ms for a 288×384 pose
//    crop, versus 0.4 ms for the normalization loop that follows it and 0.08 ms
//    to convert the frame to an ImageBitmap). The bitmap is created once per
//    frame and every crop/TTA pass — local or in a pose worker — draws from it.
//    `scripts/pose-perf-check.mjs` asserts that both paths rasterise
//    *bit-identical* pixels, so this changes speed only, never the numbers the
//    model sees.
// 2. **Reuse the canvas and the tensor buffer.** An eight with four TTA passes
//    per crop otherwise allocates ~40 MB of Float32Arrays per frame, which is
//    pure GC pressure. A Rasterizer keeps one canvas and one buffer per
//    pipeline stage; callers must therefore finish with the returned array
//    before rasterising again on the same instance (workers serialise their
//    jobs, the local fallback path awaits each inference).

import type { Box } from "./types";

/** ImageNet normalization DWPose/RTMW checkpoints expect, RGB order. */
export const POSE_MEAN = [123.675, 116.28, 103.53] as const;
export const POSE_STD = [58.395, 57.12, 57.375] as const;

/** `yolo` = ultralytics RGB 0..1, `yolox` = raw BGR 0..255, `pose` = normalized. */
export type RasterMode = "yolo" | "yolox" | "pose";

export function fillFor(mode: RasterMode): string {
  return mode === "pose" ? "black" : "rgb(114,114,114)";
}

/** Source→destination rectangle for one draw. Pure, so both the coordinator
 * and the pose workers rasterise identically and it can be unit-tested. */
export interface DrawPlan {
  sx: number;
  sy: number;
  sw: number;
  sh: number;
  dw: number;
  dh: number;
}

/** Integer pixel bounds with a two-pixel filter margin for a transferable crop. */
export function transferCrop(
  crop: Box,
  width: number,
  height: number,
): { sx: number; sy: number; sw: number; sh: number; localCrop: Box } | null {
  const sx = Math.max(0, Math.floor(crop.x - 2));
  const sy = Math.max(0, Math.floor(crop.y - 2));
  const ex = Math.min(width, Math.ceil(crop.x + crop.width + 2));
  const ey = Math.min(height, Math.ceil(crop.y + crop.height + 2));
  if (ex <= sx || ey <= sy) return null;
  return {
    sx,
    sy,
    sw: ex - sx,
    sh: ey - sy,
    localCrop: { ...crop, x: crop.x - sx, y: crop.y - sy },
  };
}

/**
 * `crop` (a pose/TTA box) is stretched to fill the whole input; without a crop
 * the frame is letterboxed to preserve aspect ratio, which is what the detector
 * adapters were trained on. Out-of-frame crops are passed through unchanged so
 * the canvas clips them exactly as it always has.
 */
export function drawPlan(
  sourceW: number,
  sourceH: number,
  w: number,
  h: number,
  crop: Box | null | undefined,
): DrawPlan {
  if (crop)
    return {
      sx: crop.x,
      sy: crop.y,
      sw: crop.width,
      sh: crop.height,
      dw: w,
      dh: h,
    };
  // Whole frame scaled to fit: the source rect stays the full image and only
  // the destination shrinks (equivalent to the 5-argument drawImage form, which
  // is what the detector adapters were always fed).
  const ratio = Math.min(w / sourceW, h / sourceH);
  return {
    sx: 0,
    sy: 0,
    sw: sourceW,
    sh: sourceH,
    dw: sourceW * ratio,
    dh: sourceH * ratio,
  };
}

/**
 * RGBA pixels → planar CHW float tensor. Channel-major loops with the mode
 * branch hoisted out: this runs up to 32× per frame, so per-pixel branching
 * showed up in profiles.
 */
export function fillTensor(
  pixels: Uint8ClampedArray,
  w: number,
  h: number,
  mode: RasterMode,
  out: Float32Array,
): Float32Array {
  const plane = w * h;
  if (out.length !== 3 * plane)
    throw Error(
      `Tensor buffer is ${out.length} floats, expected ${3 * plane}.`,
    );
  if (mode === "pose") {
    for (let c = 0; c < 3; c++) {
      const off = c * plane,
        mean = POSE_MEAN[c],
        inv = 1 / POSE_STD[c];
      for (let i = 0; i < plane; i++)
        out[off + i] = (pixels[i * 4 + c] - mean) * inv;
    }
  } else if (mode === "yolox") {
    for (let c = 0; c < 3; c++) {
      const off = c * plane,
        src = 2 - c; // BGR
      for (let i = 0; i < plane; i++) out[off + i] = pixels[i * 4 + src];
    }
  } else {
    for (let c = 0; c < 3; c++) {
      const off = c * plane;
      for (let i = 0; i < plane; i++) out[off + i] = pixels[i * 4 + c] / 255;
    }
  }
  return out;
}

/** Reusable canvas + tensor buffer for one model input shape and mode. */
export class Rasterizer {
  private canvas: OffscreenCanvas;
  private ctx: OffscreenCanvasRenderingContext2D;
  private buffer: Float32Array;

  constructor(
    private w: number,
    private h: number,
    private mode: RasterMode,
  ) {
    this.canvas = new OffscreenCanvas(w, h);
    this.ctx = this.canvas.getContext("2d", { willReadFrequently: true })!;
    this.buffer = new Float32Array(3 * w * h);
  }

  /** Floats in one tensor of this rasterizer's shape. */
  get pixels(): number {
    return 3 * this.w * this.h;
  }

  /**
   * Draws `source` (an ImageBitmap/VideoFrame/canvas) and returns the CHW
   * tensor. `out` may be supplied by callers that need several tensors alive at
   * once (concurrent TTA passes); otherwise the shared buffer is reused.
   */
  tensor(
    source: CanvasImageSource,
    sourceW: number,
    sourceH: number,
    crop: Box | null | undefined,
    flip: boolean,
    out?: Float32Array,
  ): Float32Array {
    const plan = drawPlan(sourceW, sourceH, this.w, this.h, crop);
    const ctx = this.ctx;
    ctx.save();
    ctx.fillStyle = fillFor(this.mode);
    ctx.fillRect(0, 0, this.w, this.h);
    if (flip) {
      ctx.translate(this.w, 0);
      ctx.scale(-1, 1);
    }
    ctx.drawImage(
      source,
      plan.sx,
      plan.sy,
      plan.sw,
      plan.sh,
      0,
      0,
      plan.dw,
      plan.dh,
    );
    ctx.restore();
    const pixels = ctx.getImageData(0, 0, this.w, this.h).data;
    return fillTensor(pixels, this.w, this.h, this.mode, out ?? this.buffer);
  }
}

/** A frame prepared for rasterisation: one bitmap, shared by every pass. */
export class FrameSource {
  private bitmap: ImageBitmap | null = null;
  private closed = false;
  private frame: VideoFrame | null;

  private constructor(
    frame: VideoFrame,
    /** True when this source must close the frame itself (a pipeline clone). */
    private ownsFrame: boolean,
  ) {
    this.frame = frame;
  }

  /**
   * Wraps a frame the caller keeps managing (the decoder closes it).
   */
  static async create(frame: VideoFrame): Promise<FrameSource> {
    return FrameSource.prepare(frame, false);
  }

  /**
   * Takes ownership of a frame clone: dispose() closes it. The analysis
   * pipeline clones each frame so pose work can continue after the decoder has
   * moved on; the clone is what keeps the pixels alive.
   */
  static async own(clone: VideoFrame): Promise<FrameSource> {
    return FrameSource.prepare(clone, true);
  }

  /**
   * Converts the frame to a single ImageBitmap. Conversion is lazy and cheap
   * (it is a reference until something reads it), and falls back to drawing the
   * VideoFrame directly where ImageBitmap support is missing.
   */
  private static async prepare(
    frame: VideoFrame,
    ownsFrame: boolean,
  ): Promise<FrameSource> {
    const source = new FrameSource(frame, ownsFrame);
    try {
      source.bitmap = await createImageBitmap(frame);
      if (
        source.bitmap.width !== frame.displayWidth ||
        source.bitmap.height !== frame.displayHeight
      ) {
        // A scaled/odd bitmap would change the coordinate mapping, so keep the
        // frame itself rather than silently rasterising the wrong geometry.
        source.bitmap.close();
        source.bitmap = null;
      }
    } catch {
      source.bitmap = null;
    }
    return source;
  }

  /**
   * The frame bitmap, when one could be created. Pose workers receive cropped
   * transferable handles; the source keeps its own for detection and preview.
   */
  get bitmapImage(): ImageBitmap | null {
    return this.bitmap;
  }

  get image(): CanvasImageSource {
    return this.bitmap ?? this.frame ?? new OffscreenCanvas(1, 1);
  }

  get width(): number {
    return this.bitmap?.width ?? this.frame?.displayWidth ?? 0;
  }

  get height(): number {
    return this.bitmap?.height ?? this.frame?.displayHeight ?? 0;
  }

  /** Releases this source's own handle; worker copies are released by them. */
  dispose(): void {
    if (this.closed) return;
    this.closed = true;
    this.bitmap?.close();
    this.bitmap = null;
    if (this.ownsFrame) this.frame?.close();
    this.frame = null;
  }
}
