import type { Box, Point } from "./types";
import { clamp } from "./geometry";
import { flipIndices } from "./keypoints";
function peak(data: Float32Array, start: number, n: number) {
  let k = 0;
  for (let i = 1; i < n; i++) if (data[start + i] > data[start + k]) k = i;
  const v = data[start + k];
  let offset = 0;
  if (k > 0 && k < n - 1) {
    const a = data[start + k - 1],
      c = data[start + k + 1],
      den = a - 2 * v + c;
    if (Math.abs(den) > 1e-8) offset = clamp((0.5 * (a - c)) / den, -0.5, 0.5);
  }
  return { position: k + offset, score: v };
}
export function decodeSimCC(
  x: Float32Array,
  y: Float32Array,
  nx: number,
  ny: number,
  crop: Box,
  joints: 17 | 133 = 133,
): Point[] {
  if (x.length !== joints * nx || y.length !== joints * ny)
    throw Error(
      `Pose model must output ${joints} joints as two SimCC tensors.`,
    );
  return Array.from({ length: joints }, (_, j) => {
    const a = peak(x, j * nx, nx),
      b = peak(y, j * ny, ny);
    const confidence = clamp(Math.min(a.score, b.score));
    return {
      x: crop.x + (a.position / nx) * crop.width,
      y: crop.y + (b.position / ny) * crop.height,
      confidence,
      origin: confidence >= 0.3 ? "observed" : "missing",
    };
  });
}
export function decodeHeatmap(
  data: Float32Array,
  w: number,
  h: number,
  crop: Box,
  joints: 17 | 133 = 133,
): Point[] {
  if (data.length !== joints * w * h)
    throw Error(`Heatmap model must output [1,${joints},H,W].`);
  return Array.from({ length: joints }, (_, j) => {
    const base = j * w * h;
    let k = 0;
    for (let i = 1; i < w * h; i++) if (data[base + i] > data[base + k]) k = i;
    let x = k % w,
      y = Math.floor(k / w);
    const refine = (a: number, b: number, c: number) => {
      const d = a - 2 * b + c;
      return Math.abs(d) < 1e-8 ? 0 : clamp((0.5 * (a - c)) / d, -0.5, 0.5);
    };
    if (x > 0 && x < w - 1)
      x += refine(data[base + k - 1], data[base + k], data[base + k + 1]);
    if (y > 0 && y < h - 1)
      y += refine(data[base + k - w], data[base + k], data[base + k + w]);
    const confidence = clamp(data[base + k]);
    return {
      x: crop.x + (x / w) * crop.width,
      y: crop.y + (y / h) * crop.height,
      confidence,
      origin: confidence >= 0.3 ? "observed" : "missing",
    };
  });
}
export function unflip(points: Point[], crop: Box): Point[] {
  if (points.length !== 17 && points.length !== 133)
    throw Error("Expected COCO-17 or COCO-WholeBody-133 points.");
  return flipIndices.slice(0, points.length).map((j) => {
    const point = points[j];
    return point.origin === "missing"
      ? { ...point }
      : { ...point, x: crop.x + crop.width - (point.x - crop.x) };
  });
}
export function fusePoses(poses: Point[][]): Point[] {
  return poses[0].map((_, j) => {
    const valid = poses.map((p) => p[j]).filter((p) => p.origin !== "missing"),
      total = valid.reduce((s, p) => s + p.confidence, 0);
    if (!total) return { ...poses[0][j], origin: "missing", confidence: 0 };
    const x = valid.reduce((s, p) => s + p.x * p.confidence, 0) / total,
      y = valid.reduce((s, p) => s + p.y * p.confidence, 0) / total;
    const disagreement = Math.sqrt(
      valid.reduce((s, p) => s + (p.x - x) ** 2 + (p.y - y) ** 2, 0) /
        valid.length,
    );
    return {
      x,
      y,
      confidence: clamp(total / poses.length / (1 + disagreement / 20)),
      origin: "observed",
      flags: disagreement > 20 ? ["TTA disagreement"] : [],
    };
  });
}
