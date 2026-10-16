import type { Vec3, CameraEstimate, Point } from "./types";
export const clamp = (v: number, min = 0, max = 1) =>
  Math.max(min, Math.min(max, v));
/** Linear-interpolated quantile of an ascending-sorted sample. */
export function quantile(sorted: readonly number[], q: number) {
  if (!sorted.length) return NaN;
  const i = (sorted.length - 1) * q,
    a = Math.floor(i),
    b = Math.ceil(i);
  return sorted[a] + (sorted[b] - sorted[a]) * (i - a);
}
/** Median of a sample in any order. Null when it is empty. */
export function median(values: readonly number[]): number | null {
  return values.length
    ? quantile(
        [...values].sort((a, b) => a - b),
        0.5,
      )
    : null;
}
export function rotateY(p: Vec3, azimuth: number): Vec3 {
  const t = (azimuth * Math.PI) / 180;
  return {
    x: p.x * Math.cos(t) + p.z * Math.sin(t),
    y: p.y,
    z: -p.x * Math.sin(t) + p.z * Math.cos(t),
  };
}
export function toBoatFrame(p: Vec3, azimuth: number): Vec3 {
  return rotateY(p, -azimuth);
}
export function angle3D(a: Vec3, b: Vec3, c: Vec3): number | null {
  const u = { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z },
    v = { x: c.x - b.x, y: c.y - b.y, z: c.z - b.z };
  const d = Math.hypot(u.x, u.y, u.z) * Math.hypot(v.x, v.y, v.z);
  return d < 1e-8
    ? null
    : (Math.acos(clamp((u.x * v.x + u.y * v.y + u.z * v.z) / d, -1, 1)) * 180) /
        Math.PI;
}
export function scaleFromSegment(
  pixels: number,
  meters: number,
  azimuth: number,
): number | null {
  if (pixels <= 0 || meters <= 0 || Math.abs(azimuth) > 75) return null;
  return (meters * Math.cos((azimuth * Math.PI) / 180)) / pixels;
}
export function estimateCamera(hips: Point[]): CameraEstimate {
  const valid = hips.filter(
    (p) => p.confidence > 0.5 && p.origin !== "missing",
  );
  let axis = { x: 1, y: 0 };
  if (valid.length > 1) {
    const dx = valid.at(-1)!.x - valid[0].x,
      dy = valid.at(-1)!.y - valid[0].y;
    const n = Math.hypot(dx, dy);
    if (n > 1) axis = { x: dx / n, y: dy / n };
  }
  return {
    azimuth: null,
    confidence: 0,
    source: "unresolved",
    axis,
    reason:
      "Seat alignment estimates only an image-plane axis. Monocular azimuth and depth are underconstrained; normalized angles are withheld.",
  };
}
export function metricVisibility(camera: CameraEstimate): {
  allowed: boolean;
  reason: string;
} {
  return {
    allowed:
      camera.source === "calibrated" &&
      camera.confidence >= 0.8 &&
      camera.azimuth !== null &&
      camera.azimuth < 60,
    reason:
      "Needs validated 3D pose and camera calibration. Film alongside, near perpendicular to the shell, with the whole rower visible.",
  };
}
