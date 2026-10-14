import type { Boat, CameraEstimate, Point, PoseFrame } from "./types";
import { clamp, estimateCamera } from "./geometry";
export const reliable = (p: Point | undefined): p is Point =>
  !!p &&
  (p.origin === "observed" || p.origin === "manual") &&
  p.confidence >= 0.5;
export function quantile(values: number[], q: number) {
  if (!values.length) return NaN;
  const a = [...values].sort((a, b) => a - b);
  return a[Math.round((a.length - 1) * q)];
}
const midpoint = (a: Point, b: Point) => ({
  x: (a.x + b.x) / 2,
  y: (a.y + b.y) / 2,
});
/** Weak-perspective anatomical cue estimator, NOT an intrinsic/extrinsic camera calibration. */
export function automaticCamera(
  frames: PoseFrame[],
  boat: Boat,
): CameraEstimate {
  const fallback = estimateCamera([]);
  const groups = new Map<number, PoseFrame[]>();
  for (const f of frames) {
    const g = groups.get(f.time) || [];
    g.push(f);
    groups.set(f.time, g);
  }
  const slopes: number[] = [];
  for (const group of groups.values()) {
    const valid = group
      .filter((f) => f.identityConfidence >= 0.6 && reliable(f.points[11]))
      .sort((a, b) => a.seat - b.seat);
    if (valid.length < 2) continue;
    const first = valid[0].points[11],
      last = valid.at(-1)!.points[11];
    const dx = (last.x - first.x) * (boat.bowLeft ? 1 : -1),
      dy = (last.y - first.y) * (boat.bowLeft ? 1 : -1);
    if (dx > 40) slopes.push(Math.atan2(dy, dx));
  }
  // A single rower provides no independent shell-axis cue. Do not treat torso tilt as camera roll.
  if (slopes.length < 10)
    return {
      ...fallback,
      reason:
        "Insufficient multi-seat axis evidence. Camera azimuth cannot be calibrated from a single unknown monocular view. 2D kinematics and ambiguous 3D previews remain available.",
    };
  const roll = quantile(slopes, 0.5),
    spread = quantile(slopes, 0.9) - quantile(slopes, 0.1);
  const axis = { x: Math.cos(roll), y: Math.sin(roll) };
  if (spread > 0.15)
    return {
      ...fallback,
      axis,
      reason:
        "Image-plane seat axis varies substantially. Panning/perspective or identity errors prevent a stable viewpoint estimate.",
    };
  const angles: number[] = [],
    disagreements: number[] = [];
  for (const f of frames) {
    if (
      f.identityConfidence < 0.6 ||
      ![5, 6, 11, 12].every((j) => reliable(f.points[j]))
    )
      continue;
    const p = f.points,
      shoulders = midpoint(p[5], p[6]),
      hips = midpoint(p[11], p[12]);
    const torso = Math.hypot(shoulders.x - hips.x, shoulders.y - hips.y);
    if (torso < 25) continue;
    const projected = (a: Point, b: Point) =>
      Math.abs((a.x - b.x) * axis.x + (a.y - b.y) * axis.y);
    const s = projected(p[5], p[6]) / (torso * 0.8),
      h = projected(p[11], p[12]) / (torso * 0.55);
    if (s > 1 || h > 1) continue;
    const sa = (Math.asin(clamp(s)) * 180) / Math.PI,
      ha = (Math.asin(clamp(h)) * 180) / Math.PI;
    disagreements.push(Math.abs(sa - ha));
    if (Math.abs(sa - ha) <= 15) angles.push((sa + ha) / 2);
  }
  const variation = quantile(angles, 0.9) - quantile(angles, 0.1);
  if (angles.length < 20 || variation > 25 || quantile(disagreements, 0.5) > 15)
    return {
      ...fallback,
      axis,
      reason:
        "Shoulder and hip projection cues are missing, inconsistent or unstable. Azimuth withheld rather than inferring camera geometry from the seat axis alone.",
    };
  return {
    axis,
    source: "estimated",
    azimuth: quantile(angles, 0.5),
    confidence:
      Math.min(0.55, (angles.length / 100) * 0.55) * (1 - variation / 50),
    uncertaintyDegrees: Math.max(20, variation),
    reason:
      "Experimental weak-perspective estimate using seat alignment and assumed shoulder/hip-to-torso proportions (0.80/0.55). Body shape, twist and foreshortening can invalidate it. Side/front ambiguity remains; this is NOT calibrated camera geometry and does not unlock normalized 3D measurements.",
  };
}
