import type { PoseFrame, LiftedPose } from "./types";
import { reliable, quantile } from "./calibration";
// Bone lengths are priors relative to torso length, not subject-specific measurements.
const tree: [number, number, number, number][] = [
  [11, 12, 0.55, 1],
  [11, 5, 1, -1],
  [12, 6, 1, 1],
  [5, 7, 0.62, -1],
  [7, 9, 0.55, -1],
  [6, 8, 0.62, 1],
  [8, 10, 0.55, 1],
  [11, 13, 0.95, -1],
  [13, 15, 0.93, -1],
  [12, 14, 0.95, 1],
  [14, 16, 0.93, 1],
];
/** Reprojection-preserving, temporally disambiguated kinematic lifting. Both depth signs are feasible. */
export function liftSequence(frames: PoseFrame[]): PoseFrame[] {
  const result = frames.map((f) => ({
    ...f,
    lifted: undefined as LiftedPose | undefined,
  }));
  for (const seat of new Set(result.map((f) => f.seat))) {
    const seq = result
      .filter((f) => f.seat === seat)
      .sort((a, b) => a.time - b.time);
    const torso: number[] = [];
    for (const f of seq)
      for (const [a, b] of [
        [5, 11],
        [6, 12],
      ])
        if (
          f.identityConfidence >= 0.6 &&
          reliable(f.points[a]) &&
          reliable(f.points[b])
        )
          torso.push(
            Math.hypot(
              f.points[a].x - f.points[b].x,
              f.points[a].y - f.points[b].y,
            ),
          );
    const scale = quantile(torso, 0.9);
    if (torso.length < 10 || !Number.isFinite(scale) || scale < 25) continue;
    let previous: LiftedPose | undefined,
      previousTime = -Infinity;
    for (const f of seq) {
      if (f.time - previousTime > 0.2) previous = undefined;
      if (f.identityConfidence < 0.6 || !reliable(f.points[11])) {
        previous = undefined;
        continue;
      }
      const root = f.points[11];
      const points: LiftedPose["points"] = Array.from(
        { length: 17 },
        () => null,
      );
      points[11] = {
        x: 0,
        y: 0,
        z: 0,
        confidence: Math.min(0.35, root.confidence),
      };
      for (const [a, b, length, sign] of tree) {
        const pa = points[a],
          pb = f.points[b];
        if (!pa || !reliable(pb)) continue;
        const x = (pb.x - root.x) / scale,
          y = -(pb.y - root.y) / scale;
        const d2 = (x - pa.x) ** 2 + (y - pa.y) ** 2;
        if (d2 > length ** 2) continue; // Do not shorten the image projection to make a plausible skeleton.
        const dz = Math.sqrt(Math.max(0, length ** 2 - d2));
        const zs = [pa.z + dz, pa.z - dz];
        const prev = previous?.points[b];
        const z = prev
          ? zs.reduce((best, v) =>
              Math.abs(v - prev.z) < Math.abs(best - prev.z) ? v : best,
            )
          : pa.z + sign * dz;
        points[b] = {
          x,
          y,
          z,
          confidence: Math.min(0.35, pb.confidence, f.identityConfidence),
        };
      }
      f.lifted = {
        method: "kinematic-weak-perspective-v1",
        points,
        units: "torso lengths",
        ambiguous: true,
      };
      previous = f.lifted;
      previousTime = f.time;
    }
  }
  return result;
}
