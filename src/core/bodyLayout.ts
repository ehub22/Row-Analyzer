import type { Point } from "./types";
import { missing } from "./types";

export type PoseJoints = 17 | 133;
export const MODEL_UNAVAILABLE_FLAG =
  "Not provided by the selected body-only model";

const unavailable = (): Point => ({
  ...missing(),
  flags: [MODEL_UNAVAILABLE_FLAG],
});

/** Keep the version-1 session's 133 slots without inventing model observations. */
export function padPose(points: Point[], joints: PoseJoints): Point[] {
  if (joints === 133) {
    if (points.length !== 133) throw Error("Expected 133 whole-body joints.");
    return points;
  }
  if (points.length !== 17 && points.length !== 133)
    throw Error("Expected 17 body joints before padding.");
  return points.length === 17
    ? [...points, ...Array.from({ length: 116 }, unavailable)]
    : points.map((point, i) =>
        i < 17 || point.origin === "manual" ? point : unavailable(),
      );
}
