import type { Box, Point, PoseFrame } from "./types";
import { paddedCrop } from "./detection";
import { assessPose } from "./tta";

export interface CadenceState {
  frameIndex: number;
  time: number;
  lastDetectorFrame: number;
  lastDetectorTime: number;
  completedFrames: number;
  detectorMs: number;
  poseMs: number;
  overlapping: boolean;
  recentRefreshes: number;
}

/** Never skip a detector anchor, and only predict when detection is the bottleneck. */
export function usePredictedBoxes(state: CadenceState): boolean {
  if (
    state.completedFrames < 24 ||
    state.frameIndex % 3 === 0 ||
    state.frameIndex - state.lastDetectorFrame > 2 ||
    state.time - state.lastDetectorTime > 0.12 ||
    state.recentRefreshes >= 3
  )
    return false;
  if (state.detectorMs <= 0 || state.poseMs <= 0) return false;
  return state.overlapping
    ? state.detectorMs >= state.poseMs * 2
    : state.detectorMs >= state.poseMs * 0.3;
}

const center = (b: Box) => ({ x: b.x + b.width / 2, y: b.y + b.height / 2 });
const anchors = [5, 6, 11, 12];
const observed = (p: Point) => p?.origin === "observed" && p.confidence >= 0.5;

/** Predict a short-lived crop from the motion of visible shoulders and hips. */
export function predictBoxes(
  latest: PoseFrame[] | null,
  previous: PoseFrame[] | null,
  time: number,
  width: number,
  height: number,
  region: { left: number; top: number; right: number; bottom: number },
  seats: number,
): Box[] | null {
  if (
    !latest ||
    !previous ||
    latest.length !== seats ||
    previous.length !== seats ||
    time <= latest[0].time ||
    time - latest[0].time > 0.1
  )
    return null;
  const boxes: Box[] = [];
  for (let i = 0; i < seats; i++) {
    const now = latest[i],
      before = previous[i];
    if (
      now.seat !== before.seat ||
      now.identityConfidence < 0.6 ||
      before.identityConfidence < 0.5 ||
      now.points[11]?.origin === "missing" ||
      now.points[12]?.origin === "missing"
    )
      return null;
    const shifts = anchors
      .filter((j) => observed(now.points[j]) && observed(before.points[j]))
      .map((j) => ({
        x: now.points[j].x - before.points[j].x,
        y: now.points[j].y - before.points[j].y,
      }));
    if (shifts.length < 2) return null;
    const dx = shifts.reduce((sum, p) => sum + p.x, 0) / shifts.length;
    const dy = shifts.reduce((sum, p) => sum + p.y, 0) / shifts.length;
    const b = now.box;
    if (Math.abs(dx) > b.width * 0.1 || Math.abs(dy) > b.height * 0.1)
      return null;
    const box = { ...b, x: b.x + dx, y: b.y + dy, confidence: 0 };
    const c = center(box);
    if (
      box.x < 0 ||
      box.y < 0 ||
      box.x + box.width > width ||
      box.y + box.height > height ||
      (c.x / width) * 100 <= region.left + 1 ||
      (c.x / width) * 100 >= region.right - 1 ||
      (c.y / height) * 100 <= region.top + 1 ||
      (c.y / height) * 100 >= region.bottom - 1
    )
      return null;
    boxes.push(box);
  }
  for (let i = 0; i < boxes.length; i++)
    for (let j = i + 1; j < boxes.length; j++) {
      const a = boxes[i],
        b = boxes[j];
      const overlap =
        Math.max(
          0,
          Math.min(a.x + a.width, b.x + b.width) - Math.max(a.x, b.x),
        ) *
        Math.max(
          0,
          Math.min(a.y + a.height, b.y + b.height) - Math.max(a.y, b.y),
        );
      if (overlap / Math.min(a.width * a.height, b.width * b.height) > 0.65)
        return null;
    }
  return boxes;
}

/** A crop that clipped confident body joints must be re-detected on this frame. */
export function needsFreshDetection(
  points: Point[],
  box: Box,
  width: number,
  height: number,
  gatedJoints: readonly number[],
): boolean {
  if (!assessPose(points, 0.5, 0.3, gatedJoints).accepted) return true;
  const crop = paddedCrop(box, width, height, 1.25);
  const marginX = crop.width * 0.04,
    marginY = crop.height * 0.04;
  return anchors.some((j) => {
    const p = points[j];
    return (
      observed(p) &&
      (p.x < crop.x + marginX ||
        p.x > crop.x + crop.width - marginX ||
        p.y < crop.y + marginY ||
        p.y > crop.y + crop.height - marginY)
    );
  });
}
