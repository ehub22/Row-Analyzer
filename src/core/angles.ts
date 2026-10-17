// Normalized joint angles.
//
// "Normalized" is the word ROADMAP.md uses, and it means something specific:
// an angle in the boat's frame, so that a knee at the catch reads the same
// number whatever corner of the launch the camera was sitting in. That is the
// whole point of solving the camera. A 2D image angle cannot do this — at 24°
// off square the same knee reads several degrees different, and the repo's
// standing instruction is explicit: never promise viewpoint-invariant angles
// without calibration, and no 2D cosine correction.
//
// So nothing here looks at pixels. It takes joints that are already in boat
// coordinates, in metres — produced by core/lift.ts from a learned 3D lift and
// a calibrated camera — and returns angles with a confidence and a reason. An
// angle whose joints were not all lifted is withheld, not estimated: the same
// rule core/occlusion.ts applies to a knee behind the gunwale.
//
// Frame convention, shared with core/geometry.ts and
// tests/fixtures/rowingSim3d.ts: x along the shell towards the rowers' feet
// (the stern), y up, z to a rower's right. A rower faces the stern, so
// "forward" — over the knees at the catch — is +x, and layback is negative
// while the trunk is forward of vertical and positive past it, which is the
// convention coaches already use.

import type { Metric, Origin, Stroke, Vec3 } from "./types";
import { withheld } from "./strokes";
import { angle3D } from "./geometry";

/** A joint in boat coordinates, metres, as core/lift.ts produces it. */
export interface Joint3D extends Vec3 {
  confidence: number;
  /**
   * Always "inferred" for a lifted joint, or "missing" when it was withheld.
   * A model's depth is never an observation, and the repo's rule is that a
   * guess must not be able to pass for one, so there is no "observed" here.
   */
  origin: Origin;
}

/**
 * The angles this module reports.
 *
 * The thigh is in the list alongside the knee because on the water it is the
 * one leg angle that is fully observed: feet are strapped below the gunwale, so
 * the ankle is not seen, so the knee and shin angles are withheld, while hip and
 * knee both are. A thigh that reads "parallel" at the catch is a real
 * measurement from a launch clip; a knee angle from the same clip is not, and
 * this module will not pretend otherwise.
 */
export const ANGLE_NAMES = [
  "kneeLeft",
  "kneeRight",
  "hipLeft",
  "hipRight",
  "elbowLeft",
  "elbowRight",
  "thighLeft",
  "thighRight",
  "shinLeft",
  "shinRight",
  "layback",
] as const;
export type AngleName = (typeof ANGLE_NAMES)[number];

/** Degrees, so every angle carries the same unit. */
export const ANGLE_UNIT = "deg";

/** Below this a lifted joint is a guess about a joint the camera could not
 * see, and every angle through it is withheld. Matches the 0.3 the rest of the
 * pipeline uses for "observed". */
export const MIN_JOINT_CONFIDENCE = 0.3;

/** COCO-WholeBody indices, the layout the pose model already outputs. */
const JOINTS = {
  shoulderL: 5,
  shoulderR: 6,
  elbowL: 7,
  elbowR: 8,
  wristL: 9,
  wristR: 10,
  hipL: 11,
  hipR: 12,
  kneeL: 13,
  kneeR: 14,
  ankleL: 15,
  ankleR: 16,
} as const;

/** One joint an angle is built from, named so a refusal can name it back. */
interface Part {
  name: string;
  joint: Joint3D | undefined;
}

const usable = (p: Part) =>
  p.joint !== undefined &&
  p.joint.origin !== "missing" &&
  p.joint.confidence >= MIN_JOINT_CONFIDENCE;

/** Why one part could not be used, in the words a report can print. */
function fault(p: Part): string {
  if (p.joint === undefined || p.joint.origin === "missing")
    return `${p.name} was not lifted in this frame`;
  return (
    `${p.name} was lifted at ${Math.round(p.joint.confidence * 100)}% of the ` +
    `${Math.round(MIN_JOINT_CONFIDENCE * 100)}% an angle needs`
  );
}

/**
 * Interior angle at the middle joint, in degrees.
 *
 * Interior, not the deflection: a straight leg reads 180 and a compressed one
 * reads small, which is how the 2D knee measurements in this repo already read
 * and how coaches describe them.
 */
function interior(what: string, a: Part, b: Part, c: Part): Metric {
  const parts = [a, b, c];
  const bad = parts.find((p) => !usable(p));
  if (bad)
    return withheld(
      ANGLE_UNIT,
      `${what} needs ${parts.map((p) => p.name).join(", ")}: ${fault(bad)}`,
    );
  const value = angle3D(a.joint!, b.joint!, c.joint!);
  if (value === null)
    return withheld(ANGLE_UNIT, `${what}: the joints are coincident`);
  return {
    value,
    unit: ANGLE_UNIT,
    confidence: Math.min(
      a.joint!.confidence,
      b.joint!.confidence,
      c.joint!.confidence,
    ),
  };
}

/**
 * Angle of a boat-frame segment from the vertical, in degrees, positive towards
 * the stern.
 *
 * This is what the boat frame buys that the image cannot give: the shin at the
 * catch is a statement about the rower, not about where the camera was. Only x
 * and y enter — the lateral component is a different measurement (boat set,
 * trunk twist) and stays withheld.
 */
function fromVertical(what: string, tail: Part, head: Part): Metric {
  const parts = [tail, head];
  const bad = parts.find((p) => !usable(p));
  if (bad)
    return withheld(
      ANGLE_UNIT,
      `${what} needs ${parts.map((p) => p.name).join(" and ")}: ${fault(bad)}`,
    );
  const dx = head.joint!.x - tail.joint!.x,
    dy = head.joint!.y - tail.joint!.y;
  if (Math.hypot(dx, dy) < 1e-6)
    return withheld(ANGLE_UNIT, `${what}: the segment has no length`);
  return {
    value: (Math.atan2(dx, dy) * 180) / Math.PI,
    unit: ANGLE_UNIT,
    confidence: Math.min(tail.joint!.confidence, head.joint!.confidence),
  };
}

/** One frame's angles, aligned to a PoseFrame. */
export interface AngleFrame {
  time: number;
  seat: number;
  angles: Record<AngleName, Metric>;
  /**
   * Reprojection error of the lifted pose against the keypoints that were
   * actually seen, pixels. A lifted pose that does not land on the rower it was
   * lifted from is withheld upstream; this is here so a chart can shade by it.
   */
  reprojection: number | null;
}

/**
 * Angles for one lifted pose in boat coordinates.
 *
 * `joints` is indexed by the COCO-WholeBody joint number and may be sparse: a
 * knee the gunwale hid, or feet the hull hid, simply are not there, and every
 * angle through them comes back withheld with the reason rather than filled in.
 */
export function jointAngles(
  joints: readonly (Joint3D | undefined)[],
  reprojection: number | null = null,
): Record<AngleName, Metric> {
  const part = (name: string, i: number): Part => ({ name, joint: joints[i] });
  const hipL = part("the left hip", JOINTS.hipL),
    hipR = part("the right hip", JOINTS.hipR),
    kneeL = part("the left knee", JOINTS.kneeL),
    kneeR = part("the right knee", JOINTS.kneeR),
    ankleL = part("the left ankle", JOINTS.ankleL),
    ankleR = part("the right ankle", JOINTS.ankleR),
    shoulderL = part("the left shoulder", JOINTS.shoulderL),
    shoulderR = part("the right shoulder", JOINTS.shoulderR),
    elbowL = part("the left elbow", JOINTS.elbowL),
    elbowR = part("the right elbow", JOINTS.elbowR),
    wristL = part("the left wrist", JOINTS.wristL),
    wristR = part("the right wrist", JOINTS.wristR);
  return {
    kneeLeft: interior("The left knee angle", hipL, kneeL, ankleL),
    kneeRight: interior("The right knee angle", hipR, kneeR, ankleR),
    hipLeft: interior("The left hip angle", shoulderL, hipL, kneeL),
    hipRight: interior("The right hip angle", shoulderR, hipR, kneeR),
    elbowLeft: interior("The left elbow angle", shoulderL, elbowL, wristL),
    elbowRight: interior("The right elbow angle", shoulderR, elbowR, wristR),
    thighLeft: fromVertical("The left thigh angle", hipL, kneeL),
    thighRight: fromVertical("The right thigh angle", hipR, kneeR),
    shinLeft: fromVertical("The left shin angle", ankleL, kneeL),
    shinRight: fromVertical("The right shin angle", ankleR, kneeR),
    // Trunk from vertical, averaged over both sides: a sweep rower's shoulders
    // are not square to the shell, and one side alone would read the twist as a
    // lean.
    layback: laybackAngle(
      joints[JOINTS.shoulderL],
      joints[JOINTS.shoulderR],
      joints[JOINTS.hipL],
      joints[JOINTS.hipR],
    ),
  };
}

/**
 * Layback: the trunk's angle from the boat's vertical, negative forward of it
 * (the catch) and positive past it (the finish).
 */
export function laybackAngle(
  shoulderL: Joint3D | undefined,
  shoulderR: Joint3D | undefined,
  hipL: Joint3D | undefined,
  hipR: Joint3D | undefined,
): Metric {
  const shoulders = [shoulderL, shoulderR].filter(
    (p): p is Joint3D =>
      p !== undefined &&
      p.origin !== "missing" &&
      p.confidence >= MIN_JOINT_CONFIDENCE,
  );
  const hips = [hipL, hipR].filter(
    (p): p is Joint3D =>
      p !== undefined &&
      p.origin !== "missing" &&
      p.confidence >= MIN_JOINT_CONFIDENCE,
  );
  if (!shoulders.length || !hips.length)
    return withheld(
      ANGLE_UNIT,
      "Layback needs a lifted shoulder and a lifted hip",
    );
  const mean = (ps: readonly Joint3D[]) => ({
    x: ps.reduce((s, p) => s + p.x, 0) / ps.length,
    y: ps.reduce((s, p) => s + p.y, 0) / ps.length,
  });
  const s = mean(shoulders),
    h = mean(hips),
    dx = s.x - h.x,
    dy = s.y - h.y;
  if (Math.hypot(dx, dy) < 1e-6)
    return withheld(ANGLE_UNIT, "Layback: the trunk has no length");
  return {
    value: (-Math.atan2(dx, dy) * 180) / Math.PI,
    unit: ANGLE_UNIT,
    confidence: Math.min(
      ...shoulders.map((p) => p.confidence),
      ...hips.map((p) => p.confidence),
    ),
  };
}

/**
 * The angle values at one stroke's events.
 *
 * Per-frame angles are the raw material; what a coach asks for is the knee at
 * the catch and the layback at the finish. This reads them off the per-frame
 * rows nearest each event, and reports the spread across the seat's stroke as
 * the confidence's support — one frame is a sample, not a measurement.
 */
export function anglesAtStroke(
  rows: readonly AngleFrame[],
  stroke: Stroke,
): { atCatch: Record<AngleName, Metric>; atFinish: Record<AngleName, Metric> } {
  const at = (time: number, label: string) => {
    const seat = rows.filter((r) => r.seat === stroke.seat);
    if (!seat.length)
      return emptyAngles(`${label}: no lifted frames for seat ${stroke.seat}`);
    // Nearest row, but never further away than a fifth of a stroke: an event
    // whose neighbourhood was withheld should not borrow an angle from the
    // other end of the slide.
    const span = Math.max(1e-6, stroke.nextCatch - stroke.catch);
    const near = seat
      .map((r) => ({ r, d: Math.abs(r.time - time) }))
      .sort((a, b) => a.d - b.d)[0];
    if (near.d > span / 5)
      return emptyAngles(
        `${label}: the nearest lifted frame is ${near.d.toFixed(2)} s away, ` +
          `more than a fifth of a stroke`,
      );
    return near.r.angles;
  };
  return {
    atCatch: at(stroke.catch, "The catch"),
    atFinish: at(stroke.finish, "The finish"),
  };
}

function emptyAngles(reason: string): Record<AngleName, Metric> {
  return Object.fromEntries(
    ANGLE_NAMES.map((n) => [n, withheld(ANGLE_UNIT, reason)]),
  ) as Record<AngleName, Metric>;
}

/**
 * The range an angle covers over a seat's lifted frames, and how much of them
 * was lifted.
 *
 * Reported as a metric at the 5th and 95th percentile rather than min and max,
 * because a single frame where the lift lost a wrist should not set the range a
 * coach is told about.
 */
export function angleRange(
  rows: readonly AngleFrame[],
  seat: number,
  name: AngleName,
): { lo: Metric; hi: Metric; observedFraction: number } {
  const seatRows = rows.filter((r) => r.seat === seat);
  const values = seatRows
    .map((r) => r.angles[name])
    .filter((m) => m.value !== null);
  const fraction = seatRows.length ? values.length / seatRows.length : 0;
  if (values.length < 10)
    return {
      lo: withheld(
        ANGLE_UNIT,
        `Only ${values.length} lifted frames carry ${name} for seat ${seat}`,
      ),
      hi: withheld(ANGLE_UNIT, `Insufficient lifted frames for ${name}`),
      observedFraction: fraction,
    };
  const sorted = values.map((m) => m.value!).sort((a, b) => a - b);
  const pick = (q: number) => {
    const i = (sorted.length - 1) * q,
      a = sorted[Math.floor(i)]!,
      b = sorted[Math.ceil(i)]!;
    return a + (b - a) * (i - Math.floor(i));
  };
  const confidence = Math.min(
    ...values.map((m) => m.confidence),
    Math.min(1, fraction + 0.2),
  );
  return {
    lo: { value: pick(0.05), unit: ANGLE_UNIT, confidence },
    hi: { value: pick(0.95), unit: ANGLE_UNIT, confidence },
    observedFraction: fraction,
  };
}
