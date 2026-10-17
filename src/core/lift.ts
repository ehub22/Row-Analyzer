// Learned 3D lift, behind the camera solve.
//
// Two models run on a clip. The pose model says where each joint is *in the
// image*, and that is an observation. This one says where it is *in space*, and
// that is an inference: no monocular image determines depth, so a lift is a
// learned prior about human shape and about rowing, and it is wrong in ways the
// image cannot see. Everything downstream of it has to be able to tell the two
// apart, which is why a lifted joint carries `origin: "inferred"` and never
// "observed", and why a lifted angle is reported as a lifted angle.
//
// What makes it usable here is that the lift does not have to be right on its
// own. The camera solve gives a focal length, a principal point, an azimuth and
// a depth for the shell; the pose model gives 2D joints that were observed; and
// a lifted pose that cannot be scaled and slid to land on those 2D joints,
// through that camera, is a lifted pose that is not talking about this rower.
// That check is cheap, it is exact, and it is the gate this module is built
// around: fit, then measure the reprojection, then refuse or report.
//
// The fit cannot find the depth — that is the monocular ambiguity, and a lift
// that claims to solve it is claiming more than the image holds. So the depth
// comes from the calibration (the seat line's depth, core/camera.ts) and the fit
// solves for the three things the image does constrain: the lift's scale, and
// where the rower sits in the image plane. Depth is an input, not an output,
// and a wrong seat spacing or a wrong lens shows up as a scale error here
// exactly as it does in the solve's own cues.
//
// Coordinate contract for a lift model: camera-frame metres, root relative, x
// to the image's right, y up, z away from the camera. That is the frame
// core/geometry.ts's `rotateY` maps to boat coordinates, so `toBoatFrame` with
// the solved azimuth is the whole conversion. A model with another convention
// (y down, z towards the camera, a different root) fails the reprojection gate
// loudly rather than quietly tilting every angle.
//
// Gated: with no weights configured this module is never called, and the UI says
// so. It is not a stub that returns zeros.

import type { CameraSolve, PoseFrame, Point, Vec3 } from "./types";
import type { Joint3D } from "./angles";
import { MIN_JOINT_CONFIDENCE } from "./angles";
import { seen } from "./occlusion";
import { solveAllowsAngles } from "./camera";
import { solveLinear3, toBoatFrame } from "./geometry";

/** One lift model's output: J joints, camera-frame metres, root relative. */
export interface LiftOutput {
  /** J × 3, row major: x right, y up, z away from the camera. */
  joints: ArrayLike<number>;
  /** J confidences in 0..1, one per joint. */
  confidence: ArrayLike<number>;
  jointCount: number;
}

/** The crop the lift is shown: CHW float32, the same layout the pose model gets. */
export interface LiftInput {
  raster: Float32Array;
  width: number;
  height: number;
}

/**
 * A lift model, as the worker builds it around an ONNX session.
 *
 * A function rather than a session, so the geometry below can be tested without
 * ONNX Runtime: tests pass an oracle that returns known 3D joints, the worker
 * passes `ort.InferenceSession.run`.
 */
export type LiftModel = (input: LiftInput) => Promise<LiftOutput>;

/** The lift speaks the 17-joint COCO body layout, which is the first 17 slots
 * of the COCO-WholeBody layout the rest of the pipeline uses. */
export const LIFT_BODY_JOINTS = 17;
/** Least observed 2D joints needed to place a lifted pose in the frame. Six is
 * enough to fix a scale and a translation with something left over to check. */
export const MIN_FIT_JOINTS = 6;
/** Reprojection error, as a fraction of the crop's height, that still reads as
 * the same rower. Generous on purpose: this is a wrong-lift detector, not a
 * precision gate, and the 2D keypoints themselves carry a couple of pixels. */
export const MAX_REPROJECTION_FRACTION = 0.06;
/** Floor on that limit, so a small crop does not tighten it into noise. */
export const MIN_REPROJECTION_PX = 5;
/** Fixed-point iterations for the scale fit. The pose's own depth extent is a
 * few percent of the distance to the shell, so this converges at once; the
 * iterations are there so it converges rather than because it is slow to. */
export const FIT_ITERATIONS = 4;
/** A gap-filled keypoint is an interpolation, so a joint lifted from one is
 * worth less than a joint lifted from something the model actually saw. */
const GAP_FILL_PENALTY = 0.7;
/** Plausible adult segment lengths in metres. A factor-of-two sanity check on
 * the fitted scale, not an anthropometric model: rowers come in sizes, and a
 * thigh of 1.2 m or 8 cm is not a size. */
const SEGMENTS: {
  name: string;
  a: number;
  b: number;
  range: [number, number];
}[] = [
  { name: "thigh", a: 11, b: 13, range: [0.3, 0.62] },
  { name: "thigh", a: 12, b: 14, range: [0.3, 0.62] },
  { name: "shin", a: 13, b: 15, range: [0.3, 0.62] },
  { name: "shin", a: 14, b: 16, range: [0.3, 0.62] },
  { name: "trunk", a: 11, b: 5, range: [0.38, 0.78] },
  { name: "trunk", a: 12, b: 6, range: [0.38, 0.78] },
  { name: "upper arm", a: 5, b: 7, range: [0.2, 0.45] },
  { name: "upper arm", a: 6, b: 8, range: [0.2, 0.45] },
  { name: "forearm", a: 7, b: 9, range: [0.18, 0.42] },
  { name: "forearm", a: 8, b: 10, range: [0.18, 0.42] },
];

/** What placing one lifted pose produced, or why it was refused. */
export interface LiftResult {
  /** Boat-frame metres, indexed by the COCO body joint number. Undefined where
   * nothing was lifted: a joint the 2D pipeline did not see stays absent,
   * because a lift must not resurrect a keypoint that was discarded. */
  joints: (Joint3D | undefined)[];
  /** Reprojection error against the observed keypoints, pixels. */
  reprojection: number | null;
  /** Lift units to metres, as fitted. */
  scale: number | null;
  /** Where the root was placed, metres: image-plane offset and depth. */
  placement: { x: number; y: number; depth: number } | null;
  /** Set when nothing was lifted, and says what to do about it. */
  withheld: string | null;
}

/** Nothing lifted, with the reason. Every joint absent, nothing invented. */
function refuse(reason: string): LiftResult {
  return {
    joints: [],
    reprojection: null,
    scale: null,
    placement: null,
    withheld: reason,
  };
}

/** A keypoint that may anchor the fit: placed by the model or by the user.
 * Gap fills and discards are not anchors — fitting a 3D pose to an
 * interpolation would be inventing the very thing being checked. */
const anchorable = (p: Point | undefined): p is Point =>
  seen(p) && (p.origin === "observed" || p.origin === "manual");

/** Reads a model output, or says why it is not one this adapter can place. */
function readOutput(output: LiftOutput): { p: Vec3; c: number }[] | null {
  const j = output.jointCount;
  if (j !== LIFT_BODY_JOINTS) return null;
  if (output.joints.length < j * 3 || output.confidence.length < j) return null;
  const out: { p: Vec3; c: number }[] = [];
  for (let i = 0; i < j; i++) {
    const x = output.joints[i * 3]!,
      y = output.joints[i * 3 + 1]!,
      z = output.joints[i * 3 + 2]!;
    if (![x, y, z].every(Number.isFinite)) return null;
    out.push({ p: { x, y, z }, c: output.confidence[i] ?? 0 });
  }
  return out;
}

/**
 * Places a lifted pose in boat coordinates.
 *
 * This is the whole of the geometry, and it is pure: no model, no ONNX, no
 * video. `liftFrame` below just runs the model first and hands the output here.
 */
export function placeLift(
  output: LiftOutput,
  frame: PoseFrame,
  solve: CameraSolve,
  imageSize: { width: number; height: number },
): LiftResult {
  // No calibration, no boat frame, no angles. The solve's own reason is the
  // honest one, so it is passed through rather than paraphrased.
  if (!solveAllowsAngles(solve))
    return refuse(
      solve.reason ||
        "The camera is not solved, so there is no boat frame to lift into.",
    );
  const lifted = readOutput(output);
  if (!lifted)
    return refuse(
      `The lift returned ${output.jointCount} joints; this adapter places the ` +
        `${LIFT_BODY_JOINTS}-joint COCO body layout, x right, y up, z away from ` +
        `the camera, in root-relative metres.`,
    );
  const depth = solve.distance!,
    f = solve.focalPx!,
    cx = solve.principal?.x ?? imageSize.width / 2,
    cy = solve.principal?.y ?? imageSize.height / 2;

  const anchors: { i: number; u: number; v: number; w: number }[] = [];
  for (let i = 0; i < lifted.length; i++) {
    const p = frame.points[i];
    if (!anchorable(p) || p.confidence < MIN_JOINT_CONFIDENCE) continue;
    anchors.push({ i, u: p.x, v: p.y, w: p.confidence * lifted[i]!.c });
  }
  if (anchors.length < MIN_FIT_JOINTS)
    return refuse(
      `Only ${anchors.length} of the lifted joints land on keypoints that were ` +
        `actually seen; ${MIN_FIT_JOINTS} are needed to place a pose in the ` +
        `frame. The feet are usually the missing ones — that is fine — but not ` +
        `this many of the rest.`,
    );

  // Scale and in-plane position, by weighted least squares against the observed
  // keypoints, with the depth fixed at the seat line's. The projection is
  // u = cx + f·(s·X + tx)/(s·Z + depth), which is linear in (s, tx, ty) once the
  // denominator is known, and the denominator needs s — so iterate.
  //
  // The first guess comes from the crop rather than from the model's own units,
  // which is what makes this unit-agnostic: a lift in millimetres, in metres or
  // in its own normalized space all start from the same place. The rower's
  // height in the image, at a known depth and a known focal length, is a length
  // in metres; divided by the lift's own height it is the scale.
  const ys = lifted.map((l) => l.p.y);
  const liftHeight = Math.max(...ys) - Math.min(...ys);
  const seenYs = anchors.map((a) => frame.points[a.i]!.y);
  const imageHeight = Math.max(...seenYs) - Math.min(...seenYs);
  let scale =
    liftHeight > 1e-6 && imageHeight > 1
      ? ((imageHeight * depth) / f / liftHeight) * 1.0
      : 1;
  if (!(scale > 0) || !Number.isFinite(scale)) scale = 1;
  let tx = 0,
    ty = 0;
  for (let it = 0; it < FIT_ITERATIONS; it++) {
    const A = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
      ],
      y = [0, 0, 0];
    let usable = 0;
    for (const a of anchors) {
      const L = lifted[a.i]!.p,
        Z = scale * L.z + depth;
      if (!(Z > 0.1)) continue;
      usable++;
      const rows: [number[], number][] = [
        [[L.x, 1, 0], ((a.u - cx) * Z) / f],
        [[L.y, 0, 1], ((cy - a.v) * Z) / f],
      ];
      for (const [col, target] of rows)
        for (let m = 0; m < 3; m++) {
          for (let n = 0; n < 3; n++) A[m]![n]! += a.w * col[m]! * col[n]!;
          y[m]! += a.w * col[m]! * target;
        }
    }
    if (usable < MIN_FIT_JOINTS)
      return refuse(
        "The lift puts joints behind the camera once it is placed at the shell's " +
          "depth: its depth axis points the wrong way, or its units are not " +
          "metres.",
      );
    const s = solveLinear3(A, y);
    if (!s || !Number.isFinite(s[0]) || !(s[0] > 0))
      return refuse(
        "The lift's scale could not be fitted against the observed keypoints; " +
          "they do not constrain it, which means they do not describe one pose.",
      );
    [scale, tx, ty] = s;
  }

  // Exact reprojection with the fitted placement, weighted the same way.
  let sum = 0,
    weight = 0;
  for (const a of anchors) {
    const L = lifted[a.i]!.p,
      X = scale * L.x + tx,
      Y = scale * L.y + ty,
      Z = scale * L.z + depth;
    if (!(Z > 0.1))
      return refuse(
        "The lifted pose reaches behind the camera; its depth convention is not " +
          "the one this adapter places (z away from the camera, in metres).",
      );
    const du = cx + (f * X) / Z - a.u,
      dv = cy - (f * Y) / Z - a.v;
    sum += a.w * (du * du + dv * dv);
    weight += a.w;
  }
  const reprojection = Math.sqrt(sum / Math.max(1e-9, weight));
  const limit = Math.max(
    MIN_REPROJECTION_PX,
    MAX_REPROJECTION_FRACTION * frame.box.height,
  );
  if (reprojection > limit)
    return refuse(
      `The lifted pose lands ${reprojection.toFixed(1)} px from the keypoints it ` +
        `was fitted to, more than the ${limit.toFixed(0)} px this adapter will ` +
        `accept: the lift and the 2D pose disagree about where this rower is, so ` +
        `there are no angles to report from it.`,
    );

  // The scale the fit chose has to imply a human being. This catches a lift in
  // the wrong units, or one rooted somewhere the fit compensated for by making
  // the rower enormous.
  const placed = (i: number): Vec3 | null => {
    const L = lifted[i]?.p;
    if (!L) return null;
    return { x: scale * L.x + tx, y: scale * L.y + ty, z: scale * L.z + depth };
  };
  for (const seg of SEGMENTS) {
    const a = placed(seg.a),
      b = placed(seg.b);
    if (!a || !b) continue;
    const len = Math.hypot(b.x - a.x, b.y - a.y, b.z - a.z);
    if (len < seg.range[0] || len > seg.range[1])
      return refuse(
        `The fitted scale of ${scale.toFixed(3)} implies a ${seg.name} of ` +
          `${(len * 100).toFixed(0)} cm, outside the ` +
          `${(seg.range[0] * 100).toFixed(0)}–${(seg.range[1] * 100).toFixed(0)} cm ` +
          `this adapter will believe: the lift's units are not what it claims, or ` +
          `it is rooted somewhere else.`,
      );
  }

  // Into boat coordinates: one rotation about the vertical, by the solved
  // azimuth. Angles are invariant to the translation that comes with it, so the
  // origin here — the point on the shell axis the camera was aimed at — only
  // matters for heights, which are reported as differences.
  const azimuth = solve.azimuth!;
  // Half the limit costs half the confidence: a fit that is merely poor should
  // still report, at a confidence that says it was poor, while a fit at the
  // limit is refused outright above.
  const quality = Math.max(0, 1 - reprojection / (2 * limit));
  const joints: (Joint3D | undefined)[] = [];
  for (let i = 0; i < lifted.length; i++) {
    const p2 = frame.points[i];
    // A joint the 2D pipeline did not see is not lifted, whatever the model
    // says about it. This is the same rule core/occlusion.ts applies to feet and
    // core/knees.ts to a knee behind the gunwale: the discard propagates, so a
    // guess cannot re-enter the numbers as a 3D guess.
    if (!seen(p2) || p2.confidence < MIN_JOINT_CONFIDENCE) {
      joints.push(undefined);
      continue;
    }
    const cam = placed(i);
    if (!cam) {
      joints.push(undefined);
      continue;
    }
    const boat = toBoatFrame(cam, azimuth);
    joints.push({
      ...boat,
      confidence:
        Math.min(p2.confidence, lifted[i]!.c) *
        quality *
        solve.confidence *
        (p2.origin === "inferred" ? GAP_FILL_PENALTY : 1),
      origin: "inferred",
    });
  }
  return {
    joints,
    reprojection,
    scale,
    placement: { x: tx, y: ty, depth },
    withheld: null,
  };
}

/** Runs a lift model on one frame's crop and places what came back. */
export async function liftFrame(
  model: LiftModel,
  input: LiftInput,
  frame: PoseFrame,
  solve: CameraSolve,
  imageSize: { width: number; height: number },
): Promise<LiftResult> {
  const output = await model(input);
  return placeLift(output, frame, solve, imageSize);
}

/** Whether a lift is worth running at all on this session. */
export function liftAvailable(solve: CameraSolve): {
  ok: boolean;
  reason: string;
} {
  return solveAllowsAngles(solve)
    ? { ok: true, reason: "Camera solved; a lift would be placed against it." }
    : {
        ok: false,
        reason:
          solve.reason ||
          "The camera is not solved, so a lift would have no boat frame to be placed in.",
      };
}
