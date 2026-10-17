// The 3D lift, and what it is allowed to say.
//
// The oracle below is the fixture's own camera-frame joints, root relative, in
// metres. It stands in for a trained lift so that the adapter's *geometry* can
// be checked against ground truth: the scale fit, the placement against the
// observed keypoints, the rotation into boat coordinates, and the refusals. It
// is not evidence that any published lift behaves this way — that is ROADMAP.md
// milestone 8's gate, and it needs a rowing benchmark, not a simulator.
//
// What is being claimed here, and what is not:
//   * an honest lift placed against a solved camera reproduces the boat-frame
//     angles exactly, and beats the same angles measured in the image by the
//     margin the azimuth implies;
//   * a lift that disagrees with the keypoints, or whose axes point the wrong
//     way, is refused rather than reported;
//   * a joint the 2D pipeline discarded stays discarded — the lift does not get
//     to resurrect a knee the gunwale hid;
//   * the depth axis's *sign* is not something the image can check on a nearly
//     planar pose, so that convention is a contract with the model, and the test
//     says so instead of implying a gate that does not exist.
import { describe, expect, it } from "vitest";
import {
  JOINT3D,
  JOINT3D_NAMES,
  simulateCrew,
  type Joint3DName,
  type Sim3DResult,
  type Sim3DTruth,
} from "./fixtures/rowingSim3d";
import {
  LIFT_BODY_JOINTS,
  liftAvailable,
  liftFrame,
  placeLift,
  type LiftOutput,
} from "../src/core/lift";
import { jointAngles } from "../src/core/angles";
import { solveCamera } from "../src/core/camera";
import { hiddenJoints, suppressHiddenJoints } from "../src/core/occlusion";
import { angle3D, rotateY } from "../src/core/geometry";
import type {
  BoatCalibration,
  CameraSolve,
  PoseFrame,
  Vec3,
} from "../src/core/types";

const IMAGE = { width: 1280, height: 720 };
const EIGHT = { seats: 8, bowLeft: true };

/** Feet strapped below the gunwale, as on the water: the model's guess for them
 * is discarded before anything downstream can use it. */
function prepared(sim: Sim3DResult, feetHidden = true): PoseFrame[] {
  const hidden = hiddenJoints({ feetHidden });
  return sim.frames.map((f) => ({
    ...f,
    points: suppressHiddenJoints(
      f.points.map((p) => ({ ...p })),
      hidden,
    ),
  }));
}

function solve(
  sim: Sim3DResult,
  frames: PoseFrame[],
  extra: BoatCalibration = {},
): CameraSolve {
  return solveCamera(
    frames,
    EIGHT,
    {
      seatSpacing: sim.dimensions.seatSpacing,
      focalPx: sim.camera.focal,
      axisVerified: true,
      level: true,
      ...extra,
    },
    IMAGE,
  );
}

/** The fixture's own 3D joints as a lift would return them: camera frame, root
 * relative at the hip midpoint, metres — unless `warp` breaks them. */
function oracle(
  row: Sim3DTruth,
  azimuth: number,
  warp: (p: Vec3) => Vec3 = (p) => p,
): LiftOutput {
  const cam = (n: Joint3DName) => rotateY(row.joints[n], azimuth);
  const l = cam("hipL"),
    r = cam("hipR");
  const root = { x: (l.x + r.x) / 2, y: (l.y + r.y) / 2, z: (l.z + r.z) / 2 };
  const joints = new Float32Array(LIFT_BODY_JOINTS * 3);
  const confidence = new Float32Array(LIFT_BODY_JOINTS).fill(0.9);
  for (const n of JOINT3D_NAMES) {
    const i = JOINT3D[n],
      c = cam(n),
      rel = warp({ x: c.x - root.x, y: c.y - root.y, z: c.z - root.z });
    joints[i * 3] = rel.x;
    joints[i * 3 + 1] = rel.y;
    joints[i * 3 + 2] = rel.z;
  }
  return { joints, confidence, jointCount: LIFT_BODY_JOINTS };
}

/** A clip, its prepared frames, its solve and one frame's truth, together. */
function scene(azimuth = 24, feet: "hidden" | "visible" = "hidden") {
  const sim = simulateCrew({ seconds: 8, camera: { azimuth }, feet });
  const frames = prepared(sim, feet === "hidden");
  return { sim, frames, solve: solve(sim, frames), index: 40 };
}

const row = (s: ReturnType<typeof scene>) => s.sim.truth[s.index]!;
const frame = (s: ReturnType<typeof scene>) => s.frames[s.index]!;

/** The boat-frame angle the fixture's truth implies, for comparison. */
function truthKnee(t: Sim3DTruth) {
  return angle3D(t.joints.hipL, t.joints.kneeL, t.joints.ankleL)!;
}
function truthLayback(t: Sim3DTruth) {
  const sx = (t.joints.shoulderL.x + t.joints.shoulderR.x) / 2,
    sy = (t.joints.shoulderL.y + t.joints.shoulderR.y) / 2,
    hx = (t.joints.hipL.x + t.joints.hipR.x) / 2,
    hy = (t.joints.hipL.y + t.joints.hipR.y) / 2;
  return (-Math.atan2(sx - hx, sy - hy) * 180) / Math.PI;
}
/** The same knee measured in the image, the way a 2D tool would. */
function imageKnee(f: PoseFrame) {
  const p = (i: number) => ({ x: f.points[i]!.x, y: f.points[i]!.y, z: 0 });
  return angle3D(p(11), p(13), p(15));
}

describe("placing an honest lift", () => {
  it("lands the lifted pose on the keypoints it was fitted to", () => {
    const s = scene();
    const result = placeLift(oracle(row(s), 24), frame(s), s.solve, IMAGE);
    expect(result.withheld).toBeNull();
    // The 2D keypoints carry about 1.5 px of noise; the placement should not add
    // much to that.
    expect(result.reprojection!).toBeLessThan(4);
    expect(result.scale!).toBeCloseTo(1, 1);
    expect(result.joints.length).toBe(LIFT_BODY_JOINTS);
    // Nothing lifted is ever an observation.
    for (const j of result.joints) if (j) expect(j.origin).toBe("inferred");
  });

  it("reproduces the boat-frame angles the fixture generated", () => {
    const s = scene(24, "visible");
    const result = placeLift(oracle(row(s), 24), frame(s), s.solve, IMAGE);
    const angles = jointAngles(result.joints);
    expect(result.withheld).toBeNull();
    expect(angles.kneeLeft.value!).toBeCloseTo(truthKnee(row(s)), 0);
    expect(angles.hipLeft.value!).not.toBeNull();
    expect(angles.layback.value!).toBeCloseTo(truthLayback(row(s)), 0);
    expect(angles.shinLeft.value!).not.toBeNull();
    expect(angles.thighLeft.value!).not.toBeNull();
    expect(angles.kneeLeft.confidence).toBeGreaterThan(0.5);
  });

  it("beats the same angle measured in the image, over a whole stroke", () => {
    // Measured over every lifted frame of one seat, not one hand-picked frame:
    // the error a 2D measurement makes depends on where in the stroke the leg
    // is, and quoting the best case would be the same sin as quoting the worst.
    //
    // The lifted column is exact because the oracle *is* the fixture's own 3D,
    // so what this measures is the pipeline — projection, solve, placement,
    // rotation into the boat frame, angle definition — and not a lift model's
    // accuracy. A real model has its own error on top, which is milestone 8's
    // gate and needs a benchmark, not a simulator.
    for (const azimuth of [10, 24, 45]) {
      const s = scene(azimuth, "visible");
      let n = 0,
        flatSum = 0,
        flatMax = 0,
        liftedMax = 0;
      for (let i = 0; i < s.frames.length; i++) {
        const f = s.frames[i]!,
          t = s.sim.truth[i]!;
        if (t.seat !== 1) continue;
        const result = placeLift(oracle(t, azimuth), f, s.solve, IMAGE);
        const lifted = result.withheld
          ? null
          : jointAngles(result.joints).kneeLeft.value;
        const flat = imageKnee(f);
        if (lifted === null || flat === null) continue;
        const truth = truthKnee(t);
        n++;
        flatSum += Math.abs(flat - truth);
        flatMax = Math.max(flatMax, Math.abs(flat - truth));
        liftedMax = Math.max(liftedMax, Math.abs(lifted - truth));
      }
      expect(n).toBeGreaterThan(150);
      // The pipeline reproduces the boat-frame angle to inside the keypoint
      // noise, at every azimuth...
      expect(liftedMax).toBeLessThan(0.5);
      // ...while the same angle read off the image is wrong by degrees, and gets
      // worse as the camera turns away from the beam. This is the whole reason
      // to solve the camera rather than to correct a 2D angle by its cosine.
      expect(flatSum / n).toBeGreaterThan(azimuth >= 45 ? 6 : 3);
      expect(flatMax).toBeGreaterThan(azimuth >= 45 ? 20 : 10);
    }
  });

  it("takes the lift's units as they come", () => {
    const s = scene();
    // A model that speaks millimetres: the placement must not care, because the
    // scale is fitted against the image, not assumed from the model.
    const mm = placeLift(
      oracle(row(s), 24, (p) => ({
        x: p.x * 1000,
        y: p.y * 1000,
        z: p.z * 1000,
      })),
      frame(s),
      s.solve,
      IMAGE,
    );
    expect(mm.withheld).toBeNull();
    expect(mm.scale!).toBeCloseTo(0.001, 4);
    expect(mm.reprojection!).toBeLessThan(4);
    const metres = placeLift(oracle(row(s), 24), frame(s), s.solve, IMAGE);
    expect(jointAngles(mm.joints).hipLeft.value!).toBeCloseTo(
      jointAngles(metres.joints).hipLeft.value!,
      1,
    );
  });

  it("leaves the angles alone when the calibration's distance is wrong", () => {
    const s = scene();
    // A tape measure that reads 30% long: the lift is placed 30% further out and
    // scaled up to match, and every angle — which is what is reported — is
    // unchanged. Metric lengths would not survive this, which is why they stay
    // withheld.
    const wrong: CameraSolve = {
      ...s.solve,
      distance: s.solve.distance! * 1.3,
      pxPerMetre: s.solve.pxPerMetre! / 1.3,
    };
    const a = jointAngles(
      placeLift(oracle(row(s), 24), frame(s), s.solve, IMAGE).joints,
    );
    const b = jointAngles(
      placeLift(oracle(row(s), 24), frame(s), wrong, IMAGE).joints,
    );
    expect(b.hipLeft.value!).toBeCloseTo(a.hipLeft.value!, 0);
    expect(b.layback.value!).toBeCloseTo(a.layback.value!, 0);
    expect(b.elbowLeft.value!).toBeCloseTo(a.elbowLeft.value!, 0);
  });
});

describe("refusing a lift it cannot believe", () => {
  it("refuses a lift whose vertical axis points the other way", () => {
    const s = scene();
    const r = placeLift(
      oracle(row(s), 24, (p) => ({ x: p.x, y: -p.y, z: p.z })),
      frame(s),
      s.solve,
      IMAGE,
    );
    expect(r.joints.length).toBe(0);
    expect(r.withheld).toMatch(/px from the keypoints/);
    expect(r.reprojection).toBeNull();
  });

  it("refuses a lift that is mirrored left to right", () => {
    const s = scene();
    const r = placeLift(
      oracle(row(s), 24, (p) => ({ x: -p.x, y: p.y, z: p.z })),
      frame(s),
      s.solve,
      IMAGE,
    );
    expect(r.withheld).not.toBeNull();
  });

  it("refuses a lift that puts the rower far off the shell's depth", () => {
    const s = scene();
    const r = placeLift(
      oracle(row(s), 24, (p) => ({ ...p, z: p.z + 20 })),
      frame(s),
      s.solve,
      IMAGE,
    );
    expect(r.withheld).toMatch(/px from the keypoints|cm/);
  });

  it("refuses an output that is not the 17-joint body layout", () => {
    const s = scene();
    const o = oracle(row(s), 24);
    const r = placeLift({ ...o, jointCount: 24 }, frame(s), s.solve, IMAGE);
    expect(r.withheld).toMatch(/24 joints/);
  });

  it("refuses when too few keypoints were seen to place a pose", () => {
    const s = scene();
    const f = frame(s);
    // Only five joints left: hips, knees and a shoulder.
    const keep = [5, 11, 12, 13, 14];
    const stripped: PoseFrame = {
      ...f,
      points: f.points.map((p, i) =>
        keep.includes(i) ? p : { ...p, origin: "missing", confidence: 0 },
      ),
    };
    const r = placeLift(oracle(row(s), 24), stripped, s.solve, IMAGE);
    expect(r.withheld).toMatch(/Only \d+ of the lifted joints/);
  });

  it("does not lift a joint the 2D pipeline discarded", () => {
    const s = scene(24, "visible");
    const f = frame(s);
    // A knee the gunwale hid near the finish: core/knees.ts rewrote it to
    // missing, and the lift model would happily return a number for it anyway.
    const occluded: PoseFrame = {
      ...f,
      points: f.points.map((p, i) =>
        i === 13 ? { ...p, origin: "missing", confidence: 0 } : p,
      ),
    };
    const r = placeLift(oracle(row(s), 24), occluded, s.solve, IMAGE);
    expect(r.withheld).toBeNull();
    expect(r.joints[13]).toBeUndefined();
    const angles = jointAngles(r.joints);
    // The knee, and everything through it, is withheld with the joint named...
    expect(angles.kneeLeft.value).toBeNull();
    expect(angles.kneeLeft.reason).toMatch(/left knee was not lifted/);
    expect(angles.hipLeft.value).toBeNull();
    expect(angles.shinLeft.value).toBeNull();
    // ...while the rest of the frame is still reported.
    expect(angles.kneeRight.value).not.toBeNull();
    expect(angles.layback.value).not.toBeNull();
    expect(angles.elbowLeft.value).not.toBeNull();
  });

  it("withholds on water, where the feet are below the gunwale", () => {
    const s = scene(24, "hidden");
    const r = placeLift(oracle(row(s), 24), frame(s), s.solve, IMAGE);
    const angles = jointAngles(r.joints);
    expect(r.withheld).toBeNull();
    // The ankle was discarded, so the knee and shin angles do not exist here,
    // and the reason names the joint rather than blaming the model.
    expect(angles.kneeLeft.value).toBeNull();
    expect(angles.kneeLeft.reason).toMatch(/left ankle was not lifted/);
    expect(angles.shinLeft.value).toBeNull();
    // What the launch can see is still measured, in the boat's frame.
    expect(angles.thighLeft.value).not.toBeNull();
    expect(angles.hipLeft.value).not.toBeNull();
    expect(angles.layback.value!).toBeCloseTo(truthLayback(row(s)), 0);
    expect(angles.elbowLeft.value).not.toBeNull();
  });

  it("lifts nothing at all when the camera is not solved", () => {
    const sim = simulateCrew({ seconds: 8, camera: { azimuth: 0 } });
    const frames = prepared(sim);
    const s = solve(sim, frames);
    expect(s.source).toBe("unresolved");
    const r = placeLift(oracle(sim.truth[40]!, 0), frames[40]!, s, IMAGE);
    expect(r.joints.length).toBe(0);
    // The solve's own reason is passed through: it is the honest one.
    expect(r.withheld).toBe(s.reason);
    expect(liftAvailable(s).ok).toBe(false);
    expect(liftAvailable(s).reason).toBe(s.reason);
  });
});

describe("what the image cannot check", () => {
  it("cannot see the sign of a nearly flat pose's depth axis", () => {
    // A rower is close to planar, and the plane is close to the image plane, so
    // flipping the lift's z reprojects almost as well: 6 px against a 2 px
    // honest fit. This is not a gap in the gate, it is the geometry — which is
    // why the depth-axis convention is a contract with the model, checked once
    // against a known-good release, and not something a clip can verify. The
    // interior angles survive the flip, because a reflection preserves them;
    // what it would corrupt is anything lateral, and that stays withheld.
    const s = scene();
    const flipped = placeLift(
      oracle(row(s), 24, (p) => ({ ...p, z: -p.z })),
      frame(s),
      s.solve,
      IMAGE,
    );
    const honest = placeLift(oracle(row(s), 24), frame(s), s.solve, IMAGE);
    expect(flipped.withheld).toBeNull();
    expect(flipped.reprojection!).toBeGreaterThan(honest.reprojection!);
    expect(flipped.reprojection!).toBeLessThan(10);
    expect(jointAngles(flipped.joints).hipLeft.value!).toBeCloseTo(
      jointAngles(honest.joints).hipLeft.value!,
      0,
    );
  });
});

describe("running a model", () => {
  it("runs the lift on a crop and places what came back", async () => {
    const s = scene();
    let calls = 0;
    const model = async () => {
      calls++;
      return oracle(row(s), 24);
    };
    const r = await liftFrame(
      model,
      { raster: new Float32Array(3 * 288 * 384), width: 288, height: 384 },
      frame(s),
      s.solve,
      IMAGE,
    );
    expect(calls).toBe(1);
    expect(r.withheld).toBeNull();
    expect(jointAngles(r.joints).layback.value!).toBeCloseTo(
      truthLayback(row(s)),
      0,
    );
  });

  it("does not run the model when there is no boat frame to lift into", () => {
    const sim = simulateCrew({ seconds: 8, camera: { azimuth: 0 } });
    const frames = prepared(sim);
    expect(liftAvailable(solve(sim, frames)).ok).toBe(false);
  });
});
