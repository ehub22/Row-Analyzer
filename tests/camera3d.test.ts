import { describe, it, expect } from "vitest";
import {
  solveCameraNormalization,
  toSagittalPlane,
  bodyAnglesFromPose,
  estimateReferenceLengths,
  vanishingPoint,
  focalFromOrthogonalVanishingPoints,
  lineThrough,
  intersect,
  type Line,
  type Pt2,
} from "../src/core/camera3d";
import { angle3D } from "../src/core/geometry";
import type { Point, Vec3 } from "../src/core/types";
import {
  detectHorizon,
  lockHorizon,
  type GrayImage,
} from "../src/core/horizon";

// ---- synthetic rig: known 3D world rendered through a known pinhole camera ----

type V = [number, number, number];
const dot = (a: V, b: V) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: V, b: V): V => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const unit = (a: V): V => {
  const n = Math.hypot(...a);
  return [a[0] / n, a[1] / n, a[2] / n];
};
const rad = (d: number) => (d * Math.PI) / 180;

const W = 1280,
  H = 720,
  F = 1000,
  CX = W / 2,
  CY = H / 2;

interface Rig {
  up: V;
  fore: V;
  lat: V;
}

/**
 * Camera orientation from roll (about the optical axis), elevation (pitch),
 * and azimuth (0 = shell across the view, 90 = bow/stern-on). The shell's
 * fore direction is built from the azimuth, not from the solver's formulas.
 */
function rig(rollDeg: number, elevDeg: number, azDeg: number): Rig {
  const r = rad(rollDeg),
    e = rad(elevDeg);
  const up: V = [
    Math.sin(r) * Math.cos(e),
    -Math.cos(r) * Math.cos(e),
    Math.sin(e),
  ];
  const z: V = [0, 0, 1];
  const fh = unit([
    z[0] - dot(z, up) * up[0],
    z[1] - dot(z, up) * up[1],
    z[2] - dot(z, up) * up[2],
  ]);
  const lat = unit(cross(up, fh));
  const c = rad(90 - azDeg);
  const fore = unit(
    [0, 1, 2].map((i) => Math.cos(c) * fh[i] + Math.sin(c) * lat[i]) as V,
  );
  const lateral = unit(cross(fore, up));
  return { up, fore, lat: lateral };
}

/** Boat coordinates (x bow-ward, y up, z lateral) → camera coordinates at distance D. */
const EYE_HEIGHT = 1.5; // metres the water (and the shell) sits below the lens
const toCam = (R: Rig, p: V, D: number): V => [
  p[0] * R.fore[0] + p[1] * R.up[0] + p[2] * R.lat[0] - EYE_HEIGHT * R.up[0],
  p[0] * R.fore[1] + p[1] * R.up[1] + p[2] * R.lat[1] - EYE_HEIGHT * R.up[1],
  p[0] * R.fore[2] +
    p[1] * R.up[2] +
    p[2] * R.lat[2] +
    D -
    EYE_HEIGHT * R.up[2],
];
const project = (c: V): Pt2 => ({
  x: (F * c[0]) / c[2] + CX,
  y: (F * c[1]) / c[2] + CY,
});

/** Image of the world horizon: line through the vanishing points of two horizontal directions. */
function horizonLine(R: Rig): Line {
  const z: V = [0, 0, 1];
  const fh = unit([
    z[0] - dot(z, R.up) * R.up[0],
    z[1] - dot(z, R.up) * R.up[1],
    z[2] - dot(z, R.up) * R.up[2],
  ]);
  const lat = unit(cross(R.up, fh));
  const vp = (h: V): Pt2 => ({
    x: (F * h[0]) / h[2] + CX,
    y: (F * h[1]) / h[2] + CY,
  });
  const h1 = [0, 1, 2].map((i) => fh[i] + 0.3 * lat[i]) as V;
  const h2 = [0, 1, 2].map((i) => fh[i] - 0.3 * lat[i]) as V;
  const a = vp(h1),
    b = vp(h2);
  const l = lineThrough(a, b)!;
  return l;
}

/** A rower in the sagittal plane (lateral offset `lat` metres), with known joint angles. */
function skeleton(lateral = 0) {
  const P = (x: number, y: number): V => [x, y, lateral];
  const hip = P(0, 0);
  const trunkLean = 35; // degrees toward the bow from vertical
  const shoulder = P(
    0.55 * Math.sin(rad(trunkLean)),
    0.55 * Math.cos(rad(trunkLean)),
  );
  const thighDir = 15; // hip→knee above bow-ward horizontal
  const knee = P(
    0.45 * Math.cos(rad(thighDir)),
    0.45 * Math.sin(rad(thighDir)),
  );
  const shinDir = -25;
  const ankle = P(
    knee[0] + 0.42 * Math.cos(rad(shinDir)),
    knee[1] + 0.42 * Math.sin(rad(shinDir)),
  );
  const elbow = P(
    shoulder[0] + 0.3 * Math.cos(rad(-60)),
    shoulder[1] + 0.3 * Math.sin(rad(-60)),
  );
  const wrist = P(
    elbow[0] + 0.25 * Math.cos(rad(-20)),
    elbow[1] + 0.25 * Math.sin(rad(-20)),
  );
  return { hip, shoulder, knee, ankle, elbow, wrist };
}

const v3 = (p: V): Vec3 => ({ x: p[0], y: p[1], z: p[2] });

function poseFromSkeleton(
  R: Rig,
  s: ReturnType<typeof skeleton>,
  D: number,
): Point[] {
  const pts: Point[] = Array.from({ length: 17 }, () => ({
    x: 0,
    y: 0,
    confidence: 0,
    origin: "missing",
  }));
  // Left side (COCO 5,7,9,11,13,15) only; right side left missing.
  const put = (i: number, v: V) => {
    const p = project(toCam(R, v, D));
    pts[i] = { x: p.x, y: p.y, confidence: 0.9, origin: "observed" };
  };
  put(5, s.shoulder);
  put(7, s.elbow);
  put(9, s.wrist);
  put(11, s.hip);
  put(13, s.knee);
  put(15, s.ankle);
  return pts;
}

function solveFor(
  rollDeg: number,
  elevDeg: number,
  azDeg: number,
  opts: { lateral?: number } = {},
) {
  const D = 10;
  const R = rig(rollDeg, elevDeg, azDeg);
  const gunwale = 0.5;
  const bow = project(toCam(R, [2.4, 0, gunwale], D));
  const stern = project(toCam(R, [-2.4, 0, gunwale], D));
  const solution = solveCameraNormalization({
    width: W,
    height: H,
    horizon: horizonLine(R),
    bow,
    stern,
    focalPx: F,
  });
  const s = skeleton(opts.lateral ?? 0);
  const pose = poseFromSkeleton(R, s, D);
  return { R, solution, pose, s, D, bow, stern };
}

describe("camera normalization from horizon + shell axis", () => {
  it.each([
    [0, 0, 0],
    [0, 0, 10],
    [0, 0, 25],
    [0, 0, 40],
    [0, 0, 55],
    [5, 0, 20],
    [-8, 0, 30],
    [0, 6, 15],
    [0, -6, 15],
    [7, -5, 45],
  ])("recovers roll %s, elevation %s, azimuth %s", (roll, elev, az) => {
    const { solution } = solveFor(roll, elev, az);
    expect(solution.reasons.some((r) => /unknown/i.test(r))).toBe(false);
    expect(solution.rollDeg).toBeCloseTo(roll, 4);
    expect(solution.pitchDeg).toBeCloseTo(elev, 4);
    expect(solution.azimuthDeg).toBeCloseTo(az, 4);
    expect(solution.usable).toBe(true);
  });

  it("roll needs no focal length", () => {
    const R = rig(4, 0, 20);
    const sol = solveCameraNormalization({
      width: W,
      height: H,
      horizon: horizonLine(R),
      bow: project(toCam(R, [2.4, 0, 0.5], 10)),
      stern: project(toCam(R, [-2.4, 0, 0.5], 10)),
    });
    expect(sol.usable).toBe(false);
    expect(sol.rollDeg).toBeCloseTo(4, 4);
    expect(sol.azimuthDeg).toBeNull();
    expect(sol.reasons.join(" ")).toMatch(/Focal length unknown/);
  });

  it("body angles from a side-on view match the true joint angles", () => {
    const { solution, pose, s } = solveFor(0, 0, 0);
    const out = bodyAnglesFromPose(pose, solution);
    expect(out.status).toBe("measured");
    expect(out.side).toBe("left");
    const a = out.angles;
    expect(a.trunkFromVerticalDeg).toBeCloseTo(35, 3);
    expect(a.thighFromHorizontalDeg).toBeCloseTo(15, 3);
    expect(a.shinFromHorizontalDeg).toBeCloseTo(-25, 3);
    const knee3d = angle3D(v3(s.hip), v3(s.knee), v3(s.ankle))!;
    const hip3d = angle3D(v3(s.shoulder), v3(s.hip), v3(s.knee))!;
    const elbow3d = angle3D(v3(s.shoulder), v3(s.elbow), v3(s.wrist))!;
    expect(a.kneeDeg).toBeCloseTo(knee3d, 3);
    expect(a.hipDeg).toBeCloseTo(hip3d, 3);
    expect(a.elbowDeg).toBeCloseTo(elbow3d, 3);
  });

  it.each([10, 25, 40])(
    "oblique view (azimuth %s) keeps body angles within 1° when the rower is on the sagittal plane",
    (az) => {
      const { solution, pose, s } = solveFor(3, -2, az);
      const out = bodyAnglesFromPose(pose, solution);
      expect(out.status).toBe("measured");
      expect(out.angles.trunkFromVerticalDeg!).toBeCloseTo(35, 0);
      expect(out.angles.kneeDeg!).toBeCloseTo(
        angle3D(v3(s.hip), v3(s.knee), v3(s.ankle))!,
        0,
      );
    },
  );

  it("withholds body angles at bow/stern-on azimuths", () => {
    const { solution, pose } = solveFor(0, 0, 70);
    expect(solution.usable).toBe(false);
    const out = bodyAnglesFromPose(pose, solution);
    expect(out.status).toBe("withheld");
    expect(out.angles.trunkFromVerticalDeg).toBeNull();
    expect(out.reasons.join(" ")).toMatch(/oblique|bow\/stern-on|Azimuth/);
  });

  it("the bow/stern order sets the fore direction: swapping the points flips the trunk sign", () => {
    const R = rig(0, 0, 0);
    const bow = project(toCam(R, [2.4, 0, 0.5], 10));
    const stern = project(toCam(R, [-2.4, 0, 0.5], 10));
    const hz = horizonLine(R);
    const good = solveCameraNormalization({
      width: W,
      height: H,
      horizon: hz,
      bow,
      stern,
      focalPx: F,
    });
    const swapped = solveCameraNormalization({
      width: W,
      height: H,
      horizon: hz,
      bow: stern,
      stern: bow,
      focalPx: F,
    });
    const pose = poseFromSkeleton(R, skeleton(0), 10);
    expect(
      bodyAnglesFromPose(pose, good).angles.trunkFromVerticalDeg!,
    ).toBeCloseTo(35, 3);
    expect(
      bodyAnglesFromPose(pose, swapped).angles.trunkFromVerticalDeg!,
    ).toBeCloseTo(-35, 3);
  });

  it("does not use inferred or missing joints", () => {
    const { solution, pose } = solveFor(0, 0, 0);
    const inferred = pose.map((p) => ({ ...p, origin: "inferred" as const }));
    const out = bodyAnglesFromPose(inferred, solution);
    expect(out.status).toBe("withheld");
  });

  it("reports the plane-normalized limits instead of claiming validated angles", () => {
    const { solution, pose } = solveFor(0, 0, 0);
    const out = bodyAnglesFromPose(pose, solution);
    expect(out.reasons.join(" ")).toMatch(/not validated/);
  });

  it("gates on roll, pitch and horizon confidence", () => {
    const R = rig(0, 0, 20);
    const base = {
      width: W,
      height: H,
      bow: project(toCam(R, [2.4, 0, 0.5], 10)),
      stern: project(toCam(R, [-2.4, 0, 0.5], 10)),
      focalPx: F,
    };
    const rolled = solveCameraNormalization({
      ...base,
      horizon: horizonLine(rig(40, 0, 20)),
    });
    expect(rolled.usable).toBe(false);
    const weak = solveCameraNormalization({
      ...base,
      horizon: horizonLine(R),
      horizonConfidence: 0.1,
    });
    expect(weak.usable).toBe(false);
    expect(weak.reasons.join(" ")).toMatch(/Horizon confidence/);
  });

  it("refuses a bow point equal to the stern point", () => {
    const p = { x: 100, y: 100 };
    const sol = solveCameraNormalization({
      width: W,
      height: H,
      horizon: [0, 1, -360],
      bow: p,
      stern: p,
      focalPx: F,
    });
    expect(sol.usable).toBe(false);
  });

  it("refuses a focal length outside the physical range instead of solving", () => {
    const R = rig(0, 0, 20);
    const sol = solveCameraNormalization({
      width: W,
      height: H,
      horizon: horizonLine(R),
      bow: project(toCam(R, [2.4, 0, 0.5], 10)),
      stern: project(toCam(R, [-2.4, 0, 0.5], 10)),
      focalPx: 50,
    });
    expect(sol.usable).toBe(false);
    expect(sol.focalPx).toBeNull();
  });

  it("self-calibrates focal length from the shell axis and vertical lines", () => {
    const R = rig(2, 3, 25);
    // Vertical world lines: pairs of points on verticals, projected to the image.
    const verticals: Line[] = [-1.5, -0.5, 0.7].map((x) => {
      const top = project(toCam(R, [0, 1.2, x], 10));
      const bottom = project(toCam(R, [0, -1.2, x], 10));
      return lineThrough(top, bottom)!;
    });
    const sol = solveCameraNormalization({
      width: W,
      height: H,
      horizon: horizonLine(R),
      bow: project(toCam(R, [2.4, 0, 0.5], 10)),
      stern: project(toCam(R, [-2.4, 0, 0.5], 10)),
      verticalLines: verticals,
    });
    expect(sol.focalSource).toBe("vanishing-points");
    expect(sol.focalPx!).toBeCloseTo(F, -1);
    expect(sol.azimuthDeg!).toBeCloseTo(25, 0);
  });
});

/** Sky/water frame whose boundary is exactly `line`, with sensor noise and a moving blob. */
function renderHorizonFrame(
  line: Line,
  seed: number,
  blobX: number,
): GrayImage {
  const [a, b, c] = line;
  let s = seed;
  const rand = () => (s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32;
  const data = new Float32Array(W * H);
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      const yh = -(a * x + c) / b;
      let v = y < yh ? 0.82 : 0.32;
      if ((x - blobX) ** 2 + (y - 500) ** 2 < 30 * 30) v = 0.95;
      data[y * W + x] = v + (rand() - 0.5) * 0.03;
    }
  return { data, width: W, height: H };
}

describe("end to end: pixels → horizon → camera → body angles", () => {
  it("locks the horizon from a frame and recovers roll, pitch, azimuth and joint angles", () => {
    const R = rig(3, -2, 30);
    const truth = horizonLine(R);
    const frames = [0, 1, 2, 3].map((i) =>
      renderHorizonFrame(truth, 7 + i, 200 + i * 150),
    );
    const det = lockHorizon(frames);
    expect(det.accepted).toBe(true);
    // Horizon position error below ~1 px at image centre, and orientation below 0.2°.
    const tDeg = (Math.atan2(-truth[0], truth[1]) * 180) / Math.PI;
    expect(Math.abs(det.tiltDeg - tDeg)).toBeLessThan(0.2);
    expect(
      Math.abs(det.yAtCenter - -(truth[0] * (W / 2) + truth[2]) / truth[1]),
    ).toBeLessThan(1.5);

    const D = 10;
    const bow = project(toCam(R, [2.4, 0, 0.5], D));
    const stern = project(toCam(R, [-2.4, 0, 0.5], D));
    const sol = solveCameraNormalization({
      width: W,
      height: H,
      horizon: det.line!,
      horizonConfidence: Math.min(1, det.support / 0.3),
      bow,
      stern,
      focalPx: F,
    });
    expect(sol.rollDeg).toBeCloseTo(3, 0);
    expect(sol.pitchDeg!).toBeCloseTo(-2, 0);
    expect(sol.azimuthDeg!).toBeCloseTo(30, 0);
    const pose = poseFromSkeleton(R, skeleton(0), D);
    const out = bodyAnglesFromPose(pose, sol);
    expect(out.status).toBe("measured");
    expect(out.angles.trunkFromVerticalDeg!).toBeCloseTo(35, 0);
    expect(out.angles.shinFromHorizontalDeg!).toBeCloseTo(-25, 0);
  });
});

describe("out-of-plane arms: foreshortening gate", () => {
  // Same camera, same rower; only the arm position changes between frames.
  function frame(az: number, armOut: number) {
    const R = rig(2, 3, az);
    const D = 10;
    const bow = project(toCam(R, [2.4, 0, 0.5], D));
    const stern = project(toCam(R, [-2.4, 0, 0.5], D));
    const solution = solveCameraNormalization({
      width: W,
      height: H,
      horizon: horizonLine(R),
      bow,
      stern,
      focalPx: F,
    });
    const s = skeleton(0);
    const elbow: V = [s.elbow[0], s.elbow[1], armOut];
    const wrist: V = [s.wrist[0], s.wrist[1], armOut];
    const pose = poseFromSkeleton(R, { ...s, elbow, wrist }, D);
    return { solution, pose, s, elbow, wrist };
  }

  it("withholds the elbow when the upper arm is foreshortened, keeps the trunk and knee", () => {
    const az = 40;
    const inPlane = Array.from({ length: 10 }, () => frame(az, 0));
    const solution = inPlane[0].solution;
    const reference = estimateReferenceLengths(
      inPlane.map((f) => f.pose),
      solution,
    );
    expect(reference.upperArm).toBeGreaterThan(0);
    const bad = frame(az, 0.3);
    const out = bodyAnglesFromPose(bad.pose, solution, { reference });
    expect(out.foreshortened).toContain("upperArm");
    expect(out.angles.elbowDeg).toBeNull();
    expect(out.angles.trunkFromVerticalDeg).toBeCloseTo(35, 0);
    expect(out.angles.kneeDeg).not.toBeNull();
    expect(out.reasons.join(" ")).toMatch(/foreshortened/);
  });

  it("keeps the elbow for an in-plane arm, and the gate is off without references", () => {
    const f = frame(40, 0);
    const reference = estimateReferenceLengths([f.pose], f.solution);
    const withGate = bodyAnglesFromPose(f.pose, f.solution, { reference });
    expect(withGate.angles.elbowDeg).not.toBeNull();
    expect(withGate.foreshortened).toEqual([]);
    const noRef = bodyAnglesFromPose(frame(40, 0.3).pose, f.solution);
    expect(noRef.angles.elbowDeg).not.toBeNull(); // no reference, no gate (documented limitation)
  });

  it("needs at least five samples to estimate a reference", () => {
    const f = frame(0, 0);
    expect(estimateReferenceLengths([f.pose, f.pose], f.solution)).toEqual({});
  });
});

describe("vanishing-point utilities", () => {
  it("recovers a common vanishing point from noisy lines", () => {
    const vp = { x: 900, y: -300 };
    const lines: Line[] = [];
    for (let i = 0; i < 12; i++) {
      const p = { x: 100 + i * 40, y: 600 - i * 20 + (i % 3) };
      const l = lineThrough(p, vp)!;
      lines.push(l);
    }
    // One gross outlier must be rejected.
    lines.push(lineThrough({ x: 10, y: 10 }, { x: 40, y: 400 })!);
    const r = vanishingPoint(lines)!;
    expect(r.point!.x).toBeCloseTo(vp.x, 0);
    expect(r.point!.y).toBeCloseTo(vp.y, 0);
  });

  it("focal from orthogonal pair (Caprile–Torre) and a degenerate pair", () => {
    // Orthogonal directions with f = 700: (v1-p)·(v2-p) = -f².
    const p = { x: 640, y: 360 };
    const f = focalFromOrthogonalVanishingPoints(
      { x: 640 + 1000, y: 360 },
      { x: 640 - 490, y: 360 + 300 },
      p,
    );
    expect(f).toBeCloseTo(700, 6);
    expect(
      focalFromOrthogonalVanishingPoints(
        { x: 700, y: 360 },
        { x: 700, y: 400 },
        p,
      ),
    ).toBeNull();
  });

  it("intersect returns null for parallel lines", () => {
    expect(intersect([0, 1, -1], [0, 1, -5])).toBeNull();
    expect(intersect([0, 1, -1], [1, 0, -5])).toEqual({ x: 5, y: 1 });
  });

  it("sagittal mapping is scale-invariant for side-on views", () => {
    const { solution, s } = solveFor(0, 0, 0);
    const a = toSagittalPlane(
      solution,
      project(toCam(rig(0, 0, 0), s.hip, 10)),
    )!;
    const b = toSagittalPlane(
      solution,
      project(toCam(rig(0, 0, 0), s.shoulder, 10)),
    )!;
    // Unit-distance plane: lengths are scaled by 1/depth (D = 10 m here); angles are not.
    expect(Math.hypot(b.x - a.x, b.y - a.y)).toBeCloseTo(0.55 / 10, 4);
  });
});
