// Deterministic 3D rowing simulator for camera and lifting tests.
//
// `rowingSim.ts` projects a sagittal-plane rower with a single pixels-per-metre
// scale. That is enough to pin occlusion and cycle timing, but it cannot
// validate a camera solve: there is no depth, no perspective, and one seat, so
// nothing in its pixels carries the information an azimuth is recovered from.
//
// This fixture puts the same kinematics in boat-frame metres — x along the
// shell towards the rowers' feet, y up, z to a rower's right — gives the rower a
// left and a right side, seats a crew along the shell, and projects it through a
// pinhole camera at a known azimuth, focal length and distance. `truth` carries
// the noise-free boat-frame joints and the projection they came from, so a solve
// can be checked against what generated the pixels instead of against an
// assumption about them.
//
// Scope, stated plainly: like the 2D fixture this validates the *geometry
// implementation* — projection, the solve that inverts it, and the angle
// definitions built on top. It is not evidence that monocular depth recovery
// works on real footage, and ROADMAP.md says as much ("Synthetic projection
// tests validate the geometry implementation, not monocular depth recovery").

import type { Box, Point, PoseFrame, Vec3 } from "../../src/core/types";
import { missing } from "../../src/core/types";
import {
  armDraw,
  legExtension,
  rng,
  trunkAngle,
  type Rower,
} from "./rowingSim";
import { rotateY } from "../../src/core/geometry";

export type { Vec3 };

export interface SimCamera {
  /** Focal length in pixels. */
  focal: number;
  /**
   * Degrees the shell axis is turned away from parallel to the image plane. 0
   * is a perfect side view, where the image carries no information separating
   * azimuth from distance and a solve must abstain.
   */
  azimuth: number;
  /** Camera distance from the shell centreline, metres. */
  distance: number;
  /** Camera height above the ankle datum (about the waterline), metres. */
  height: number;
  /** Principal point, pixels. */
  cx: number;
  cy: number;
}

/** Lateral and arm dimensions the sagittal fixture has no use for. */
export interface Rower3D extends Rower {
  /** Hip joint separation, metres. */
  hipWidth: number;
  /** Shoulder joint separation, metres. */
  shoulderWidth: number;
  /** Ankle separation on the stretcher, metres. */
  footSpread: number;
  upperArm: number;
  forearm: number;
  /** Hand separation across the two handles of a scull, metres. */
  handleSpread: number;
}

export const defaultRower3D: Rower3D = {
  thigh: 0.45,
  shank: 0.45,
  trunk: 0.55,
  hipAboveAnkle: 0.19,
  catchKneeAngle: 50,
  finishKneeAngle: 175,
  hipWidth: 0.32,
  shoulderWidth: 0.42,
  footSpread: 0.16,
  // Almost straight at full reach, which is what the sagittal pose asks for:
  // the handle sits 0.60 m from the shoulder there and 0.41 m at the body.
  upperArm: 0.32,
  forearm: 0.3,
  handleSpread: 0.55,
};

/** COCO-WholeBody indices this fixture produces, by name. */
export const JOINT3D = {
  nose: 0,
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

export type Joint3DName = keyof typeof JOINT3D;
export const JOINT3D_NAMES = Object.keys(JOINT3D) as Joint3DName[];

/** One rower's joints in boat-frame metres, ankle datum at the origin. */
export type Pose3D = Record<Joint3DName, Vec3>;

const rad = (deg: number) => (deg * Math.PI) / 180;
const sub = (a: Vec3, b: Vec3): Vec3 => ({
  x: a.x - b.x,
  y: a.y - b.y,
  z: a.z - b.z,
});
const norm = (a: Vec3) => Math.hypot(a.x, a.y, a.z);
const cross = (a: Vec3, b: Vec3): Vec3 => ({
  x: a.y * b.z - a.z * b.y,
  y: a.z * b.x - a.x * b.z,
  z: a.x * b.y - a.y * b.x,
});

/**
 * The elbow of a two-link arm from shoulder to wrist, in the vertical plane
 * through both and below that line: elbows row low and behind the hands. Null
 * when the wrist is out of reach, which the caller reports rather than papers
 * over by straightening the arm.
 */
export function elbowBetween(
  shoulder: Vec3,
  wrist: Vec3,
  upperArm: number,
  forearm: number,
): Vec3 | null {
  const d = sub(wrist, shoulder),
    len = norm(d);
  if (len < 1e-6 || len > upperArm + forearm) return null;
  const u = { x: d.x / len, y: d.y / len, z: d.z / len };
  let n = cross(cross(u, { x: 0, y: 1, z: 0 }), u);
  const nl = norm(n);
  if (nl < 1e-6) return null;
  n = { x: n.x / nl, y: n.y / nl, z: n.z / nl };
  const a = (upperArm ** 2 - forearm ** 2 + len ** 2) / (2 * len),
    h = Math.sqrt(Math.max(0, upperArm ** 2 - a * a)),
    p = {
      x: shoulder.x + a * u.x,
      y: shoulder.y + a * u.y,
      z: shoulder.z + a * u.z,
    };
  return n.y <= 0
    ? { x: p.x + h * n.x, y: p.y + h * n.y, z: p.z + h * n.z }
    : { x: p.x - h * n.x, y: p.y - h * n.y, z: p.z - h * n.z };
}

/**
 * Boat-frame joints of one rower at one phase, ankle datum at the origin.
 *
 * The sagittal centreline follows the 2D fixture's kinematics — the seat slides
 * on the rail, the knee is the upper intersection of the thigh and shank
 * circles, trunk swing and arm draw follow the legs — and each joint is then
 * placed at its lateral offset, with the arms solved as real two-link chains
 * rather than the 2D fixture's midpoint heuristic.
 */
export function rowerPose3D(
  r: Rower3D,
  phase: number,
  discipline: "sweep" | "scull" = "sweep",
): Pose3D {
  const e = legExtension(phase);
  const phi = rad(
    r.catchKneeAngle + (r.finishKneeAngle - r.catchKneeAngle) * e,
  );
  const d = Math.sqrt(
    r.thigh ** 2 + r.shank ** 2 - 2 * r.thigh * r.shank * Math.cos(phi),
  );
  const hipY = r.hipAboveAnkle,
    hipX = -Math.sqrt(Math.max(0, d * d - hipY ** 2));
  const ux = -hipX / d,
    uy = -hipY / d;
  const a = (r.thigh ** 2 - r.shank ** 2 + d * d) / (2 * d),
    h = Math.sqrt(Math.max(0, r.thigh ** 2 - a * a)),
    px = hipX + a * ux,
    py = hipY + a * uy;
  const k1 = { x: px - h * uy, y: py + h * ux },
    k2 = { x: px + h * uy, y: py - h * ux };
  const knee = k1.y > k2.y ? k1 : k2; // knees bend upwards in a shell
  const tau = rad(trunkAngle(phase));
  const shoulderX = hipX - r.trunk * Math.sin(tau),
    shoulderY = hipY + r.trunk * Math.cos(tau);
  const m = armDraw(phase);
  const reach = { x: shoulderX + 0.58, y: shoulderY - 0.16 },
    drawn = { x: hipX + 0.14, y: hipY + 0.3 };
  const handleX = reach.x + (drawn.x - reach.x) * m,
    handleY = reach.y + (drawn.y - reach.y) * m;
  // Sweep hands share one handle; sculling hands are apart on two.
  const handZ = discipline === "sweep" ? 0.09 : r.handleSpread / 2;
  const at = (x: number, y: number, z: number): Vec3 => ({ x, y, z });

  const pose: Pose3D = {
    nose: at(
      shoulderX + 0.1 * Math.cos(tau) - 0.2 * Math.sin(tau),
      shoulderY + 0.2 * Math.cos(tau) + 0.1 * Math.sin(tau),
      0,
    ),
    shoulderL: at(shoulderX, shoulderY, -r.shoulderWidth / 2),
    shoulderR: at(shoulderX, shoulderY, r.shoulderWidth / 2),
    elbowL: at(shoulderX, shoulderY, -r.shoulderWidth / 2),
    elbowR: at(shoulderX, shoulderY, r.shoulderWidth / 2),
    wristL: at(handleX, handleY, -handZ),
    wristR: at(handleX, handleY, handZ),
    hipL: at(hipX, hipY, -r.hipWidth / 2),
    hipR: at(hipX, hipY, r.hipWidth / 2),
    kneeL: at(knee.x, knee.y, -r.footSpread / 2),
    kneeR: at(knee.x, knee.y, r.footSpread / 2),
    ankleL: at(0, 0, -r.footSpread / 2),
    ankleR: at(0, 0, r.footSpread / 2),
  };
  for (const side of ["L", "R"] as const) {
    const elbow = elbowBetween(
      pose[`shoulder${side}`],
      pose[`wrist${side}`],
      r.upperArm,
      r.forearm,
    );
    // An unreachable wrist is a fixture bug, not something to hide: leaving the
    // elbow on the shoulder makes the test that catches it fail loudly.
    if (elbow) pose[`elbow${side}`] = elbow;
  }
  return pose;
}

export interface Sim3DOptions {
  seed?: number;
  seconds?: number;
  fps?: number;
  /** Strokes per minute, shared by the crew; `spread` sets the lag. */
  rate?: number;
  /** Time of the first catch, seconds. */
  offset?: number;
  /** Seats in the shell and their spacing along it, metres. */
  seats?: number;
  seatSpacing?: number;
  /** Shell length bow to stern, metres (reported, not drawn). */
  shellLength?: number;
  discipline?: "sweep" | "scull";
  rower?: Partial<Rower3D>;
  camera?: Partial<SimCamera>;
  /** Keypoint noise σ as a fraction of detection-box height. */
  noise?: number;
  /** Per-frame probability that a knee is briefly dropped. */
  kneeDropout?: number;
  /**
   * Feet are strapped below the gunwale, so by default they are not visible and
   * whatever the model returns for them is a guess; `visible` models an erg.
   */
  feet?: "hidden" | "visible";
  /** Image direction the rowers' feet point to: the bow is the other way. */
  feetToRight?: boolean;
  /** Seat-to-seat catch lag in cycles (crew timing); positive = higher seats later. */
  spread?: number;
  /** Boat surge along the shell, metres, stroke-locked. */
  surge?: number;
  /**
   * Degrees the camera turns over the whole clip: a launch following the crew
   * does not hold one azimuth. `camera.azimuth` is then the azimuth at t = 0 and
   * the truth's depths follow the turn.
   */
  azimuthDrift?: number;
  /** Image size the crew is projected into. */
  width?: number;
  height?: number;
}

export interface Sim3DTruth {
  time: number;
  phase: number;
  seat: number;
  /** Leg extension: 0 at the catch, 1 once the legs are flat. */
  extension: number;
  /** Boat-frame metres, shell-fixed, including the seat's place along it. */
  joints: Pose3D;
  /** The same joints as projected, noise free, in pixels. */
  image: Record<Joint3DName, { x: number; y: number }>;
  /**
   * Camera-frame depth of the seat's hip midpoint, metres: what a seat-line
   * solve reports, and so what it must be checked against.
   */
  depth: number;
}

export interface Sim3DResult {
  frames: PoseFrame[];
  /** Per-frame ground truth, same length and order as `frames`. */
  truth: Sim3DTruth[];
  /** True catch times of the bow seat. */
  catches: number[];
  period: number;
  camera: SimCamera;
  /** The dimensions a calibration has to supply to solve this clip. */
  dimensions: { seats: number; seatSpacing: number; shellLength: number };
}

/**
 * Projects one boat-frame point to pixels.
 *
 * The camera is aimed at the shell's centre from `distance` metres off it,
 * `height` above the ankle datum, turned `azimuth` away from a perfect side
 * view, so that centre lands on the principal point. Camera coordinates are
 * exactly core/geometry.ts's `rotateY` — the same transform `toBoatFrame`
 * inverts — so a lift put into the boat frame with those helpers lands where
 * this fixture put it, and the sign convention is one convention rather than
 * two: **positive azimuth means the end of the shell towards the rowers' feet,
 * the stern, is the nearer end**, and the seat spacing therefore widens towards
 * it. That depth gradient along a line of known spacing is the whole of the
 * monocular cue this fixture exists to test; at azimuth 0 there is none.
 */
export function project(
  p: Vec3,
  cam: SimCamera,
  flip = 1,
): { x: number; y: number; depth: number } {
  const q = rotateY({ x: flip * p.x, y: p.y, z: p.z }, cam.azimuth),
    X = q.x,
    Y = q.y - cam.height,
    Z = q.z + cam.distance;
  return {
    x: cam.cx + (cam.focal * X) / Z,
    y: cam.cy - (cam.focal * Y) / Z,
    depth: Z,
  };
}

export function simulateCrew(options: Sim3DOptions = {}): Sim3DResult {
  const o = {
    seed: 1,
    seconds: 20,
    fps: 30,
    rate: 24,
    offset: 0.4,
    seats: 8,
    seatSpacing: 0.9,
    shellLength: 17.5,
    discipline: "sweep" as const,
    noise: 0.012,
    kneeDropout: 0.03,
    feet: "hidden" as const,
    feetToRight: true,
    spread: 0,
    surge: 0.08,
    azimuthDrift: 0,
    width: 1280,
    height: 720,
    ...options,
  };
  const r: Rower3D = { ...defaultRower3D, ...o.rower };
  const cam: SimCamera = {
    focal: 1800,
    azimuth: 18,
    distance: 16,
    height: 1.6,
    cx: o.width / 2,
    cy: o.height * 0.58,
    ...o.camera,
  };
  const R = rng(o.seed);
  // The knees draw from their own stream, as in the 2D fixture, so changing
  // what the model does with them never perturbs the body's noise.
  const K = rng(o.seed ^ 0x51ed27);
  const flip = o.feetToRight ? 1 : -1;
  const period = 60 / o.rate;
  const frames: PoseFrame[] = [];
  const truth: Sim3DTruth[] = [];
  const catches: number[] = [];
  for (let t = o.offset; t < o.seconds; t += period) catches.push(t);

  for (let i = 0; i < o.seconds * o.fps; i++) {
    const time = i / o.fps;
    // A launch follows the crew, so the camera can turn during the clip.
    const view: SimCamera = o.azimuthDrift
      ? { ...cam, azimuth: cam.azimuth + (o.azimuthDrift * time) / o.seconds }
      : cam;
    for (let s = 0; s < o.seats; s++) {
      const seat = s + 1;
      // Seat 1 is the bow seat: the bow is away from the rowers' feet, which
      // this frame calls +x, so the seats run from negative x to positive.
      const along = (s - (o.seats - 1) / 2) * o.seatSpacing;
      const phase =
        (((((time - o.offset) / period) % 1) + 1) % 1) + o.spread * s;
      const pose = rowerPose3D(r, phase % 1, o.discipline);
      const surge = o.surge * Math.sin(2 * Math.PI * (phase % 1) + 0.6);
      const joints = {} as Pose3D;
      const image = {} as Record<Joint3DName, { x: number; y: number }>;
      let hipL = view.distance,
        hipR = view.distance;
      for (const name of JOINT3D_NAMES) {
        const p = pose[name];
        joints[name] = { x: p.x + along + surge, y: p.y, z: p.z };
        const q = project(joints[name], view, flip);
        image[name] = { x: q.x, y: q.y };
        if (name === "hipL") hipL = q.depth;
        if (name === "hipR") hipR = q.depth;
      }
      // The seat line a solve reads is the midpoint of the two hips, so that is
      // the depth it should recover: averaging the pair cancels their 16 cm
      // lateral offset, which would otherwise lean into the depth by cos(az).
      const depth = (hipL + hipR) / 2;
      truth.push({
        time,
        phase: phase % 1,
        seat,
        extension: legExtension(phase % 1),
        joints,
        image,
        depth,
      });

      const points: Point[] = Array.from({ length: 133 }, () => missing());
      for (const name of JOINT3D_NAMES) {
        // Feet strapped below the gunwale are not visible, so the model's
        // answer for them is a guess placed under the hull.
        if (name.startsWith("ankle") && o.feet === "hidden") continue;
        const q = image[name];
        points[JOINT3D[name]] = {
          x: q.x,
          y: q.y,
          confidence: 0.55 + 0.4 * R.next(),
          origin: "observed",
        };
      }
      if (o.feet === "hidden")
        for (const j of [15, 16]) {
          const ankle = image[j === 15 ? "ankleL" : "ankleR"];
          points[j] = {
            x: ankle.x + R.gauss() * 14,
            y: ankle.y + 46 + R.gauss() * 14,
            confidence: 0.31 + 0.35 * R.next(),
            origin: "observed",
          };
        }

      const xs: number[] = [],
        ys: number[] = [];
      for (const name of JOINT3D_NAMES) {
        if (name.startsWith("ankle") && o.feet === "hidden") continue;
        xs.push(image[name].x);
        ys.push(image[name].y);
      }
      const box: Box = {
        x: Math.min(...xs) - 24,
        y: Math.min(...ys) - 36,
        width: Math.max(...xs) - Math.min(...xs) + 48,
        height: Math.max(...ys) - Math.min(...ys) + 60,
        confidence: 0.7 + 0.25 * R.next(),
      };
      // Keypoint noise scaled to the box, as a detector's is.
      for (const name of JOINT3D_NAMES) {
        const j = JOINT3D[name],
          p = points[j];
        if (p.origin === "missing") continue;
        points[j] = {
          ...p,
          x: p.x + R.gauss() * o.noise * box.height,
          y: p.y + R.gauss() * o.noise * box.height,
        };
      }
      if (o.kneeDropout)
        for (const j of [13, 14])
          if (points[j].origin === "observed" && K.next() < o.kneeDropout)
            points[j] = {
              ...points[j],
              confidence: 0.05 + 0.2 * K.next(),
              origin: "missing",
            };
      frames.push({
        time,
        seat,
        points,
        box,
        identityConfidence: 0.9,
        boxSource: "detected",
        detectorAge: 0,
      });
    }
  }
  return {
    frames,
    truth,
    catches,
    period,
    camera: cam,
    dimensions: {
      seats: o.seats,
      seatSpacing: o.seatSpacing,
      shellLength: o.shellLength,
    },
  };
}
