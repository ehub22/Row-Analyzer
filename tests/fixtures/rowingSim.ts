// Deterministic on-water rowing simulator for regression tests.
//
// Real rowing footage is not available to the test suite, so this generates
// what a whole-body pose model plausibly returns for a rower filmed from a
// launch: the body follows rowing kinematics, visible joints carry detector-
// like noise, and the feet — strapped to the stretcher below the gunwale —
// are *not visible*, so the model's ankle/foot outputs are hallucinations
// (or dropped; `feet: "visible"` models erg footage). Optionally the hull
// also hides the knees whenever they drop below the saxboard (`kneeGunwale`),
// and optionally the oar loom crosses them near the finish (`oarKnee`): both
// are what a rower filming from a launch actually sees at the finish.
// `truth` carries the noise-free hip and knee positions and which occluder
// covered the knee, so tests can measure keypoint error directly instead of
// only cycle timing.
//
// Kinematics (boat frame, metres, +x towards the feet/stern, +y up):
//  * the ankle is fixed to the foot stretcher (origin);
//  * the hip slides on a horizontal line a fixed height above the ankle;
//  * the knee is the upper intersection of the thigh circle around the hip
//    and the shank circle around the ankle (two-link leg);
//  * trunk swing and arm draw follow the usual legs → body → arms sequence.
//
// This is a fixture, not a validation: it encodes assumptions about how a
// model fails on hidden feet. It exists so that the pipeline's behaviour on
// that failure is pinned by tests instead of being rediscovered on footage.

import type { Box, Point, PoseFrame } from "../../src/core/types";
import { missing } from "../../src/core/types";

export function rng(seed: number) {
  let a = seed >>> 0;
  const next = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const gauss = () => {
    let u = 0;
    while (u === 0) u = next();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * next());
  };
  const uniform = (lo: number, hi: number) => lo + (hi - lo) * next();
  return { next, gauss, uniform };
}

interface V {
  x: number;
  y: number;
}

const ease = (x: number) =>
  (1 - Math.cos(Math.PI * Math.min(1, Math.max(0, x)))) / 2;
const rad = (deg: number) => (deg * Math.PI) / 180;

/** Leg extension: 0 at the catch, 1 once the legs are flat. */
export function legExtension(phase: number) {
  if (phase < 0.28) return ease(phase / 0.28); // leg drive
  if (phase < 0.55) return 1; // body swing, arm draw, hands away, body over
  return 1 - ease((phase - 0.55) / 0.45); // slide forward to the next catch
}
/** Trunk layback in degrees (negative = leaning towards the feet). */
export function trunkAngle(phase: number) {
  if (phase < 0.12) return -28;
  if (phase < 0.38) return -28 + 50 * ease((phase - 0.12) / 0.26);
  if (phase < 0.45) return 22;
  if (phase < 0.58) return 22 - 50 * ease((phase - 0.45) / 0.13);
  return -28;
}
/** Arm draw: 0 = arms straight, 1 = handle at the body. */
export function armDraw(phase: number) {
  if (phase < 0.26) return 0;
  if (phase < 0.4) return ease((phase - 0.26) / 0.14);
  if (phase < 0.42) return 1;
  if (phase < 0.5) return 1 - ease((phase - 0.42) / 0.08);
  return 0;
}

export interface Rower {
  thigh: number;
  shank: number;
  trunk: number;
  /** Hip joint height above the ankle (seat height over heel depth). */
  hipAboveAnkle: number;
  catchKneeAngle: number;
  finishKneeAngle: number;
}
export const defaultRower: Rower = {
  thigh: 0.45,
  shank: 0.45,
  trunk: 0.55,
  hipAboveAnkle: 0.19,
  catchKneeAngle: 50,
  finishKneeAngle: 175,
};

/** Boat-frame joints of one rower (ankle at the origin). */
export function rowerPose(r: Rower, phase: number) {
  const e = legExtension(phase);
  const phi = rad(
    r.catchKneeAngle + (r.finishKneeAngle - r.catchKneeAngle) * e,
  );
  const d = Math.sqrt(
    r.thigh ** 2 + r.shank ** 2 - 2 * r.thigh * r.shank * Math.cos(phi),
  );
  const hip: V = {
    x: -Math.sqrt(Math.max(0, d * d - r.hipAboveAnkle ** 2)),
    y: r.hipAboveAnkle,
  };
  const ankle: V = { x: 0, y: 0 };
  const ux = (ankle.x - hip.x) / d,
    uy = (ankle.y - hip.y) / d;
  const a = (r.thigh ** 2 - r.shank ** 2 + d * d) / (2 * d),
    h = Math.sqrt(Math.max(0, r.thigh ** 2 - a * a));
  const px = hip.x + a * ux,
    py = hip.y + a * uy;
  const k1 = { x: px - h * uy, y: py + h * ux },
    k2 = { x: px + h * uy, y: py - h * ux };
  const knee = k1.y > k2.y ? k1 : k2; // knees bend upwards in a shell
  const tau = rad(trunkAngle(phase));
  const shoulder = {
    x: hip.x - r.trunk * Math.sin(tau),
    y: hip.y + r.trunk * Math.cos(tau),
  };
  const nose = {
    x: shoulder.x + 0.1 * Math.cos(tau) - 0.2 * Math.sin(tau),
    y: shoulder.y + 0.2 * Math.cos(tau) + 0.1 * Math.sin(tau),
  };
  const reach = { x: shoulder.x + 0.58, y: shoulder.y - 0.16 };
  const drawn = { x: hip.x + 0.14, y: hip.y + 0.3 };
  const m = armDraw(phase);
  const wrist = {
    x: reach.x + (drawn.x - reach.x) * m,
    y: reach.y + (drawn.y - reach.y) * m,
  };
  const elbow = {
    x: (shoulder.x + wrist.x) / 2 - 0.12 * m,
    y: (shoulder.y + wrist.y) / 2 - 0.05 * m,
  };
  return { hip, knee, ankle, shoulder, nose, wrist, elbow, extension: e };
}

export type FeetModel =
  /** The model guesses hidden ankles/feet with fluctuating confidence. */
  | "hallucinate"
  /** The model returns hidden ankles/feet below threshold almost always. */
  | "dropout"
  /** Feet visible (ergometer / overhead view): ordinary noise. */
  | "visible";

export interface SimOptions {
  seed?: number;
  seconds?: number;
  fps?: number;
  /** Strokes per minute. */
  rate?: number;
  /** Time of the first catch, seconds. */
  offset?: number;
  /** Image direction the rower's feet point to. */
  feetToRight?: boolean;
  /** Pixels per metre. */
  scale?: number;
  /** Keypoint noise σ as a fraction of box height. */
  noise?: number;
  feet?: FeetModel;
  /** Launch drift amplitude in metres (slow, relative to the boat). */
  drift?: number;
  /** Boat surge relative to the launch in metres (stroke-locked). */
  surge?: number;
  /** Per-frame probability that a knee is briefly hidden (hands, handle). */
  kneeDropout?: number;
  /**
   * Gunwale height relative to the hip joint, in metres, when the hull also
   * hides the knees whenever they drop below it (legs flat). With a level
   * camera the saxboard is roughly at hip height, i.e. 0. Undefined: knees
   * are never hidden by the hull.
   */
  kneeGunwale?: number;
  /** What the model does with a knee behind the hull: drop it (confidence
   * below 0.3 every frame), place it on the gunwale edge with moderate
   * confidence, or answer a fresh coin flip between the two every frame — a
   * model does not treat the same covered knee the same way twice. */
  kneeHidden?: "dropout" | "snap" | "mixed";
  /**
   * The oar. Near the finish the loom crosses the legs in a side view, so the
   * knee is behind it for part of every cycle: this is the fraction of the
   * cycle it covers, centred on the finish (phase 0.45). Undefined/0: no oar
   * occlusion.
   */
  oarKnee?: number;
  /** What the model does with a knee behind the loom: drop it, report the knee
   * where the shaft is (the classic "latched onto the oar" answer), or a
   * per-frame coin flip between the two. */
  oarHidden?: "dropout" | "drag" | "mixed";
  seat?: number;
  rower?: Partial<Rower>;
}

/** What the rower was actually doing on one frame (fixture ground truth). */
export interface SimTruth {
  time: number;
  phase: number;
  /** Leg extension: 0 at the catch, 1 once the legs are flat. */
  extension: number;
  /** Image-space position of the near-side hip and knee, noise free. */
  hip: V;
  knee: V;
  /** The hull edge hid the knee on this frame. */
  hullHidKnee: boolean;
  /** The oar loom hid the knee on this frame. */
  oarHidKnee: boolean;
}

export interface SimResult {
  frames: PoseFrame[];
  /** True catch times (phase 0). */
  catches: number[];
  period: number;
  /** Per-frame ground truth, same length and order as `frames`. */
  truth: SimTruth[];
}

/** COCO-WholeBody: ankles 15/16, feet 17–22. */
const FEET = [15, 16, 17, 18, 19, 20, 21, 22];

export function simulateRower(options: SimOptions = {}): SimResult {
  const o = {
    seed: 1,
    seconds: 30,
    fps: 30,
    rate: 24,
    offset: 0.4,
    feetToRight: true,
    scale: 300,
    noise: 0.012,
    feet: "hallucinate" as FeetModel,
    drift: 0.3,
    surge: 0.08,
    kneeDropout: 0.03,
    seat: 1,
    ...options,
  };
  const r: Rower = { ...defaultRower, ...o.rower };
  const R = rng(o.seed);
  // Feet draw from their own stream so that changing what the model does
  // with them never perturbs the body's noise: the same seed gives the same
  // hips and knees under every feet model.
  const F = rng(o.seed ^ 0x9e3779b9);
  // Hull occlusion of the knees likewise has its own stream.
  const K = rng(o.seed ^ 0x51ed27);
  // …and so does the oar, so that adding it does not perturb the hull's draws.
  const O = rng(o.seed ^ 0x1b873593);
  const period = 60 / o.rate;
  const sign = o.feetToRight ? 1 : -1;
  const frames: PoseFrame[] = [];
  const truth: SimTruth[] = [];
  const catches: number[] = [];
  for (let t = o.offset; t < o.seconds; t += period) catches.push(t);
  // Temporally correlated hallucination state (models answer smoothly to
  // smoothly changing pixels, then jump when they latch onto something else).
  let guess = { angle: 0, length: 0.8, jx: 0, jy: 0, snap: 0 };
  for (let i = 0; i < o.seconds * o.fps; i++) {
    const time = i / o.fps;
    const phase = ((((time - o.offset) / period) % 1) + 1) % 1;
    const p = rowerPose(r, phase);
    const boat =
      o.drift * Math.sin((2 * Math.PI * time) / 23) +
      o.surge * Math.sin(2 * Math.PI * phase + 0.6);
    const toImage = (v: V): V => ({
      x: 640 + o.scale * (sign * v.x + boat),
      y: 520 - o.scale * v.y,
    });
    const img = {
      hip: toImage(p.hip),
      knee: toImage(p.knee),
      ankle: toImage(p.ankle),
      shoulder: toImage(p.shoulder),
      nose: toImage(p.nose),
      wrist: toImage(p.wrist),
      elbow: toImage(p.elbow),
    };
    // Detector box: head to the gunwale line (feet and shins below it are
    // hidden), as wide as the visible body.
    const gunwale = toImage({ x: 0, y: 0.06 }).y;
    const xs = [img.hip.x, img.knee.x, img.shoulder.x, img.nose.x, img.wrist.x];
    const top = img.nose.y - 0.12 * o.scale;
    const box: Box = {
      x: Math.min(...xs) - 0.08 * o.scale,
      y: top,
      width: Math.max(...xs) - Math.min(...xs) + 0.16 * o.scale,
      height: Math.max(gunwale, img.knee.y + 0.04 * o.scale) - top,
      confidence: 0.95,
    };
    const sd = o.noise * box.height;
    const points: Point[] = Array.from({ length: 133 }, missing);
    const put = (j: number, v: V, conf: number, extra = 1, G = R) => {
      points[j] = {
        x: v.x + G.gauss() * sd * extra,
        y: v.y + G.gauss() * sd * extra,
        confidence: conf,
        origin: conf >= 0.3 ? "observed" : "missing",
      };
    };
    // Near side (left) is clearer than the far side in a side view.
    const pair = (l: number, v: V, dx = 4, dy = 2) => {
      put(l, v, R.uniform(0.75, 0.95));
      put(l + 1, { x: v.x + dx, y: v.y + dy }, R.uniform(0.55, 0.9), 1.4);
    };
    put(0, img.nose, R.uniform(0.8, 0.95));
    put(
      1,
      { x: img.nose.x - sign * 8, y: img.nose.y - 8 },
      R.uniform(0.7, 0.9),
    );
    put(
      2,
      { x: img.nose.x - sign * 4, y: img.nose.y - 9 },
      R.uniform(0.4, 0.8),
    );
    put(
      3,
      { x: img.nose.x - sign * 30, y: img.nose.y - 4 },
      R.uniform(0.6, 0.9),
    );
    put(
      4,
      { x: img.nose.x - sign * 26, y: img.nose.y - 6 },
      R.uniform(0.3, 0.7),
    );
    pair(5, img.shoulder);
    pair(7, img.elbow);
    pair(9, img.wrist);
    pair(11, img.hip);
    pair(13, img.knee);
    for (const j of [13, 14])
      if (R.next() < o.kneeDropout) points[j] = missing();
    const hullHidKnee =
      o.kneeGunwale !== undefined && p.knee.y < r.hipAboveAnkle + o.kneeGunwale;
    if (hullHidKnee) {
      const edge = toImage({ x: 0, y: r.hipAboveAnkle + o.kneeGunwale! }).y;
      // One answer per frame, not per joint: both knees are behind the same
      // gunwale, and "mixed" re-decides every frame.
      const snap =
        o.kneeHidden === "snap" || (o.kneeHidden === "mixed" && K.next() < 0.5);
      for (const j of [13, 14]) {
        const conf = snap ? K.uniform(0.3, 0.6) : K.uniform(0.02, 0.28);
        points[j] = {
          x: img.knee.x + (j - 13) * 4 + K.gauss() * sd * 2,
          y: (snap ? edge : img.knee.y) + K.gauss() * sd * 2,
          confidence: conf,
          origin: conf >= 0.3 ? "observed" : "missing",
        };
      }
    }
    // The oar: near the finish the loom crosses the legs, so the knee is
    // behind it for a window of every cycle centred on phase 0.45.
    const oarHidKnee =
      !!o.oarKnee &&
      Math.abs(((phase - 0.45 + 1.5) % 1) - 0.5) <= o.oarKnee / 2;
    if (oarHidKnee) {
      // The shaft only covers the knee when it passes within a few centimetres
      // of it, so what the model reports is the knee pulled onto the shaft —
      // a few centimetres across it, plus a slide along it — not a joint
      // teleported to the handle. The shaft's image angle swinging with the
      // stroke, and the size of both displacements, are assumptions of this
      // fixture, not measurements of a boat.
      const theta = 38 + 14 * Math.sin(2 * Math.PI * phase);
      const a = rad(sign > 0 ? theta : 180 - theta);
      const ux = Math.cos(a),
        uy = Math.sin(a);
      // Normal to the shaft, pointing up in the image: the loom crosses over
      // the legs, so a knee reported on it sits above the real one.
      const nx = uy,
        ny = -ux;
      const dragged =
        o.oarHidden === "drag" || (o.oarHidden === "mixed" && O.next() < 0.5);
      for (const j of [13, 14]) {
        const across = O.uniform(0.02, 0.09) * o.scale,
          along = O.uniform(-0.05, 0.05) * o.scale,
          pull = O.uniform(0.5, 1);
        const conf = dragged ? O.uniform(0.35, 0.78) : O.uniform(0.02, 0.28);
        const drag = dragged ? pull : 0;
        points[j] = {
          x:
            img.knee.x +
            (across * nx + along * ux) * drag +
            (j - 13) * 4 +
            O.gauss() * sd * 1.6,
          y:
            img.knee.y +
            (across * ny + along * uy) * drag +
            O.gauss() * sd * 1.6,
          confidence: conf,
          origin: conf >= 0.3 ? "observed" : "missing",
        };
      }
    }
    if (o.feet === "visible") {
      put(15, img.ankle, F.uniform(0.75, 0.95), 1, F);
      put(
        16,
        { x: img.ankle.x + 4, y: img.ankle.y + 2 },
        F.uniform(0.55, 0.9),
        1.4,
        F,
      );
      FEET.slice(2).forEach((j, k) =>
        put(
          j,
          { x: img.ankle.x + sign * (k % 3) * 18, y: img.ankle.y + 12 },
          F.uniform(0.5, 0.85),
          1,
          F,
        ),
      );
    } else {
      // The model has no pixels for the feet. It extrapolates a shin from
      // the knee — typically straight down, like a seated person — with a
      // drifting angle and length, and every so often latches onto the
      // handle, rigger or hull edge instead.
      guess = {
        angle: 0.8 * guess.angle + F.gauss() * 9,
        length: Math.min(1.05, Math.max(0.45, guess.length + F.gauss() * 0.05)),
        jx: 0.7 * guess.jx + F.gauss() * 0.03,
        jy: 0.7 * guess.jy + F.gauss() * 0.02,
        snap: guess.snap > 0 ? guess.snap - 1 : F.next() < 0.04 ? 6 : 0,
      };
      const down = rad(guess.angle);
      const shin = guess.length * r.shank * o.scale;
      const fake: V =
        guess.snap > 0
          ? { x: img.wrist.x + F.gauss() * 10, y: img.wrist.y + 30 }
          : {
              x: img.knee.x + Math.sin(down) * shin + guess.jx * o.scale,
              y: img.knee.y + Math.cos(down) * shin + guess.jy * o.scale,
            };
      const conf = () =>
        o.feet === "dropout" ? F.uniform(0.02, 0.32) : F.uniform(0.1, 0.62);
      put(15, fake, conf(), 1.5, F);
      put(16, { x: fake.x + 6, y: fake.y + 3 }, conf(), 1.8, F);
      FEET.slice(2).forEach((j, k) =>
        put(
          j,
          { x: fake.x + sign * (k % 3) * 18, y: fake.y + 14 },
          conf() * 0.8,
          2,
          F,
        ),
      );
    }
    frames.push({ time, seat: o.seat, points, box, identityConfidence: 0.92 });
    truth.push({
      time,
      phase,
      extension: p.extension,
      hip: img.hip,
      knee: img.knee,
      hullHidKnee,
      oarHidKnee,
    });
  }
  return { frames, catches, period, truth };
}
