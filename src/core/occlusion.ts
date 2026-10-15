import type { Boat, PoseFrame, Correction, Point } from "./types";

// Expected occlusion.
//
// Most occlusion is transient (an oar, spray, a neighbouring rower) and the
// short gap filler below handles it. The feet are different: they are
// strapped to the stretcher inside the hull, below the gunwale, for the whole
// clip. A whole-body model still returns an answer for every one of its 133
// joints, so for hidden feet it returns a *guess*: usually a shin extrapolated
// straight down from the knee like a seated person, sometimes the handle, the
// rigger or the hull edge. Those guesses routinely clear the 0.3 observed
// threshold, so before this they were tracked as observed joints: they jumped
// between frames, were counted as evidence, drove stroke segmentation, and
// made the adaptive TTA gate pay four passes on nearly every crop.
//
// No amount of tracking or test-time augmentation turns a guess into an
// observation, so when the session says the feet are hidden the guesses are
// dropped at the source and the joints are marked missing with a reason.

/** COCO-WholeBody ankles (15, 16) and feet (17–22: big toe, small toe, heel). */
export const FOOT_JOINTS: readonly number[] = [15, 16, 17, 18, 19, 20, 21, 22];

/** Flag carried by every joint dropped as an expected occlusion. */
export const HIDDEN_FEET_FLAG =
  "Feet hidden by the hull (session setup); model output discarded";

/** Joints the camera cannot see for this boat setup. */
export function hiddenJoints(
  boat: Pick<Boat, "feetHidden">,
): readonly number[] {
  return boat.feetHidden === false ? [] : FOOT_JOINTS;
}

/**
 * Replaces model output for joints that cannot be visible with explicit
 * missing points. Manual points are kept: a person who can see the joint in a
 * frame outranks the session-wide expectation.
 */
export function suppressHiddenJoints(
  points: Point[],
  hidden: readonly number[],
): Point[] {
  if (!hidden.length) return points;
  const out = points.slice();
  for (const j of hidden) {
    if (!out[j] || out[j].origin === "manual") continue;
    out[j] = {
      x: 0,
      y: 0,
      confidence: 0,
      origin: "missing",
      flags: [HIDDEN_FEET_FLAG],
    };
  }
  return out;
}

export function recoverOcclusion(
  input: PoseFrame[],
  maxGap = 0.2,
): PoseFrame[] {
  const frames = structuredClone(input);
  const seats = [...new Set(frames.map((f) => f.seat))];
  for (const seat of seats) {
    const seq = frames
      .filter((f) => f.seat === seat)
      .sort((a, b) => a.time - b.time);
    for (let j = 0; j < 133; j++) {
      for (let i = 1; i < seq.length; i++) {
        const p = seq[i].points[j],
          prev = seq[i - 1].points[j];
        if (
          p.origin === "missing" ||
          prev.origin === "missing" ||
          p.origin === "manual"
        )
          continue;
        const dt = seq[i].time - seq[i - 1].time;
        const jump = Math.hypot(p.x - prev.x, p.y - prev.y);
        if (
          dt > 0 &&
          jump >
            Math.max(seq[i].box.height * 0.2, seq[i].box.height * dt * 4) &&
          p.confidence < 0.8
        ) {
          p.origin = "missing";
          p.confidence = 0;
          p.flags = ["Low-confidence jump; not bridged"];
        }
      }
      let a = 0;
      while (a < seq.length) {
        if (seq[a].points[j].origin !== "missing") {
          a++;
          continue;
        }
        let b = a;
        while (b < seq.length && seq[b].points[j].origin === "missing") b++;
        if (
          a > 0 &&
          b < seq.length &&
          seq[b].time - seq[a - 1].time <= maxGap &&
          !seq.slice(a, b).some((f) => f.points[j].flags?.length) &&
          seq.slice(a, b).every((f) => f.identityConfidence > 0.5)
        ) {
          const l = seq[a - 1],
            r = seq[b],
            p = l.points[j],
            q = r.points[j];
          if (Math.hypot(p.x - q.x, p.y - q.y) < r.box.height * 0.4)
            for (let k = a; k < b; k++) {
              const t = (seq[k].time - l.time) / (r.time - l.time);
              seq[k].points[j] = {
                x: p.x + (q.x - p.x) * t,
                y: p.y + (q.y - p.y) * t,
                confidence: Math.min(0.45, p.confidence, q.confidence),
                origin: "inferred",
                flags: ["Short bidirectional gap fill"],
              };
            }
        }
        a = b;
      }
    }
  }
  return frames;
}
export function applyCorrection(
  frames: PoseFrame[],
  correction: Correction,
  radius = 0.15,
): PoseFrame[] {
  return frames.map((f) => {
    if (f.seat !== correction.seat) return f;
    const dt = Math.abs(f.time - correction.time);
    if (dt > radius) return f;
    const points = [...f.points],
      old = points[correction.joint];
    if (dt < 1e-5)
      points[correction.joint] = {
        ...correction.after,
        origin: "manual",
        confidence: 1,
      };
    else if (
      old.origin !== "manual" &&
      old.origin !== "missing" &&
      !old.flags?.length
    ) {
      const w = 1 - dt / radius;
      points[correction.joint] = {
        ...old,
        x: old.x + (correction.after.x - correction.before.x) * w,
        y: old.y + (correction.after.y - correction.before.y) * w,
        origin: "inferred",
        confidence: Math.min(0.45, old.confidence),
        flags: ["Manual correction propagated; not ground truth"],
      };
    }
    return { ...f, points };
  });
}
export function evidence(points: Point[]) {
  const valid = points.filter((p) => p.origin !== "missing");
  return {
    fraction:
      points.filter((p) => p.origin === "observed" || p.origin === "manual")
        .length / Math.max(1, points.length),
    confidence:
      valid.reduce((s, p) => s + p.confidence, 0) / Math.max(1, points.length),
  };
}
