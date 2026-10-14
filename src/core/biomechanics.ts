import type { BiomechanicsRow, Metric, PoseFrame, Session } from "./types";
import { reliable } from "./calibration";
import { angle3D } from "./geometry";
import { mean } from "./strokes";
const evidence = (f: PoseFrame, joints: number[]) =>
  f.identityConfidence >= 0.6 && joints.every((j) => reliable(f.points[j]));
function angle(f: PoseFrame, joints: number[]) {
  if (!evidence(f, joints)) return null;
  const [a, b, c] = joints.map((j) => ({ ...f.points[j], z: 0 }));
  return angle3D(a, b, c);
}
function trunk(f: PoseFrame, side: number) {
  const a = 5 + side,
    b = 11 + side;
  return evidence(f, [a, b])
    ? (Math.atan2(
        f.points[a].x - f.points[b].x,
        f.points[b].y - f.points[a].y,
      ) *
        180) /
        Math.PI
    : null;
}
function metric(
  value: number | null,
  unit: string,
  confidence: number,
  reason = "Image-plane estimate, not viewpoint-normalized. Joint confidence is uncalibrated.",
): Metric {
  return {
    value,
    unit,
    confidence: value === null ? 0 : confidence,
    reason:
      value === null
        ? "Insufficient observed/manual evidence, identity confidence, or temporal coverage."
        : reason,
  };
}
const covered = (values: (number | null)[]) =>
  values.length >= 3 &&
  values.filter((v) => v !== null).length / values.length >= 0.8;
function peakVelocity(seq: PoseFrame[], values: (number | null)[]) {
  let best = -Infinity,
    time: number | null = null;
  for (let i = 1; i < seq.length; i++) {
    const dt = seq[i].time - seq[i - 1].time;
    if (values[i] === null || values[i - 1] === null || dt <= 0 || dt > 0.15)
      continue;
    const speed = Math.abs(values[i]! - values[i - 1]!) / dt;
    if (speed > best) {
      best = speed;
      time = (seq[i].time + seq[i - 1].time) / 2;
    }
  }
  return best > 10 ? time : null;
}
export function biomechanics(
  s: Pick<Session, "frames" | "strokes">,
): BiomechanicsRow[] {
  const seats = new Map<number, PoseFrame[]>();
  for (const f of s.frames) {
    const fs = seats.get(f.seat) || [];
    fs.push(f);
    seats.set(f.seat, fs);
  }
  return s.strokes.map((stroke) => {
    const seq = (seats.get(stroke.seat) || [])
      .filter((f) => f.time >= stroke.catch && f.time <= stroke.nextCatch)
      .sort((a, b) => a.time - b.time);
    const sides = [0, 1].map(
      (side) =>
        seq.filter((f) =>
          evidence(f, [
            5 + side,
            7 + side,
            9 + side,
            11 + side,
            13 + side,
            15 + side,
          ]),
        ).length,
    );
    const side = sides[0] >= sides[1] ? 0 : 1;
    const knee = seq.map((f) => angle(f, [11 + side, 13 + side, 15 + side]));
    const elbow = seq.map((f) => angle(f, [5 + side, 7 + side, 9 + side]));
    const body = seq.map((f) => trunk(f, side));
    const conf = Math.min(0.75, stroke.confidence);
    const near = (values: (number | null)[], t: number) => {
      let best = -1,
        d = 0.08;
      seq.forEach((f, i) => {
        if (Math.abs(f.time - t) < d && values[i] !== null) {
          d = Math.abs(f.time - t);
          best = i;
        }
      });
      return best < 0 ? null : values[best];
    };
    const range = (values: (number | null)[]) =>
      covered(values)
        ? Math.max(...values.filter((v): v is number => v !== null)) -
          Math.min(...values.filter((v): v is number => v !== null))
        : null;
    const drive = seq.filter((f) => f.time <= stroke.finish),
      knees = knee.slice(0, drive.length),
      trunks = body.slice(0, drive.length),
      arms = elbow.slice(0, drive.length);
    // Five-sample means reduce derivative noise; gaps are never imputed.
    const smooth = (a: (number | null)[]) =>
      a.map((_, i) => {
        const w = a.slice(Math.max(0, i - 2), i + 3);
        return w.every((v) => v !== null) ? mean(w) : null;
      });
    const peaks = [knees, trunks, arms].map((a) =>
      covered(a) ? peakVelocity(drive, smooth(a)) : null,
    );
    const lag = (a: number | null, b: number | null) =>
      a !== null && b !== null ? (b - a) * 1000 : null;
    const wrist = seq.filter((f) => evidence(f, [9 + side, 11 + side]));
    const heights = wrist.map(
      (f) => f.points[9 + side].y - f.points[11 + side].y,
    );
    const oars = seq.flatMap((f) =>
      (f.oars || [])
        .filter((o) => o.age >= 3 && o.confidence >= 0.5)
        .map(
          (o) =>
            (Math.atan2(o.end.y - o.start.y, o.end.x - o.start.x) * 180) /
            Math.PI,
        ),
    );
    // Oar lines have 180-degree orientation ambiguity. Do not report a misleading range across a wrap.
    const oarRange =
      oars.length >= seq.length * 0.8 &&
      oars.length >= 3 &&
      Math.max(...oars) - Math.min(...oars) < 90
        ? Math.max(...oars) - Math.min(...oars)
        : null;
    return {
      seat: stroke.seat,
      stroke: stroke.index,
      metrics: {
        "Catch knee flexion (2D)": metric(near(knee, stroke.catch), "°", conf),
        "Finish elbow angle (2D)": metric(
          near(elbow, stroke.finish),
          "°",
          conf,
        ),
        "Knee range (2D)": metric(range(knee), "°", conf),
        "Trunk range (2D)": metric(range(body), "°", conf),
        "Elbow range (2D)": metric(range(elbow), "°", conf),
        "Leg → trunk peak lag": metric(
          lag(peaks[0], peaks[1]),
          "ms",
          conf,
          "Smoothed projected angular-speed peaks; not force production or a sequencing diagnosis.",
        ),
        "Trunk → arm peak lag": metric(
          lag(peaks[1], peaks[2]),
          "ms",
          conf,
          "Smoothed projected angular-speed peaks; not force production or a sequencing diagnosis.",
        ),
        "Wrist path height relative to hip": metric(
          heights.length >= seq.length * 0.8 && heights.length >= 3
            ? Math.max(...heights) - Math.min(...heights)
            : null,
          "px",
          conf,
          "Wrist proxy, not observed handle height. Translation-compensated, not perspective/scale compensated.",
        ),
        "Candidate shaft angle range (2D)": metric(
          oarRange,
          "°",
          Math.min(conf, 0.5),
          "Experimental image line near wrist; may not be an oar. No blade entry/release is inferred.",
        ),
      },
    };
  });
}
