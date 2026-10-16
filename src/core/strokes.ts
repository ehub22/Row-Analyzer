import type { PoseFrame, Stroke, Metric, Point } from "./types";
import { evidence } from "./occlusion";
export const withheld = (unit: string, reason: string): Metric => ({
  value: null,
  confidence: 0,
  unit,
  reason,
});
// Stroke cycles from leg compression.
//
// The catch is maximum compression (seat forward, knees highest) and the
// finish proxy is the legs-flat end of the drive. Both used to be read from the
// hip-to-ankle distance. On-water footage almost never shows the ankles: the
// feet are strapped to the stretcher below the gunwale, so a whole-body model
// returns guesses for them (see core/occlusion.ts). Guessed ankles produced
// catches that were not there, missed ones that were, and — because guesses
// often clear the 0.3 threshold — still passed the evidence gate.
//
// The thigh carries the same events using only joints that stay above the
// gunwale. With the ankle fixed to the stretcher and the hip sliding on a
// horizontal track, the knee is the apex of a two-link leg: it rises as the
// seat comes forward (highest when the shins are vertical) and drops as the
// legs flatten. So the elevation of the hip→knee segment peaks around the
// catch and bottoms out when the legs are flat, without the ankle.
//
// This is image-space cycle timing, not a normalized joint-angle measurement:
// the angle is in the image plane, so its *values* depend on camera azimuth,
// but for a fixed camera its extrema (and therefore the timing) do not.

/** COCO-WholeBody hips. */
const HIPS: readonly number[] = [11, 12];
/** [hip, knee] of each leg in the COCO-WholeBody layout. */
const THIGHS: readonly (readonly [number, number])[] = [
  [11, 13],
  [12, 14],
];
/** Joints whose observation backs a cycle: hips and knees, never ankles. */
export const CYCLE_JOINTS: readonly number[] = [11, 12, 13, 14];
/** Degrees a catch must stand above the troughs around it. Smoothed keypoint
 * noise is about a degree; a full-slide stroke spans ~40–60°. */
export const MIN_PROMINENCE = 8;
/** Longest blind spot, in seconds, a cycle may span. Longer gaps split the
 * signal so a catch hidden inside the gap cannot halve the reported rate —
 * unless the hull explains the gap (see `MAX_TROUGH_GAP`). */
const MAX_GAP = 0.5;
/**
 * Longest blind spot, in seconds, bridged as a hull trough. With the legs
 * flat the knee sits near the saxboard, so in many side views it drops out
 * of sight for 40–50% of every cycle; splitting there would leave no
 * complete cycle at all. The hull can only hide a knee while it is low,
 * never at the catch, where it is highest. So a blind spot is bridged when
 * the signal on both sides lies in the bottom `TROUGH_FRACTION` of the
 * seat's range and a hip stays in view throughout: it cannot contain a catch.
 */
const MAX_TROUGH_GAP = 2.5;
const TROUGH_FRACTION = 0.35;
/** A cycle across a bridged blind spot may not run longer than this multiple
 * of the seat's median cycle: a doubled period means a catch went missing. */
const MAX_BRIDGED_PERIOD_RATIO = 1.5;
/** Half-width of the moving average, in seconds (±3 frames at 30 fps). */
const SMOOTH_HALF_WIDTH = 0.1;
/** Fraction of the cycle amplitude defining the catch and legs-flat bands. */
const EVENT_BAND = 0.1;

/**
 * Thigh elevation in degrees above the image horizontal (positive = knee
 * above hip), confidence-weighted across both legs; null without a confident
 * hip and knee on the same side. Uses |dx|, so it is independent of which way
 * the crew faces, of image scale and of camera translation.
 */
const seen = (p: Point | undefined): p is Point =>
  !!p && p.origin !== "missing" && p.confidence > 0.3;

export function thighElevation(f: PoseFrame): number | null {
  let sum = 0,
    weight = 0;
  for (const [hip, knee] of THIGHS) {
    const h = f.points[hip],
      k = f.points[knee];
    if (!seen(h) || !seen(k)) continue;
    const dx = Math.abs(k.x - h.x),
      dy = h.y - k.y; // image y grows downwards
    if (dx < 1e-6 && Math.abs(dy) < 1e-6) continue;
    const w = Math.min(h.confidence, k.confidence);
    sum += w * Math.atan2(dy, dx);
    weight += w;
  }
  return weight ? (sum / weight) * (180 / Math.PI) : null;
}

/**
 * Local maxima of `v` whose topographic prominence is at least
 * `minProminence`, at least `minDistance` seconds apart (the higher peak wins).
 *
 * Prominence — how far a peak stands above the lowest point separating it
 * from higher signal — is what makes the detector immune to plateaus. The
 * legs stay flat for a large part of every recovery at low rates; noise on
 * that plateau forms local extrema that a plain neighbour test accepts, and a
 * fixed merge window only hides them below one particular stroke rate.
 *
 * A side that runs off the end of the data does not lead to higher signal, so
 * it does not define the saddle; otherwise the first catch of every clip (or
 * of every run after a blind spot) would be judged by a partial rise.
 */
export function prominentPeaks(
  v: readonly number[],
  t: readonly number[],
  minProminence: number,
  minDistance: number,
): { index: number; prominence: number }[] {
  const candidates: number[] = [];
  for (let i = 1; i < v.length - 1; i++) {
    if (!(v[i] > v[i - 1])) continue;
    let j = i;
    while (j + 1 < v.length && v[j + 1] === v[i]) j++; // flat top
    if (j + 1 < v.length && v[j + 1] < v[i]) candidates.push((i + j) >> 1);
    i = j;
  }
  const prominence = (p: number) => {
    let left = v[p],
      right = v[p],
      leftCloses = false,
      rightCloses = false;
    for (let k = p - 1; k >= 0; k--) {
      if (v[k] > v[p]) {
        leftCloses = true;
        break;
      }
      left = Math.min(left, v[k]);
    }
    for (let k = p + 1; k < v.length; k++) {
      if (v[k] > v[p]) {
        rightCloses = true;
        break;
      }
      right = Math.min(right, v[k]);
    }
    const saddle =
      leftCloses && rightCloses
        ? Math.max(left, right)
        : leftCloses
          ? left
          : rightCloses
            ? right
            : Math.min(left, right); // highest peak of the run
    return v[p] - saddle;
  };
  const kept: { index: number; prominence: number }[] = [];
  for (const peak of candidates
    .map((index) => ({ index, prominence: prominence(index) }))
    .filter((p) => p.prominence >= minProminence)
    .sort((a, b) => v[b.index] - v[a.index]))
    if (kept.every((q) => Math.abs(t[q.index] - t[peak.index]) >= minDistance))
      kept.push(peak);
  return kept.sort((a, b) => a.index - b.index);
}

/**
 * Index nearest the middle (in time) of the contiguous run of samples around
 * `i` that satisfy `inside`, without leaving [first, last]. Null when the run
 * reaches either bound, i.e. when the band is cut off and its middle unknown.
 *
 * Rowing extrema are broad: the legs stay flat through the body swing, arm
 * draw and hands away, and near full compression the knee is highest when the
 * shin is vertical — before *and* after the catch if a rower compresses past
 * vertical. The single extreme sample therefore wanders across that region
 * with keypoint noise (±0.3 s at low rates in the simulation tests), while
 * the middle of the band within a fraction of the extreme stays put. It sits
 * slightly early of maximum compression, by the same amount for every seat,
 * so crew timing differences are preserved.
 */
function bandMiddle(
  v: readonly number[],
  t: readonly number[],
  i: number,
  inside: (x: number) => boolean,
  first: number,
  last: number,
): number | null {
  let a = i,
    b = i;
  while (a - 1 >= first && inside(v[a - 1])) a--;
  while (b + 1 <= last && inside(v[b + 1])) b++;
  // Reaching a bound means the band continues past what may be inspected.
  if (a === first || b === last) return null;
  const middle = (t[a] + t[b]) / 2;
  let best = a;
  for (let k = a; k <= b; k++)
    if (Math.abs(t[k] - middle) < Math.abs(t[best] - middle)) best = k;
  return best;
}

function smooth(
  raw: readonly (number | null)[],
  times: readonly number[],
  half: number,
): (number | null)[] {
  return raw.map((v, i) => {
    if (v === null) return null;
    let s = 0,
      n = 0;
    for (let k = i; k >= 0 && times[i] - times[k] <= half + 1e-9; k--)
      if (raw[k] !== null) ((s += raw[k]!), n++);
    for (
      let k = i + 1;
      k < raw.length && times[k] - times[i] <= half + 1e-9;
      k++
    )
      if (raw[k] !== null) ((s += raw[k]!), n++);
    return s / n;
  });
}

function quantile(sorted: readonly number[], q: number) {
  const i = (sorted.length - 1) * q,
    a = Math.floor(i),
    b = Math.ceil(i);
  return sorted[a] + (sorted[b] - sorted[a]) * (i - a);
}

/** A contiguous stretch of usable signal: indices into the seat's frames,
 * plus positions p where a hull trough is bridged between idx[p] and
 * idx[p + 1]. */
interface Run {
  idx: number[];
  bridges: number[];
}

export function segmentStrokes(frames: PoseFrame[]): Stroke[] {
  const result: Stroke[] = [];
  for (const seat of new Set(frames.map((f) => f.seat))) {
    const seq = frames
      .filter((f) => f.seat === seat)
      .sort((a, b) => a.time - b.time);
    if (seq.length < 10) continue;
    const times = seq.map((f) => f.time);
    const signal = smooth(seq.map(thighElevation), times, SMOOTH_HALF_WIDTH);
    const values = signal
      .filter((x): x is number => x !== null)
      .sort((a, b) => a - b);
    if (values.length < 10) continue;
    const lo = quantile(values, 0.05),
      troughLine = lo + TROUGH_FRACTION * (quantile(values, 0.95) - lo);
    const hipSeen = seq.map((f) => HIPS.some((j) => seen(f.points[j])));
    /** Whether the blind spot between usable samples i < k is a hull trough
     * (see MAX_TROUGH_GAP). Both ends have a hip, since they have signal. */
    const hullTrough = (i: number, k: number) => {
      if (times[k] - times[i] > MAX_TROUGH_GAP) return false;
      if (signal[i]! > troughLine || signal[k]! > troughLine) return false;
      let last = times[i];
      for (let m = i + 1; m <= k; m++) {
        if (!hipSeen[m]) continue;
        if (times[m] - last > MAX_GAP) return false;
        last = times[m];
      }
      return true;
    };
    const runs: Run[] = [];
    let run: Run = { idx: [], bridges: [] };
    for (let i = 0; i < seq.length; i++) {
      if (signal[i] === null) continue;
      const prev = run.idx[run.idx.length - 1];
      if (prev !== undefined && times[i] - times[prev] > MAX_GAP) {
        if (hullTrough(prev, i)) run.bridges.push(run.idx.length - 1);
        else {
          runs.push(run);
          run = { idx: [], bridges: [] };
        }
      }
      run.idx.push(i);
    }
    if (run.idx.length) runs.push(run);

    const cycles: { stroke: Omit<Stroke, "index">; bridged: boolean }[] = [];
    for (const { idx, bridges } of runs) {
      const v = idx.map((i) => signal[i]!),
        t = idx.map((i) => times[i]);
      // Each catch is timed at the middle of its near-peak band (see
      // bandMiddle); a band cut off by the edge of the run has no middle.
      const catches = prominentPeaks(v, t, MIN_PROMINENCE, 1).flatMap((p) => {
        const top = v[p.index] - EVENT_BAND * p.prominence;
        const i = bandMiddle(v, t, p.index, (x) => x >= top, 0, v.length - 1);
        return i === null ? [] : [i];
      });
      for (let k = 0; k < catches.length - 1; k++) {
        const a = catches[k],
          b = catches[k + 1],
          period = t[b] - t[a];
        if (period < 1 || period > 10) continue;
        let low = a + 1;
        for (let i = a + 1; i < b; i++) if (v[i] < v[low]) low = i;
        const amplitude = (v[a] + v[b]) / 2 - v[low];
        if (amplitude < MIN_PROMINENCE) continue;
        const inside = bridges.filter((p) => p >= a && p < b);
        // When the lowest visible sample borders a bridged blind spot, the
        // legs-flat phase itself was behind the gunwale. The knee left and
        // re-entered the view at the same height, so the middle of the blind
        // spot is the middle of the legs-flat band at gunwale level: the same
        // estimate as below, at a band edge the hull chose.
        const hidden = inside.find((p) => p === low || p + 1 === low);
        let finish: number;
        if (hidden !== undefined) {
          const middle = (t[hidden] + t[hidden + 1]) / 2;
          let m = idx[hidden];
          for (let j = idx[hidden]; j <= idx[hidden + 1]; j++)
            if (Math.abs(times[j] - middle) < Math.abs(times[m] - middle))
              m = j;
          finish = times[m];
        } else {
          const floor = v[low] + EVENT_BAND * amplitude;
          const f = bandMiddle(v, t, low, (x) => x <= floor, a + 1, b - 1);
          if (f === null) continue;
          finish = t[f];
        }
        const drive = finish - t[a],
          recovery = t[b] - finish;
        if (drive < 0.2 || recovery < 0.2) continue;
        const cycle = seq.slice(idx[a], idx[b] + 1),
          ev = evidence(
            cycle.flatMap((f) => CYCLE_JOINTS.map((j) => f.points[j])),
          ),
          confidence = Math.min(
            ev.confidence,
            cycle.reduce((s, f) => s + f.identityConfidence, 0) / cycle.length,
          );
        const reliable = ev.fraction >= 0.65 && confidence >= 0.5;
        cycles.push({
          bridged: inside.length > 0,
          stroke: {
            seat,
            catch: t[a],
            finish,
            nextCatch: t[b],
            rate: reliable
              ? { value: 60 / period, unit: "spm", confidence }
              : withheld("spm", "Insufficient observed hip/knee evidence"),
            ratio: reliable
              ? { value: drive / recovery, unit: ":1", confidence }
              : withheld(":1", "Insufficient observed hip/knee evidence"),
            observedFraction: ev.fraction,
            confidence,
          },
        });
      }
    }
    const periods = cycles
      .map((c) => c.stroke.nextCatch - c.stroke.catch)
      .sort((a, b) => a - b);
    const typical = periods.length ? quantile(periods, 0.5) : 0;
    let index = 0;
    for (const { stroke, bridged } of cycles) {
      const period = stroke.nextCatch - stroke.catch;
      if (bridged && period > MAX_BRIDGED_PERIOD_RATIO * typical) continue;
      result.push({ ...stroke, index: ++index });
    }
  }
  return result;
}
export function crewTiming(strokes: Stroke[], strokeSeat: number) {
  const ref = strokes.filter(
    (s) => s.seat === strokeSeat && s.rate.value !== null,
  );
  return strokes
    .filter((s) => s.seat !== strokeSeat && s.rate.value !== null)
    .flatMap((s) => {
      const r = ref.reduce<Stroke | undefined>(
        (best, v) =>
          !best || Math.abs(v.catch - s.catch) < Math.abs(best.catch - s.catch)
            ? v
            : best,
        undefined,
      );
      if (!r || Math.abs(r.catch - s.catch) > (r.nextCatch - r.catch) * 0.25)
        return [];
      return [
        {
          seat: s.seat,
          index: s.index,
          catchMs: (s.catch - r.catch) * 1000,
          finishMs: (s.finish - r.finish) * 1000,
          phase: ((s.catch - r.catch) / (r.nextCatch - r.catch)) * 360,
          confidence: Math.min(s.confidence, r.confidence),
        },
      ];
    });
}
export const mean = (values: (number | null)[]) => {
  const valid = values.filter(
    (v): v is number => v !== null && Number.isFinite(v),
  );
  return valid.length ? valid.reduce((a, b) => a + b) / valid.length : null;
};

/** A crew average cannot stand for the crew when only one seat has evidence. */
export function crewMean(
  strokes: Stroke[],
  seats: number,
  value: (stroke: Stroke) => number | null,
): number | null {
  const valid = strokes.filter((stroke) => value(stroke) !== null);
  if (new Set(valid.map((stroke) => stroke.seat)).size < Math.ceil(seats / 2))
    return null;
  return mean(valid.map(value));
}
